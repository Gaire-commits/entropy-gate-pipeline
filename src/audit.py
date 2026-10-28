"""Data audit measurements, read-only: nothing here fits a model or chooses a setting.

- Sessions rebuilt from the cached 5-minute bars on the NYSE calendar (src/nyse.py): a half-day ends at
  13:00, and bars after it are counted apart instead of joining the session. Each symbol-session keeps
  its status and the reason when it is not usable; nothing is dropped from the timeline, and an overnight
  return records how many sessions it spans.
- Every bar with an impossible value, and every non-finite ratio, located.
- Dollar volume, per session and as a trailing past-only median.
- Vendor daily bars against the rebuilt sessions, after checking the two share an adjustment basis.
- Security identity and corporate actions for gaps and large moves.

Classifications are rules, stated where they are applied. A case a rule cannot settle goes to the
unresolved list with the reason, rather than into a class.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import nyse

EASTERN = "America/New_York"
OPEN, FULL_CLOSE, EARLY_CLOSE = nyse.OPEN, nyse.FULL_CLOSE, nyse.EARLY_CLOSE
BAR = pd.Timedelta(minutes=5)
TOL_BP = 0.5          # two prices "match" within half a basis point
JUMP = 0.25           # a large move: |log return| above this (about +28% / -22%)
LONG_GAP = 20         # sessions; a gap this long is checked for a change of issuer
LOG4 = 4 * np.log(2)
SYMBOL_FIELDS = ["symbol", "old_symbol", "new_symbol", "source_symbol", "acquirer_symbol", "acquiree_symbol",
                 "alternate_symbol"]
DATE_FIELDS = ["ex_date", "process_date", "record_date", "payable_date", "effective_date"]


# ---------------------------------------------------------------- reading and per-symbol tables

def read_bars(path: Path) -> pd.DataFrame | None:
    """Cached bars as saved, indexed by naive New York bar start and sorted."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    ts = df.index.get_level_values("timestamp") if isinstance(df.index, pd.MultiIndex) else df.index
    ts = pd.DatetimeIndex(ts)
    ts = (ts.tz_convert(EASTERN) if ts.tz is not None else ts.tz_localize("UTC").tz_convert(EASTERN)).tz_localize(None)
    out = df.reset_index(drop=True)
    out.index = ts
    return out.sort_index(kind="stable")


def bar_problems(bars: pd.DataFrame, symbol: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Every bar with an impossible or misplaced value, with the reasons."""
    ts = bars.index
    o, h, lo, c = (bars[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    v = bars["volume"].to_numpy(np.float64)
    price = np.column_stack([o, h, lo, c])
    finite = np.isfinite(price)
    tod = ts - ts.normalize()
    tol = 1e-9 * np.abs(np.where(np.isfinite(h), h, 0))
    checks = {
        "non-finite price": ~finite.all(axis=1),
        "zero or negative price": (np.where(finite, price, 1.0) <= 0).any(axis=1),
        "high below low": h < lo,
        "open or close outside high-low": (np.fmax(o, c) > h + tol) | (np.fmin(o, c) < lo - tol),
        "bad volume": ~np.isfinite(v) | (v < 0),
        "not on the 5-minute grid": np.asarray((tod % BAR) != pd.Timedelta(0)),
        "outside 09:30-16:00": np.asarray((tod < OPEN) | (tod >= FULL_CLOSE)),
        "duplicate timestamp": np.asarray(ts.duplicated(keep=False)),
        "not an NYSE session": ~np.asarray(ts.normalize().isin(calendar)),
    }
    bad = np.zeros(len(ts), dtype=bool)
    for flag in checks.values():
        bad |= flag
    if not bad.any():
        return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume", "reason"])
    reason = [";".join(name for name, flag in checks.items() if flag[i]) for i in np.flatnonzero(bad)]
    return pd.DataFrame({"symbol": symbol, "timestamp": ts[bad], "open": o[bad], "high": h[bad], "low": lo[bad],
                         "close": c[bad], "volume": v[bad], "reason": reason})


def session_table(bars: pd.DataFrame, symbol: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per NYSE session from the symbol's first to its last day with bars.

    Session bars: 09:30 to the close (16:00, or 13:00 on a half-day), excluding impossible bars. Also kept:
    the clock-window rebuild the first audit used (09:30-16:00 whatever the calendar), the bars after a
    half-day's close, and the raw extremes including impossible bars (what the first audit's ratios saw).
    status: ok (at least 90% of the session's bars), sparse, or no_bars (a session with no bar at all)."""
    ts = bars.index
    day = ts.normalize()
    tod = ts - day
    o, h, lo, c = (bars[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    v = bars["volume"].to_numpy(np.float64)
    n = bars["trade_count"].to_numpy(np.float64) if "trade_count" in bars else np.full(len(ts), np.nan)
    vwap = bars["vwap"].to_numpy(np.float64) if "vwap" in bars else np.full(len(ts), np.nan)
    typical = (h + lo + c) / 3
    price_for_dv = np.where(np.isfinite(vwap) & (vwap > 0), vwap, typical)
    half = np.asarray(day.isin(nyse.HALF_DAYS))
    close_t = np.where(half, EARLY_CLOSE.to_timedelta64(), FULL_CLOSE.to_timedelta64())
    tod64 = np.asarray(tod.to_numpy())
    finite = np.isfinite(np.column_stack([o, h, lo, c])).all(axis=1)
    valid = finite & (np.fmin(np.fmin(o, h), np.fmin(lo, c)) > 0) & (h >= lo) & np.isfinite(v) & (v >= 0)
    clock = (tod64 >= OPEN.to_timedelta64()) & (tod64 < FULL_CLOSE.to_timedelta64())
    in_session = clock & (tod64 < close_t)
    post = clock & half & ~in_session
    minutes = tod64.astype("timedelta64[m]").astype(np.int64)
    f = pd.DataFrame({"day": day, "tod": minutes, "o": o, "h": h, "l": lo, "c": c, "v": v, "n": n,
                      "dv": v * price_for_dv})

    s = f[in_session & valid].groupby("day")
    out = pd.DataFrame({
        "open": s["o"].first(), "high": s["h"].max(), "low": s["l"].min(), "close": s["c"].last(),
        "volume": s["v"].sum(), "trade_count": s["n"].sum(min_count=1), "dollar_volume": s["dv"].sum(),
        "bars": s.size(), "first_bar": s["tod"].min(), "last_bar": s["tod"].max()})
    close_min = np.where(half, 13 * 60, 16 * 60) - 5
    out["open_bar"] = f[in_session & valid & (minutes == 9 * 60 + 30)].groupby("day").size().reindex(out.index).fillna(0) > 0
    out["close_bar"] = f[in_session & valid & (minutes == close_min)].groupby("day").size().reindex(out.index).fillna(0) > 0
    out["zero_volume_bars"] = f[in_session & (v == 0)].groupby("day").size().reindex(out.index).fillna(0)
    k = f[clock & valid].groupby("day")
    clock_cols = pd.DataFrame({"high_clock": k["h"].max(), "low_clock": k["l"].min(), "close_clock": k["c"].last(),
                               "volume_clock": k["v"].sum(), "bars_clock": k.size(), "last_bar_clock": k["tod"].max()})
    p = f[post & valid].groupby("day")
    post_cols = pd.DataFrame({"bars_after_close": p.size(), "volume_after_close": p["v"].sum()})
    r = f[clock].groupby("day")
    raw_cols = pd.DataFrame({"high_raw": r["h"].max(), "low_raw": r["l"].min(),
                             "invalid_bars": f[clock & ~valid].groupby("day").size(),
                             "bar_1255": f[clock & (minutes == 12 * 60 + 55) & np.isfinite(c)].groupby("day").size() > 0,
                             "bar_1555": f[clock & (minutes == 15 * 60 + 55) & np.isfinite(c)].groupby("day").size() > 0})
    days_with_bars = pd.DatetimeIndex(sorted(set(day[clock])))
    if not len(days_with_bars):
        return pd.DataFrame()
    span = calendar[(calendar >= days_with_bars.min()) & (calendar <= days_with_bars.max())]
    index = span.union(days_with_bars)
    out = pd.concat([out, clock_cols, post_cols, raw_cols], axis=1).reindex(index)
    for col in ("bars", "bars_clock", "bars_after_close", "invalid_bars", "zero_volume_bars"):
        out[col] = out[col].fillna(0).astype(np.int32)
    for col in ("open_bar", "close_bar", "bar_1255", "bar_1555"):
        out[col] = out[col].astype("boolean").fillna(False).astype(bool)
    out["volume_after_close"] = out["volume_after_close"].fillna(0.0)
    out.index.name = "date"
    out = out.reset_index()
    out.insert(0, "symbol", symbol)
    out["half_day"] = np.asarray(out["date"].isin(nyse.HALF_DAYS))
    out["nyse_session"] = np.asarray(out["date"].isin(calendar))
    out["slots"] = np.where(out["half_day"], 42, 78).astype(np.int16)
    out["status"] = np.select([out["bars"] == 0, out["bars"] < 0.9 * out["slots"]], ["no_bars", "sparse"], "ok")
    return out


def add_sequence(table: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Previous close and how many sessions back it is, per symbol, never compressing the timeline: an
    overnight return over a missing session says so (gap_sessions > 1) instead of looking like one night."""
    t = table.sort_values(["symbol", "date"]).reset_index(drop=True)
    pos = calendar.get_indexer(t["date"])
    t["session_index"] = pos
    has = t["bars"] > 0
    prev_close = t["close"].where(has).groupby(t["symbol"]).shift()
    prev_pos = pd.Series(np.where(has, pos, np.nan)).groupby(t["symbol"]).shift()
    # carry the last session that had bars forward over sessions without
    t["prev_close"] = prev_close.groupby(t["symbol"]).ffill()
    t["prev_session_index"] = prev_pos.groupby(t["symbol"]).ffill()
    t["gap_sessions"] = np.where(has & t["prev_session_index"].notna(), pos - t["prev_session_index"], np.nan)
    t["prev_half_day"] = t["half_day"].where(has).groupby(t["symbol"]).shift().groupby(t["symbol"]).ffill()
    return t


# ---------------------------------------------------------------- calendar, half-days, coverage

def calendar_check(table: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict:
    with_bars = table[table["bars"] > 0]
    per_day = with_bars.groupby("date")["symbol"].nunique()
    universe = table["symbol"].nunique()
    data_days = pd.DatetimeIndex(per_day[per_day >= 0.5 * universe].index)
    lo, hi = data_days.min(), data_days.max()
    nyse_days = calendar[(calendar >= lo) & (calendar <= hi)]
    extra = table[~table["nyse_session"] & (table["bars"] > 0)]
    return {"first": lo, "last": hi, "sessions_nyse": len(nyse_days), "sessions_data": len(data_days),
            "nyse_not_in_data": [d.date() for d in nyse_days.difference(data_days)],
            "data_not_nyse": [d.date() for d in data_days.difference(nyse_days)],
            "bars_on_non_sessions": extra[["symbol", "date", "bars"]],
            "per_day": per_day}


def half_day_check(table: pd.DataFrame) -> dict:
    t = table[table["bars"] > 0]
    half = t[t["half_day"]]
    by_day = half.groupby("date").agg(
        symbols=("symbol", "nunique"),
        with_bars_after_13=("bars_after_close", lambda x: int((x > 0).sum())),
        median_after_13_volume_share=("volume_after_close", "sum"),
        median_last_bar=("last_bar", "median"),
        first_audit_called_early=("last_bar_clock", lambda x: int((x < 13 * 60).sum())))
    share = (half["volume_after_close"] / (half["volume"] + half["volume_after_close"])).groupby(half["date"]).median()
    by_day["median_after_13_volume_share"] = share
    full = t[~t["half_day"] & t["nyse_session"]]
    early_end = full.assign(e=full["last_bar"] < 15 * 60).groupby("date")["e"].mean()
    no_open = t[t["nyse_session"]].groupby("date")["open_bar"].mean()
    return {"half_days": by_day.reset_index(),
            "unlisted_early_close_candidates": early_end[early_end >= 0.5].rename("share_last_bar_before_15h").reset_index(),
            "sessions_missing_0930_bar": no_open[no_open < 0.5].rename("share_with_0930_bar").reset_index()}


def coverage(table: pd.DataFrame, by: str = "year") -> pd.DataFrame:
    t = table[table["nyse_session"]].copy()
    t["year"] = t["date"].dt.year
    t["month"] = t["date"].dt.to_period("M").astype(str)
    t["day_type"] = np.where(t["half_day"], "half", "full")
    has = t["bars"] > 0
    t["with_bars"] = has
    t["ok"] = t["status"] == "ok"
    t["missing_open"] = has & ~t["open_bar"]
    t["missing_close"] = has & ~t["close_bar"]
    first_early = t["bars_clock"].gt(0) & (t["last_bar_clock"] < 13 * 60)
    t["first_audit_early"] = first_early
    t["first_audit_missing_close"] = t["bars_clock"].gt(0) & ~np.where(first_early, t["bar_1255"], t["bar_1555"])
    g = t.groupby([by, "day_type"])
    out = g.agg(sessions=("date", "nunique"), symbol_sessions=("symbol", "size"), with_bars=("with_bars", "sum"),
                ok=("ok", "sum"), bars=("bars", "sum"), slots=("slots", "sum"), missing_open=("missing_open", "sum"),
                missing_close=("missing_close", "sum"), first_audit_early=("first_audit_early", "sum"),
                first_audit_missing_close=("first_audit_missing_close", "sum"))
    out["slot_share"] = out["bars"] / out["slots"]
    out["ok_share"] = out["ok"] / out["symbol_sessions"]
    return out.drop(columns=["bars", "slots"]).reset_index()


def coverage_change(cov_year: pd.DataFrame, year: int = 2026) -> dict:
    """Is a year's higher coverage the half-days (it had none) or something in the feed? Full days only."""
    full = cov_year[cov_year["day_type"] == "full"].set_index("year")
    half = cov_year[cov_year["day_type"] == "half"].set_index("year")
    before = full.loc[full.index < year]
    plain = lambda s: {int(k): int(v) for k, v in s.items()}  # noqa: E731  (JSON wants plain keys)
    return {"year": year,
            "full_day_slot_share": float(full.loc[year, "slot_share"]) if year in full.index else np.nan,
            "full_day_slot_share_before": float(before["slot_share"].mean()) if len(before) else np.nan,
            "full_day_missing_close_rate": float(full.loc[year, "missing_close"] / full.loc[year, "with_bars"])
            if year in full.index else np.nan,
            "full_day_missing_close_rate_before": float((before["missing_close"] / before["with_bars"]).mean())
            if len(before) else np.nan,
            "half_days_by_year": plain(half["sessions"]),
            "first_audit_missing_close_on_half_days": plain(half["first_audit_missing_close"]),
            "first_audit_missing_close_on_full_days": plain(full["first_audit_missing_close"])}


# ---------------------------------------------------------------- dollar volume and invalid values

def dollar_volume(table: pd.DataFrame, window: int = 63, min_obs: int = 20) -> tuple[pd.Series, pd.DataFrame]:
    """Trailing median dollar volume per symbol-session from earlier usable sessions only (the session
    itself is never in its own window), and the yearly distribution of each symbol's median."""
    t = table.sort_values(["symbol", "date"])
    usable = t["dollar_volume"].where(t["status"] == "ok")
    trailing = usable.groupby(t["symbol"]).transform(
        lambda x: x.shift().rolling(window, min_periods=min_obs).median())
    ok = t[t["status"] == "ok"].assign(year=lambda x: x["date"].dt.year)
    med = ok.groupby(["year", "symbol"])["dollar_volume"].median()
    rows = []
    for year, s in med.groupby(level=0):
        row = {"year": year, "symbols": len(s)}
        row.update({f"p{q}": float(s.quantile(q / 100)) for q in (1, 5, 10, 25, 50)})
        row.update({f"below_{int(x / 1e6)}m": int((s < x).sum()) for x in (5e6, 1e7, 2.5e7, 5e7, 1e8)})
        rows.append(row)
    return trailing.reindex(table.index), pd.DataFrame(rows)


def first_audit_nonfinite(sip: pd.DataFrame, iex: pd.DataFrame) -> pd.DataFrame:
    """Every symbol-day where the first audit's IEX/SIP Parkinson ratio was not a finite number, with the
    extremes behind it. The first audit took highs and lows over all bars in 09:30-16:00, impossible
    values included, so this uses the raw extremes."""
    m = sip[sip["bars_clock"] > 0][["symbol", "date", "high_raw", "low_raw"]].merge(
        iex[iex["bars_clock"] > 0][["symbol", "date", "high_raw", "low_raw"]], on=["symbol", "date"],
        suffixes=("_sip", "_iex"))
    with np.errstate(divide="ignore", invalid="ignore"):
        pk_s = np.log(m["high_raw_sip"] / m["low_raw_sip"]) ** 2
        pk_i = np.log(m["high_raw_iex"] / m["low_raw_iex"]) ** 2
        ratio = pk_i / pk_s
    bad = ~np.isfinite(ratio)
    out = m[bad].copy()

    def why(r):
        parts = []
        for feed in ("sip", "iex"):
            hi, lo = r[f"high_raw_{feed}"], r[f"low_raw_{feed}"]
            if not (np.isfinite(hi) and np.isfinite(lo)):
                parts.append(f"{feed}: missing extreme")
            elif lo <= 0 or hi <= 0:
                parts.append(f"{feed}: zero or negative price")
            elif hi == lo:
                parts.append(f"{feed}: high equals low (zero range)")
        return "; ".join(parts) or "other"
    out["reason"] = out.apply(why, axis=1) if len(out) else []
    return out


def nonfinite_sessions(table: pd.DataFrame, feed: str) -> pd.DataFrame:
    """Rebuilt sessions (impossible bars excluded) whose log ratios are not finite."""
    t = table[table["bars"] > 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        checks = {"overnight": np.log(t["open"] / t["prev_close"]), "range": np.log(t["high"] / t["low"]),
                  "open_to_close": np.log(t["close"] / t["open"])}
    rows = []
    for name, x in checks.items():
        bad = ~np.isfinite(x)
        if name == "overnight":
            bad &= t["prev_close"].notna()  # the first session of a symbol has no overnight return to check
        rows.append(t.loc[bad, ["symbol", "date", "open", "high", "low", "close", "prev_close"]].assign(ratio=name))
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "feed", feed)
    return out


# ---------------------------------------------------------------- vendor daily bars against rebuilt sessions

def basis(m: pd.DataFrame) -> pd.DataFrame:
    """Per symbol: the log ratio of vendor close to rebuilt close. "same" basis within 0.1%; "constant
    offset" when it is off by a steady amount (e.g. a split after the cache was fetched); otherwise
    "varies over time" (adjustment history or symbol mapping differs), which is left unresolved."""
    r = np.log(m["close_v"] / m["close"])
    g = r.groupby(m["symbol"])
    out = pd.DataFrame({"median": g.median(), "iqr": g.quantile(0.75) - g.quantile(0.25), "n": g.size()})
    out["basis"] = np.select([out["median"].abs() <= 1e-3, out["iqr"] <= 5e-4], ["same", "constant offset"],
                             "varies over time")
    return out


def basis_breaks(m: pd.DataFrame, symbols, jump: float = 0.01) -> pd.DataFrame:
    """Where the vendor/rebuilt ratio steps, for symbols whose basis varies."""
    rows = []
    for sym, g in m[m["symbol"].isin(symbols)].sort_values("date").groupby("symbol"):
        r = np.log(g["close_v"] / g["close"]).to_numpy()
        steps = np.flatnonzero(np.abs(np.diff(r)) > jump)
        rows += [{"symbol": sym, "date": g["date"].iloc[i + 1], "log_ratio_before": r[i], "log_ratio_after": r[i + 1]}
                 for i in steps]
    return pd.DataFrame(rows, columns=["symbol", "date", "log_ratio_before", "log_ratio_after"])


def vendor_comparison(table: pd.DataFrame, vendor: pd.DataFrame, tol_bp: float = TOL_BP) -> dict:
    v = vendor[["symbol", "date", "open", "high", "low", "close", "volume", "vwap"]]
    m = table[table["bars"] > 0].merge(v, on=["symbol", "date"], suffixes=("", "_v"))
    b = basis(m)
    same = b.index[b["basis"] == "same"]
    a = m[m["symbol"].isin(same)].copy()
    a["year"] = a["date"].dt.year
    a["day_type"] = np.where(a["half_day"], "half", "full")
    for f in ("open", "high", "low", "close"):
        a[f"d_{f}"] = 1e4 * np.log(a[f"{f}_v"] / a[f])
    for f in ("high", "low", "close"):
        a[f"d_{f}_clock"] = 1e4 * np.log(a[f"{f}_v"] / a[f"{f}_clock"])
    a["volume_ratio"] = a["volume_v"] / a["volume"]
    a["volume_ratio_clock"] = a["volume_v"] / a["volume_clock"]

    def summarize(g):
        row = {"n": len(g)}
        for f in ("open", "high", "low", "close"):
            d = g[f"d_{f}"]
            row[f"{f}_match"] = float((d.abs() <= tol_bp).mean())
            row[f"{f}_median_abs_bp"] = float(d.abs().median())
            row[f"{f}_p99_abs_bp"] = float(d.abs().quantile(0.99))
        row["vendor_high_above"] = float((g["d_high"] > tol_bp).mean())
        row["vendor_high_below"] = float((g["d_high"] < -tol_bp).mean())
        row["vendor_low_below"] = float((g["d_low"] < -tol_bp).mean())
        row["vendor_low_above"] = float((g["d_low"] > tol_bp).mean())
        row["volume_ratio_median"] = float(g["volume_ratio"].median())
        row["volume_ratio_p5"] = float(g["volume_ratio"].quantile(0.05))
        row["volume_ratio_p95"] = float(g["volume_ratio"].quantile(0.95))
        for f in ("high", "low", "close"):
            row[f"{f}_match_clock_rebuild"] = float((g[f"d_{f}_clock"].abs() <= tol_bp).mean())
        row["volume_ratio_clock_median"] = float(g["volume_ratio_clock"].median())
        return pd.Series(row)

    def per_group(keys):
        rows = []
        for k, g in a.groupby(keys):
            k = k if isinstance(k, tuple) else (k,)
            rows.append({**dict(zip(keys, k)), **summarize(g).to_dict()})
        return pd.DataFrame(rows)

    by_year = per_group(["year", "day_type"])
    overall = per_group(["day_type"])

    pk_v = np.log(a["high_v"] / a["low_v"]) ** 2 / LOG4
    pk_r = np.log(a["high"] / a["low"]) ** 2 / LOG4
    with np.errstate(divide="ignore", invalid="ignore"):
        range_ratio = np.sqrt(pk_v / pk_r)
    range_effect = pd.DataFrame({"day_type": a["day_type"], "pk_v": pk_v, "pk_r": pk_r,
                                 "log_range_ratio": np.log(range_ratio)}).groupby("day_type").agg(
        mean_parkinson_vendor=("pk_v", "mean"), mean_parkinson_rebuilt=("pk_r", "mean"),
        median_log_range_ratio=("log_range_ratio", "median"),
        share_range_differs_10pct=("log_range_ratio", lambda x: float((x.abs() > 0.1).mean())))
    range_effect["mean_ratio"] = range_effect["mean_parkinson_vendor"] / range_effect["mean_parkinson_rebuilt"]

    seq = a.sort_values(["symbol", "date"])
    prev_v = seq.groupby("symbol")["close_v"].shift()
    consecutive = seq["gap_sessions"] == 1
    ovn = pd.DataFrame({"year": seq["year"], "vendor": np.log(seq["open_v"] / prev_v) ** 2,
                        "rebuilt": np.log(seq["open"] / seq["prev_close"]) ** 2})[consecutive & prev_v.notna()]
    overnight_effect = ovn.groupby("year").mean()
    overnight_effect["ratio"] = overnight_effect["vendor"] / overnight_effect["rebuilt"]

    a["worst_bp"] = a[["d_open", "d_high", "d_low", "d_close"]].abs().max(axis=1)
    worst = a.nlargest(200, "worst_bp")[["symbol", "date", "half_day", "status", "bars", "open", "open_v", "high",
                                         "high_v", "low", "low_v", "close", "close_v", "d_open", "d_high", "d_low",
                                         "d_close", "volume_ratio"]]
    return {"basis": b.reset_index(), "basis_breaks": basis_breaks(m, b.index[b["basis"] == "varies over time"]),
            "by_year": by_year, "overall": overall, "range_effect": range_effect.reset_index(),
            "overnight_effect": overnight_effect.reset_index(), "worst": worst, "aligned_rows": len(a),
            "merged_rows": len(m)}


# ---------------------------------------------------------------- the session-definition sample

def sessions_from_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Rebuild each symbol-day of a raw bar frame (columns symbol, t, open..vwap) under three session
    definitions: the clock window 09:30-16:00 (what the cache keeps), the calendar session (09:30 to the
    close), and the whole New York day (extended hours included)."""
    b = bars.copy()
    ts = b["t"].dt.tz_convert(EASTERN).dt.tz_localize(None)
    b["date"], b["tod"] = ts.dt.normalize(), ts - ts.dt.normalize()
    close = np.where(b["date"].isin(nyse.HALF_DAYS), EARLY_CLOSE.to_timedelta64(), FULL_CLOSE.to_timedelta64())
    tod = b["tod"].to_numpy()
    masks = {"clock": (tod >= OPEN.to_timedelta64()) & (tod < FULL_CLOSE.to_timedelta64()),
             "calendar": (tod >= OPEN.to_timedelta64()) & (tod < close),
             "whole_day": np.ones(len(b), dtype=bool)}
    frames = []
    for name, mask in masks.items():
        g = b[mask].sort_values("t").groupby(["symbol", "date"])
        frames.append(pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                                    "close": g["close"].last(), "volume": g["volume"].sum()}).add_suffix(f"_{name}"))
    return pd.concat(frames, axis=1).reset_index()


def session_definition_check(ext: pd.DataFrame, vendor: pd.DataFrame, table: pd.DataFrame,
                             tol_bp: float = TOL_BP) -> dict:
    """Which session definition the vendor's daily bar follows, field by field, and whether the 5-minute
    bars fetched again today agree with the cache inside the session."""
    s = sessions_from_bars(ext)
    m = s.merge(vendor[["symbol", "date", "open", "high", "low", "close", "volume"]], on=["symbol", "date"])
    m["day_type"] = np.where(m["date"].isin(nyse.HALF_DAYS), "half", "full")
    rows = []
    for (day_type), g in m.groupby("day_type"):
        for name in ("clock", "calendar", "whole_day"):
            row = {"day_type": day_type, "definition": name, "n": len(g)}
            for f in ("open", "high", "low", "close"):
                d = 1e4 * np.log(g[f] / g[f"{f}_{name}"])
                row[f"{f}_match"] = float((d.abs() <= tol_bp).mean())
            row["volume_ratio_median"] = float((g["volume"] / g[f"volume_{name}"]).median())
            row["volume_exact_share"] = float((np.abs(g["volume"] / g[f"volume_{name}"] - 1) < 1e-6).mean())
            rows.append(row)
    cache = table[table["bars"] > 0][["symbol", "date", "open", "high", "low", "close", "volume"]]
    c = s.merge(cache, on=["symbol", "date"])
    agree = {f: float((1e4 * np.log(c[f] / c[f"{f}_calendar"])).abs().le(tol_bp).mean()) for f in
             ("open", "high", "low", "close")} if len(c) else {}
    agree["volume_exact_share"] = float((np.abs(c["volume"] / c["volume_calendar"] - 1) < 1e-6).mean()) if len(c) else np.nan
    agree["n"] = len(c)
    return {"definitions": pd.DataFrame(rows), "cache_vs_refetch": agree}


def auction_check(auctions: pd.DataFrame, vendor: pd.DataFrame, table: pd.DataFrame, tol_bp: float = TOL_BP) -> dict:
    """For the sample symbols on every day: does the vendor's open (close) equal an opening (closing)
    auction print, and which condition and exchange; and the same for the rebuilt open and close."""
    if auctions.empty:
        return {"summary": pd.DataFrame(), "by_print": pd.DataFrame()}
    a = auctions.copy()
    a["date"] = pd.to_datetime(a["date"]).dt.normalize()
    v = vendor[["symbol", "date", "open", "close"]].rename(columns={"open": "vendor_open", "close": "vendor_close"})
    r = table[table["bars"] > 0][["symbol", "date", "open", "close"]].rename(
        columns={"open": "rebuilt_open", "close": "rebuilt_close"})
    m = a.merge(v, on=["symbol", "date"], how="left").merge(r, on=["symbol", "date"], how="left")
    for who in ("vendor", "rebuilt"):
        ref = np.where(m["side"] == "open", m[f"{who}_open"], m[f"{who}_close"])
        m[f"{who}_equal"] = np.abs(1e4 * np.log(m["price"] / ref)) <= tol_bp
    by_print = m.groupby(["side", "condition", "exchange"]).agg(prints=("price", "size"),
                                                                 vendor_equal=("vendor_equal", "mean"),
                                                                 rebuilt_equal=("rebuilt_equal", "mean")).reset_index()
    days = m.groupby(["symbol", "date", "side"]).agg(vendor_any=("vendor_equal", "any"),
                                                     rebuilt_any=("rebuilt_equal", "any")).reset_index()
    summary = days.groupby("side")[["vendor_any", "rebuilt_any"]].mean().reset_index()
    summary["symbol_days"] = days.groupby("side").size().to_numpy()
    return {"summary": summary, "by_print": by_print}


def trades_check(trades: pd.DataFrame, vendor: pd.DataFrame, table: pd.DataFrame, tol_bp: float = TOL_BP) -> pd.DataFrame:
    """For each trade-sample symbol-day: the first trades at or after 09:30 and the auction-flagged prints,
    and which of them the vendor's and the rebuilt open and close equal."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    ts = t["t"].dt.tz_convert(EASTERN).dt.tz_localize(None)
    t["date"], t["clock"] = ts.dt.normalize(), ts - ts.dt.normalize()
    v = vendor.set_index(["symbol", "date"])
    r = table[table["bars"] > 0].set_index(["symbol", "date"])
    rows = []
    for (sym, day, window), g in t.sort_values("t").groupby(["symbol", "date", "window"]):
        field = "open" if window == "open" else "close"
        close = nyse.close_time(day)
        g = g[g["clock"] >= OPEN] if window == "open" else g
        if g.empty:
            continue
        cands = {"first trade from 09:30": g.iloc[0] if window == "open" else None,
                 "last trade before the close": g[g["clock"] < close].iloc[-1] if window == "close" and
                 (g["clock"] < close).any() else None}
        for code in (("O", "Q") if window == "open" else ("6", "M")):
            flagged = g[g["conditions"].str.split().apply(lambda c: code in c)]
            cands[f"first print with condition {code}"] = flagged.iloc[0] if len(flagged) else None
        vend = v[field].get((sym, day), np.nan)
        rebuilt = r[field].get((sym, day), np.nan)
        for name, trade in cands.items():
            if trade is None:
                continue
            rows.append({"symbol": sym, "date": day, "window": window, "candidate": name, "price": trade["price"],
                         "conditions": trade["conditions"], "exchange": trade["exchange"],
                         "time": trade["clock"], "vendor": vend, "rebuilt": rebuilt,
                         "vendor_equal": abs(1e4 * np.log(trade["price"] / vend)) <= tol_bp if vend == vend else None,
                         "rebuilt_equal": abs(1e4 * np.log(trade["price"] / rebuilt)) <= tol_bp
                         if rebuilt == rebuilt else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- identity and corporate actions

def normalize_actions(ca: pd.DataFrame) -> pd.DataFrame:
    ca = ca.copy()
    for col in SYMBOL_FIELDS + ["kind", "id"]:
        if col not in ca.columns:
            ca[col] = None
    for col in DATE_FIELDS:
        ca[col] = pd.to_datetime(ca[col], errors="coerce") if col in ca.columns else pd.NaT
    for col in ("rate", "old_rate", "new_rate", "source_rate", "cash_rate", "acquirer_rate", "acquiree_rate"):
        ca[col] = pd.to_numeric(ca[col], errors="coerce") if col in ca.columns else np.nan
    if "special" not in ca.columns:
        ca["special"] = None
    ca["when"] = ca["ex_date"].fillna(ca["effective_date"]).fillna(ca["process_date"])
    return ca


def aliases(ca: pd.DataFrame, symbol: str, depth: int = 4) -> set[str]:
    """The ticker and every earlier ticker that name-change records lead to it from. Alpaca's default
    symbol mapping returns those tickers' bars under today's symbol, so their records belong here too."""
    names = {symbol}
    renames = ca[ca["kind"].astype(str).str.contains("name_change")]
    for _ in range(depth):
        older = set(renames.loc[renames["new_symbol"].isin(names), "old_symbol"].dropna()) - names
        if not older:
            break
        names |= older
    return names


def actions_for(ca: pd.DataFrame, symbol, lo=None, hi=None) -> pd.DataFrame:
    """Records naming this ticker (or any ticker in a set) in any symbol field, optionally in a window."""
    names = {symbol} if isinstance(symbol, str) else set(symbol)
    hit = np.zeros(len(ca), dtype=bool)
    for col in SYMBOL_FIELDS:
        hit |= ca[col].isin(names).to_numpy()
    out = ca[hit]
    if lo is not None:
        out = out[(out["when"] >= pd.Timestamp(lo)) & (out["when"] <= pd.Timestamp(hi))]
    return out


def describe(ca_rows: pd.DataFrame) -> str:
    parts = []
    for _, r in ca_rows.iterrows():
        kind = str(r["kind"])
        when = r["when"].date() if pd.notna(r["when"]) else "?"
        if "name_change" in kind:
            parts.append(f"name change {r['old_symbol']}->{r['new_symbol']} {when}")
        elif "spin_off" in kind:
            parts.append(f"spin-off {r['source_symbol']}->{r['new_symbol']} {when}")
        elif "merger" in kind:
            parts.append(f"{kind[:-1].replace('_', ' ')} {r['acquiree_symbol']} into {r['acquirer_symbol']} {when}")
        elif "split" in kind:
            parts.append(f"{kind[:-1].replace('_', ' ')} {r['old_rate']:g}->{r['new_rate']:g} {when}")
        elif "dividend" in kind:
            special = " special" if str(r.get("special")).lower() == "true" else ""
            parts.append(f"cash dividend{special} {r['rate']:g} {when}" if pd.notna(r["rate"]) else f"{kind} {when}")
        else:
            parts.append(f"{kind} {when}")
    return "; ".join(parts)


def gaps(table: pd.DataFrame, min_len: int = 5) -> pd.DataFrame:
    """Runs of NYSE sessions with no bar at all, inside each symbol's span."""
    rows = []
    t = table[table["nyse_session"]].sort_values(["symbol", "date"])
    for sym, g in t.groupby("symbol"):
        missing = (g["bars"] == 0).to_numpy()
        if not missing.any():
            continue
        edges = np.flatnonzero(np.diff(np.r_[0, missing.astype(np.int8), 0]))
        dates, opens, closes = g["date"].to_numpy(), g["open"].to_numpy(), g["close"].to_numpy()
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s < min_len:
                continue
            before = closes[s - 1] if s > 0 else np.nan
            after = opens[e] if e < len(g) else np.nan
            rows.append({"symbol": sym, "first_missing": pd.Timestamp(dates[s]), "last_missing": pd.Timestamp(dates[e - 1]),
                         "sessions": int(e - s), "close_before": before, "open_after": after,
                         "log_move_across": np.log(after / before) if before > 0 and after > 0 else np.nan})
    return pd.DataFrame(rows, columns=["symbol", "first_missing", "last_missing", "sessions", "close_before",
                                       "open_after", "log_move_across"])


def mapping_splices(mapped: pd.DataFrame, unmapped: pd.DataFrame) -> pd.DataFrame:
    """Per symbol, the days Alpaca's symbol mapping adds (bars under an earlier ticker of the same issuer,
    by Alpaca's records) and the days only the unmapped query has."""
    a = mapped.groupby("symbol")["date"].apply(set)
    b = unmapped.groupby("symbol")["date"].apply(set)
    rows = []
    for sym in sorted(set(a.index) | set(b.index)):
        sa, sb = a.get(sym, set()), b.get(sym, set())
        only_m, only_u = sorted(sa - sb), sorted(sb - sa)
        if only_m or only_u:
            rows.append({"symbol": sym, "mapped_only_days": len(only_m),
                         "mapped_only_first": only_m[0] if only_m else pd.NaT,
                         "mapped_only_last": only_m[-1] if only_m else pd.NaT, "unmapped_only_days": len(only_u),
                         "unmapped_only_first": only_u[0] if only_u else pd.NaT,
                         "unmapped_only_last": only_u[-1] if only_u else pd.NaT})
    return pd.DataFrame(rows)


def classify_gaps(gap_rows: pd.DataFrame, ca: pd.DataFrame, unmapped: pd.DataFrame) -> pd.DataFrame:
    """Rule, for gaps of LONG_GAP sessions or more: an exit record for an earlier issuer under this ticker
    near the gap's start (the ticker as the old symbol of a name change, the target of a merger, or a
    worthless removal) makes it "ticker reused by a different issuer"; bars under the bare ticker during
    the gap make it "missing from the cache"; anything else is unresolved. Shorter gaps are listed only."""
    out = gap_rows.copy()
    classes, evidence = [], []
    have = unmapped.groupby("symbol")["date"].apply(lambda d: pd.DatetimeIndex(d)) if len(unmapped) else pd.Series(dtype=object)
    for _, g in out.iterrows():
        sym, lo, hi = g["symbol"], g["first_missing"], g["last_missing"]
        names = aliases(ca, sym)
        if g["sessions"] < LONG_GAP:
            classes.append("short gap (listed)")
            evidence.append(describe(actions_for(ca, names, lo - pd.Timedelta(days=10), hi + pd.Timedelta(days=10))))
            continue
        near_start = actions_for(ca, names, lo - pd.Timedelta(days=15), lo + pd.Timedelta(days=45))
        # an exit under this very ticker; a rename into it (same issuer, per Alpaca's mapping) is not an exit
        exits = near_start[(near_start["old_symbol"] == sym) | (near_start["acquiree_symbol"] == sym) |
                           (near_start["kind"].astype(str).str.contains("worthless") & (near_start["symbol"] == sym))]
        near_end = actions_for(ca, names, hi - pd.Timedelta(days=45), hi + pd.Timedelta(days=15))
        entries = near_end[near_end["new_symbol"] == sym]
        days = have.get(sym, pd.DatetimeIndex([])) if len(have) else pd.DatetimeIndex([])
        during = int(((days >= lo) & (days <= hi)).sum()) if len(days) else 0
        if len(exits):
            classes.append("ticker reused by a different issuer")
        elif during:
            classes.append("missing from the cache (the bare ticker has bars)")
        else:
            classes.append("unresolved: no corporate action explains the gap")
        evidence.append("; ".join(x for x in (describe(exits), describe(entries),
                                              f"bare-ticker bars during gap: {during}") if x))
    out["class"], out["evidence"] = classes, evidence
    return out


def large_moves(table: pd.DataFrame, threshold: float = JUMP) -> pd.DataFrame:
    t = table[table["bars"] > 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ovn = np.log(t["open"] / t["prev_close"])
        ses = np.log(t["close"] / t["open"])
    cols = ["symbol", "date", "gap_sessions", "half_day", "prev_half_day", "prev_close", "open", "close", "status"]
    rows = [t.loc[ovn.abs() > threshold, cols].assign(kind="overnight", move=ovn[ovn.abs() > threshold]),
            t.loc[ses.abs() > threshold, cols].assign(kind="session", move=ses[ses.abs() > threshold])]
    return pd.concat(rows, ignore_index=True).sort_values(["symbol", "date", "kind"]).reset_index(drop=True)


def vendor_move(vendor: pd.DataFrame, kind: str) -> pd.Series:
    v = vendor.sort_values(["symbol", "date"])
    if kind == "overnight":
        move = np.log(v["open"] / v.groupby("symbol")["close"].shift())
    else:
        move = np.log(v["close"] / v["open"])
    return pd.Series(move.to_numpy(), index=pd.MultiIndex.from_frame(v[["symbol", "date"]]))


def classify_moves(moves: pd.DataFrame, settings: dict[str, pd.DataFrame], ca: pd.DataFrame,
                   gap_rows: pd.DataFrame, splices: pd.DataFrame) -> pd.DataFrame:
    """Each large move against the corporate-action records and the vendor's adjustment settings.

    Corporate-action records are looked up under the ticker and every earlier ticker that name changes
    lead to it from (aliases), since Alpaca's symbol mapping returns those tickers' bars.
    Rules, in order (the first that applies):
      identity change     an overnight move on the first session after a gap of LONG_GAP+ sessions, or
                          within 5 days after the last day a symbol mapping adds;
      unadjusted split    a split record on that day and a move within 0.05 of the split ratio;
      cash distribution   a cash dividend on that day whose amount (against the unadjusted previous close)
                          explains the move within 0.05, or which the split+dividend setting takes out
                          (reported: whether it does);
      spin-off            a spin-off record with this symbol as source on that day (reported: whether the
                          "all" setting, which includes spin-offs, takes it out);
      market move         no relevant record (anything on that day except an ordinary dividend under 2% of
                          the price, or an identity record in the window), every adjustment setting shows
                          the same move as the split-only data, and that agrees with the cache: kept as a
                          genuine event, never excluded;
      unresolved          anything else, with the reason.
    """
    moves = moves.copy()
    vm = {name: {k: vendor_move(df, k) for k in ("overnight", "session")} for name, df in settings.items()}
    raw = settings.get("raw")
    raw_prev = (raw.sort_values(["symbol", "date"]).assign(p=lambda x: x.groupby("symbol")["close"].shift())
                .set_index(["symbol", "date"])["p"]) if raw is not None else pd.Series(dtype=float)
    identity_kinds = "name_change|merger|worthless|reorganization"
    gap_ends = {}
    for _, g in gap_rows[gap_rows["sessions"] >= LONG_GAP].iterrows():
        gap_ends.setdefault(g["symbol"], []).append(g["last_missing"])
    splice_ends = {r["symbol"]: r["mapped_only_last"] for _, r in splices.iterrows()
                   if pd.notna(r.get("mapped_only_last"))} if len(splices) else {}
    out_rows = []
    for _, mv in moves.iterrows():
        sym, day, kind, move = mv["symbol"], mv["date"], mv["kind"], mv["move"]
        got = {name: vm[name][kind].get((sym, day), np.nan) for name in vm}
        names = aliases(ca, sym)
        recs = actions_for(ca, names, day - pd.Timedelta(days=5), day + pd.Timedelta(days=1))
        on_day = recs[recs["when"] == day]
        prev_raw = raw_prev.get((sym, day), np.nan)
        is_div = on_day["kind"].astype(str).str.contains("cash_dividend")
        small = is_div & (on_day["special"].astype(str).str.lower() != "true") & \
            (on_day["rate"] / prev_raw < 0.02 if prev_raw == prev_raw else False)
        relevant = pd.concat([on_day[~small], recs[recs["kind"].astype(str).str.contains(identity_kinds)]])
        relevant = relevant[~relevant.index.duplicated()]
        cls, reason = None, ""
        after_gap = any(0 <= (day - e).days <= 10 for e in gap_ends.get(sym, []))
        splice = splice_ends.get(sym)
        if kind == "overnight" and after_gap:
            cls = "identity change (after a long gap)"
        elif kind == "overnight" and splice is not None and 0 <= (day - splice).days <= 5:
            cls = "identity change (symbol mapping boundary)"
        if cls is None and len(on_day):
            splits = on_day[on_day["kind"].astype(str).str.contains("split")]
            divs = on_day[on_day["kind"].astype(str).str.contains("cash_dividend")]
            spins = on_day[on_day["kind"].astype(str).str.contains("spin_off") & on_day["source_symbol"].isin(names)]
            if len(splits) and kind == "overnight":
                ratio = splits["new_rate"].iloc[0] / splits["old_rate"].iloc[0]
                if np.isfinite(ratio) and ratio > 0 and abs(move + np.log(ratio)) < 0.05:
                    cls = "unadjusted split"
            if cls is None and len(divs) and kind == "overnight":
                prev = prev_raw  # dividend amounts are in the dollars of the day, so compare with unadjusted prices
                rate = divs["rate"].sum()
                implied = np.log(1 - rate / prev) if prev == prev and 0 < rate < prev else np.nan
                removed = abs(got.get("split+dividend", np.nan)) < JUMP
                if (np.isfinite(implied) and abs(move - implied) < 0.05) or removed:
                    cls = "cash distribution"
                    reason = f"removed by split+dividend: {bool(removed)}"
            if cls is None and len(spins):
                cls = "spin-off"
                reason = f"removed by all: {bool(abs(got.get('all', np.nan)) < JUMP)}"
        if cls is None:
            same = [abs(got[name] - got.get("split", np.nan)) < 0.01 for name in got if np.isfinite(got[name])]
            agrees = abs(got.get("split", np.nan) - move) < 0.02
            if relevant.empty and same and all(same) and agrees:
                cls = "market move (no corporate action found)"
            else:
                cls = "unresolved"
                if not np.isfinite(got.get("split", np.nan)):
                    reason = "no vendor bar that day"
                elif not agrees:
                    reason = "vendor split-adjusted move differs from the cache's"
                elif len(relevant):
                    reason = "corporate action nearby that the rules do not match: " + describe(relevant)
                else:
                    reason = "an adjustment setting changes the move but no corporate action record"
        out_rows.append({**mv.to_dict(), **{f"vendor_{k}": got[k] for k in got}, "class": cls, "reason": reason,
                         "records": describe(recs)})
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------- reporting helpers

def md(df: pd.DataFrame, floats: str = ".3f", max_rows: int | None = None) -> str:
    """A markdown table that keeps integers as integers."""
    if df is None or len(df) == 0:
        return "(none)"
    df = df.head(max_rows) if max_rows else df
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (bool, np.bool_)):
                cells.append(str(bool(v)))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):,}")
            elif isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else (f"{int(v):,}" if float(v).is_integer() and abs(v) >= 1000
                                                           else format(v, floats)))
            elif isinstance(v, pd.Timestamp):
                cells.append(str(v.date()) if v == v.normalize() else str(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str))
