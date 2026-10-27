#!/usr/bin/env python3
"""Read-only audit of the cached 5-minute bars (no model, no setting chosen, nothing fetched).

    python scripts/data_audit.py --sip data/bars_sip --iex data/bars --universe universe/sp500.csv

Reads the caches and writes outputs/data_audit/report.md plus the tables behind it (CSV). Covers: coverage by
year, session conventions, missing days and slots, zero-volume days, suspected unadjusted splits and bad
bars, daily bars rebuilt from the 5-minute bars, the SIP bars against the IEX bars on the days both cover, how
those differences move a range-based variance estimate, and how many symbols exist each year.
Daily bars are not in the cache, so rebuilt-versus-published daily bars cannot be compared here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT  # noqa: F401  (puts the repo root on the path)

from src.data import cache_path, load_symbol, read_meta, resolve_universe

LOG4 = 4 * np.log(2)
SPLIT_RATIOS = (2, 3, 4, 5, 6, 8, 10, 15, 20, 1.5)


def daily_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per session, rebuilt from the gridded 5-minute bars of one symbol."""
    ts = df.index.get_level_values("timestamp")
    d = df.reset_index(drop=True)
    d["date"] = ts.tz_localize(None).normalize() if ts.tz is not None else ts.normalize()
    g = d.groupby("date")
    slots = g.size()
    last_bar = slots - 1
    out = pd.DataFrame({
        "slots": slots, "bars": g["close"].count(),
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(min_count=1),
        "zero_volume_bars": (d["volume"] == 0).groupby(d["date"]).sum(),
        "open_bar": d["close"].notna().groupby(d["date"]).first(),
        "close_bar": (d["close"].notna() & (d["bar_index"] == d["date"].map(last_bar))).groupby(d["date"]).any(),
    })
    bad = (d["high"] < d[["open", "close", "low"]].max(axis=1) - 1e-9) | (d["low"] > d[["open", "close", "high"]].min(axis=1) + 1e-9)
    out["inconsistent_bars"] = bad.groupby(d["date"]).sum()
    out["max_bar_move"] = (np.log(d["close"] / d["open"]).abs()).groupby(d["date"]).max()
    out["early"] = slots < 78
    return out


def load_feed(cache_dir: str, symbols: list[str], timeframe: str = "5Min") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily tables for every cached symbol (long), and one meta row per symbol."""
    frames, metas = [], []
    for s in symbols:
        if not cache_path(cache_dir, s, timeframe).exists():
            continue
        df = load_symbol(cache_dir, s, timeframe)
        m = read_meta(cache_dir, s, timeframe) or {}
        metas.append({"symbol": s, **{k: m.get(k) for k in ("feed", "adjustment", "start", "end")}})
        if df is not None and not df.empty:
            frames.append(daily_table(df).assign(symbol=s).reset_index())
    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return daily, pd.DataFrame(metas)


def calendar(daily: pd.DataFrame, share: float = 0.5) -> pd.DatetimeIndex:
    """Sessions on which at least `share` of the symbols have a bar: the market's trading days."""
    n = daily[daily["bars"] > 0].groupby("date")["symbol"].nunique()
    return pd.DatetimeIndex(n[n >= share * daily["symbol"].nunique()].index)


def coverage(daily: pd.DataFrame, cal: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    d = daily.assign(year=daily["date"].dt.year)
    d["usable"] = d["bars"] >= 0.9 * d["slots"]
    first_last = d.groupby("symbol")["date"].agg(first="min", last="max", days="count")
    by_year = d.groupby("year").agg(
        symbols=("symbol", "nunique"), stock_days=("symbol", "size"), usable_stock_days=("usable", "sum"),
        slot_share=("bars", "sum"), slots=("slots", "sum"), zero_volume_days=("volume", lambda v: int(((v == 0) | v.isna()).sum())),
        early_closes=("early", "sum"), missing_open_bar=("open_bar", lambda v: int((~v).sum())),
        missing_close_bar=("close_bar", lambda v: int((~v).sum())), inconsistent_bars=("inconsistent_bars", "sum"))
    by_year["slot_share"] = by_year["slot_share"] / by_year["slots"]
    by_year = by_year.drop(columns="slots")
    by_year["market_days"] = pd.Series(cal.year).value_counts().sort_index().reindex(by_year.index)
    full = d[d["usable"]].groupby(["symbol", "year"]).size().unstack(fill_value=0)
    per = (full.ge(0.95 * by_year["market_days"].reindex(full.columns).to_numpy())).sum()
    by_year["symbols_with_95pct_of_days"] = per.reindex(by_year.index)
    gaps = []
    for s, g in d.groupby("symbol"):
        have = set(g.loc[g["bars"] > 0, "date"])
        span = cal[(cal >= g["date"].min()) & (cal <= g["date"].max())]
        miss = [x for x in span if x not in have]
        gaps.append({"symbol": s, "missing_market_days": len(miss), "span_days": len(span),
                     "longest_run": _longest_run(span, set(miss))})
    return {"by_year": by_year.reset_index(), "symbols": first_last.reset_index().merge(pd.DataFrame(gaps), on="symbol")}


def _longest_run(span: pd.DatetimeIndex, miss: set) -> int:
    best = run = 0
    for x in span:
        run = run + 1 if x in miss else 0
        best = max(best, run)
    return best


def jumps(daily: pd.DataFrame, level: float = 0.25) -> pd.DataFrame:
    """Overnight gaps (previous close to the open) and in-session moves beyond `level` in log terms, flagged
    when they sit on a common split ratio, which would mean a split the adjustment missed."""
    d = daily.sort_values(["symbol", "date"]).copy()
    prev = d.groupby("symbol")["close"].shift()
    d["overnight"] = np.log(d["open"] / prev)
    d["session"] = np.log(d["close"] / d["open"])
    out = d[(d["overnight"].abs() > level) | (d["session"].abs() > level)].copy()

    def ratio(x):
        r = np.exp(abs(x))
        return next((k for k in SPLIT_RATIOS if abs(r / k - 1) < 0.04), np.nan) if np.isfinite(r) else np.nan
    out["split_like_ratio"] = out["overnight"].map(ratio)
    return out[["symbol", "date", "overnight", "session", "split_like_ratio"]]


def variance_parts(daily: pd.DataFrame) -> pd.DataFrame:
    """Per year, mean squared overnight gap, Parkinson range variance and open-to-close variance (descriptive:
    it shows how much of a day's variance the range estimator would leave to the gap)."""
    d = daily.sort_values(["symbol", "date"]).copy()
    d["ovn"] = np.log(d["open"] / d.groupby("symbol")["close"].shift()) ** 2
    d["park"] = np.log(d["high"] / d["low"]) ** 2 / LOG4
    d["oc"] = np.log(d["close"] / d["open"]) ** 2
    d = d[(d["bars"] >= 0.9 * d["slots"]) & np.isfinite(d["ovn"]) & np.isfinite(d["park"])]
    t = d.groupby(d["date"].dt.year)[["ovn", "park", "oc"]].mean()
    t["overnight_share_of_ovn_plus_park"] = t["ovn"] / (t["ovn"] + t["park"])
    return t.reset_index().rename(columns={"date": "year"})


def feed_comparison(sip: pd.DataFrame, iex: pd.DataFrame) -> dict:
    """SIP against IEX on the same symbol-days, both rebuilt from 5-minute bars."""
    m = sip.merge(iex, on=["symbol", "date"], suffixes=("_s", "_i"))
    m = m[(m["bars_s"] >= 0.9 * m["slots_s"]) & (m["bars_i"] > 0)]
    if m.empty:
        return {}
    lg = lambda a, b: np.log(m[a] / m[b])
    pk = lambda sfx: np.log(m[f"high_{sfx}"] / m[f"low_{sfx}"]) ** 2 / LOG4
    return {
        "symbol_days": len(m), "first": m["date"].min(), "last": m["date"].max(),
        "median_abs_log_open_diff_bps": float(1e4 * lg("open_i", "open_s").abs().median()),
        "median_abs_log_close_diff_bps": float(1e4 * lg("close_i", "close_s").abs().median()),
        "iex_high_above_sip_share": float((m["high_i"] > m["high_s"] * (1 + 1e-9)).mean()),
        "iex_low_below_sip_share": float((m["low_i"] < m["low_s"] * (1 - 1e-9)).mean()),
        "median_iex_to_sip_volume": float((m["volume_i"] / m["volume_s"]).median()),
        "median_iex_slot_share": float((m["bars_i"] / m["slots_i"]).median()),
        "mean_parkinson_ratio_iex_over_sip": float(pk("i").mean() / pk("s").mean()),
        "median_parkinson_ratio_iex_over_sip": float((pk("i") / pk("s")).replace([np.inf, -np.inf], np.nan).median()),
        "share_days_range_differs_over_10pct": float((np.abs(np.log(np.sqrt(pk("i")) / np.sqrt(pk("s")))) > 0.1).mean()),
    }


def md(df: pd.DataFrame, spec: str = ".3f") -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(format(v, spec) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                      else ("" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sip", default="data/bars_sip")
    ap.add_argument("--iex", default="data/bars")
    ap.add_argument("--universe", default="universe/sp500.csv")
    ap.add_argument("--out", default="outputs/data_audit")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    universe = resolve_universe(args.universe)
    t0 = time.time()
    sip, sip_meta = load_feed(args.sip, universe)
    if sip.empty:
        print(f"no cached SIP bars under {args.sip}")
        return 1
    print(f"SIP: {sip['symbol'].nunique()} of {len(universe)} symbols ({time.time() - t0:.0f}s)")
    iex, iex_meta = load_feed(args.iex, universe)
    print(f"IEX: {iex['symbol'].nunique() if len(iex) else 0} symbols ({time.time() - t0:.0f}s)")

    cal = calendar(sip)
    cov = coverage(sip, cal)
    jp = jumps(sip)
    vp = variance_parts(sip)
    cmp_ = feed_comparison(sip, iex) if len(iex) else {}
    cov["by_year"].to_csv(out / "coverage_by_year.csv", index=False)
    cov["symbols"].to_csv(out / "coverage_by_symbol.csv", index=False)
    jp.to_csv(out / "jumps.csv", index=False)
    vp.to_csv(out / "variance_parts.csv", index=False)
    sip_meta.to_csv(out / "sip_meta.csv", index=False)

    s = cov["symbols"]
    end = sip["date"].max()
    lines = ["# Data audit (read-only)", "",
             f"Universe file `{args.universe}`: {len(universe)} symbols. SIP cache `{args.sip}`: {sip['symbol'].nunique()} symbols, "
             f"{sip['date'].min().date()} to {end.date()}, {len(cal):,} market days (days with a bar for at least half the symbols).", "",
             "## 1. Coverage", "", "Feed and adjustment recorded in the cache's own metadata:", "",
             md(sip_meta.groupby(["feed", "adjustment", "start", "end"], dropna=False).size().rename("symbols").reset_index()), "",
             f"Symbols with a full history (first day within 5 market days of {cal.min().date()}): "
             f"{int((s['first'] <= cal[min(5, len(cal) - 1)]).sum())}. Symbols whose last day is more than 10 days before "
             f"{end.date()}: {int((s['last'] < end - pd.Timedelta(days=10)).sum())}. Symbols starting after "
             f"{(cal.min() + pd.Timedelta(days=370)).date()}: {int((s['first'] > cal.min() + pd.Timedelta(days=370)).sum())}.", "",
             md(cov["by_year"]), "",
             "## 2. Missingness", "",
             f"Missing market days per symbol (between its first and last day): median {s['missing_market_days'].median():.0f}, "
             f"90th percentile {s['missing_market_days'].quantile(.9):.0f}, maximum {s['missing_market_days'].max():.0f}; "
             f"longest run of missing days: median {s['longest_run'].median():.0f}, maximum {s['longest_run'].max():.0f}. "
             f"Symbols with a run of 10 or more missing market days: {int((s['longest_run'] >= 10).sum())}.", "",
             "Symbols with the most missing market days:", "",
             md(s.sort_values("missing_market_days", ascending=False).head(15)[["symbol", "first", "last", "missing_market_days", "longest_run"]]), "",
             "## 3. Suspected unadjusted splits and bad bars", "",
             f"Overnight gaps or in-session moves beyond 25% in log terms: {len(jp)} symbol-days across {jp['symbol'].nunique()} symbols; "
             f"{int(jp['split_like_ratio'].notna().sum())} of them sit within 4% of a common split ratio {SPLIT_RATIOS}.", "",
             md(jp[jp["split_like_ratio"].notna()].head(30)) if jp["split_like_ratio"].notna().any() else "None on a split ratio.", "",
             "## 4. Daily bars rebuilt from the 5-minute bars, and what a range estimator would see", "",
             "Open is the first bar's open, close the last bar's close, high and low the extremes over the session's bars; a missing "
             "bar contributes nothing. Mean squared overnight gap, Parkinson range variance and open-to-close variance per year "
             "(sessions with at least 90% of their bars):", "", md(vp, ".6f"), ""]
    if cmp_:
        lines += ["## 5. SIP against IEX on the same symbol-days", "",
                  "IEX prints only the trades routed to IEX, so its high is expected to be at or below SIP's and its low at or above.", "",
                  md(pd.DataFrame([{k: (str(v.date()) if isinstance(v, pd.Timestamp) else v) for k, v in cmp_.items()}]).T.reset_index()
                     .rename(columns={"index": "measure", 0: "value"}), ".4f"), ""]
    else:
        lines += ["## 5. SIP against IEX", "", "No IEX cache found, so the comparison was not run.", ""]
    lines += ["## Not done here", "",
              "- Daily bars published by the vendor are not in the cache, so the rebuilt daily bars cannot be compared with them.",
              "- Dividend adjustment: the cache is split-adjusted only, and no dividend dates are stored, so ex-dividend behaviour is not examined.",
              "- Halts and ticker changes or reuse cannot be told apart from missing data in this cache; the runs of missing days above are the closest evidence.",
              "- Which exchange or auction prints enter a bar's open and close is not recorded in the bars."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "run.json").write_text(json.dumps({"sip": args.sip, "iex": args.iex, "universe": args.universe, "symbols": int(sip["symbol"].nunique())}))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
