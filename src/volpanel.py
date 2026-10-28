"""The volatility project's stock-session panel: the variance target, why a value is missing, and causal eligibility.

Read-only and causal: nothing is fitted, and everything dated t uses only what was known at the close of t (the target
is the next session's). One row per symbol and NYSE session. Inputs:

  * Alpaca's vendor daily bars (scripts/fetch_reference.py): adjustment `all` (splits, cash dividends, spin-offs) for the
    open, high, low and close; `split` and `raw` to see where the vendor adjusts and the price actually quoted;
  * the audit's sessions.parquet (scripts/data_audit_v2.py): the 5-minute cache rebuilt on the NYSE calendar, used to
    cross-check the vendor's range and for the past-only in-session dollar volume;
  * Alpaca's corporate-action records, and registry/vol_registry.csv: identity changes and corporate actions that the
    records miss and that were checked against public records.

Definitions (RESULTS.md, "Variance target"), per session t of one security:

    r_on(t) = ln O(t) - ln C(t-1)                  overnight log return from the previous session's close
    rho(t)  = (ln H(t) - ln L(t))^2 / (4 ln 2)     Parkinson range variance of the regular session (a half-day ends 13:00),
                                                   from the quoted high and low: a log range does not depend on the
                                                   adjustment, and quoted prices are exact where adjusted ones are rounded
    V(t)    = r_on(t)^2 + rho(t)                   the primary target
    V_id(t) = rho(t)                               intraday only, a sensitivity target

No session is dropped to close a gap in the timeline: a missing value stays in its row with the reason it is missing,
and r_on is defined only against the immediately preceding NYSE session.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import nyse

PARKINSON = 1.0 / (4.0 * np.log(2.0))
SPLIT_KINDS = ("forward_splits", "reverse_splits", "unit_splits", "stock_dividends")

# Windows by target date (HEAT1's split, so each has the exposure already recorded in RESULTS.md); the session between
# two windows (2022-01-03, 2024-01-02) is an embargo, as in HEAT.
WINDOWS = {"development": ("2016-01-04", "2021-12-31"), "validation": ("2022-01-04", "2023-12-29"),
           "test": ("2024-01-03", "2026-10-12")}

# The order in which a session's missing value is explained; the first that applies is its reason.
RANGE_REASONS = ["symbol has no vendor bars", "outside the security's trading span", "no vendor bar",
                 "no trades (placeholder bar)", "invalid bar", "market-wide truncated day",
                 "range disagrees with the 5-minute cache"]
OVERNIGHT_REASONS = ["previous session has no valid bar", "previous session is a truncated day",
                     "first session of a segment (identity change)", "corporate action or adjustment step"]


@dataclass(frozen=True)
class Rules:
    """Every number here is fixed before any forecast is looked at; the audit's tables say what each one removes."""
    range_conflict: float = float(np.log(1.25))   # |ln(vendor range / 5-minute range)| above this on a complete day
    market_gap_ratio: float = 0.05                # median trade count over its trailing median, below this: truncated day
    market_gap_min_symbols: int = 50              # symbols that must have a trailing median for the date to be judged
    min_raw_close: float = 5.0                    # the price as quoted that day, in dollars
    min_dollar_volume: float = 10e6               # trailing 63-session median, in-session, past-only
    weekly: int = 5                               # HAR windows, in sessions, and how many must hold a value
    weekly_min: int = 4
    monthly: int = 22
    monthly_min: int = 18
    coverage: float = 0.9                         # share of the warm-up window that must hold a valid value
    step_split: float = 1e-3                      # a split-type adjustment step (log) above this marks an event session
    step_cash: float = 0.03                       # a dividend or spin-off adjustment step above this does too


def load_registry(path) -> pd.DataFrame:
    """registry/vol_registry.csv: symbol, date, effect (segment_break | overnight_unreliable), kind, status, evidence."""
    r = pd.read_csv(path)
    r["date"] = pd.to_datetime(r["date"])
    return r


def vendor_frame(raw: pd.DataFrame, split: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """One row per vendor daily bar (adjustment `all`), with the quoted close and the adjustment steps: how far the
    adjusted-to-raw ratio moves between a security's consecutive real bars. A step marks an ex-date of some kind."""
    keys = ["symbol", "date"]
    v = adj[keys + ["open", "high", "low", "close", "volume", "trade_count"]].copy()
    v = v.merge(raw[keys + ["high", "low", "close"]].rename(columns={"high": "raw_high", "low": "raw_low", "close": "raw_close"}),
                on=keys, how="left")
    v = v.merge(split[keys + ["close"]].rename(columns={"close": "split_close"}), on=keys, how="left")
    v["real"] = (v["volume"] > 0) & (v["trade_count"] > 0)
    v = v.sort_values(keys).reset_index(drop=True)
    r = v[v["real"]].copy()
    r["lf_split"] = np.log(r["split_close"] / r["raw_close"])
    r["lf_all"] = np.log(r["close"] / r["raw_close"])
    r["step_split"] = r.groupby("symbol")["lf_split"].diff()
    r["step_cash"] = r.groupby("symbol")["lf_all"].diff() - r["step_split"]
    return v.merge(r[keys + ["step_split", "step_cash"]], on=keys, how="left")


def market_gap_days(v: pd.DataFrame, ratio: float, min_symbols: int = 50) -> pd.Series:
    """Dates on which the cross-section's median trade count is under `ratio` of each stock's own trailing 63-bar median
    (past-only): the vendor's daily bar holds a handful of prints instead of the day. Known at the close of that day."""
    r = v[v["real"]].sort_values(["symbol", "date"])
    med = r.groupby("symbol")["trade_count"].transform(lambda x: x.shift(1).rolling(63, min_periods=40).median())
    d = (r["trade_count"] / med).groupby(r["date"]).agg(["median", "count"])
    return (d["median"] < ratio) & (d["count"] >= min_symbols)


def _first_session_on_or_after(cal: pd.DatetimeIndex, dates: pd.Series) -> pd.Series:
    """Each date moved to the first NYSE session on or after it (NaT past the end of the calendar)."""
    i = cal.searchsorted(pd.DatetimeIndex(dates))
    return pd.Series(pd.DatetimeIndex(cal[np.minimum(i, len(cal) - 1)]).where(i < len(cal)), index=dates.index)


def record_events(actions: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Sessions on which a corporate-action record says the quoted price moves for a mechanical reason: splits, stock
    dividends and spin-offs (the parent's ticker, and every later ticker of the same issuer: Alpaca's symbol mapping
    returns an earlier ticker's bars under today's). Cash dividends are not events: the vendor adjusts for them."""
    kinds = actions["kind"].astype(str)
    renames = actions[kinds.str.contains("name_change")]
    nxt = dict(zip(renames["old_symbol"], renames["new_symbol"]))
    rows = []
    for _, r in actions[kinds.isin(SPLIT_KINDS) | kinds.str.contains("spin_off")].iterrows():
        spin = "spin_off" in str(r["kind"])
        name = r["source_symbol"] if spin else r["symbol"]
        if pd.isna(name) or pd.isna(r["when"]):
            continue
        i = cal.searchsorted(r["when"])
        if i >= len(cal):
            continue
        label = f"record: {str(r['kind'])[:-1].replace('_', ' ')}"
        chain = [name]
        while nxt.get(chain[-1]) and nxt[chain[-1]] not in chain and len(chain) < 5:
            chain.append(nxt[chain[-1]])
        rows += [(s, cal[i], label) for s in chain]
    return pd.DataFrame(rows, columns=["symbol", "date", "source"])


def step_events(v: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    split = v["step_split"].abs() > rules.step_split
    cash = v["step_cash"].abs() > rules.step_cash
    out = v.loc[split | cash, ["symbol", "date"]].copy()
    out["source"] = np.where(split[split | cash], "vendor split-type adjustment", "vendor dividend or spin-off adjustment")
    return out


def event_sessions(actions: pd.DataFrame, v: pd.DataFrame, registry: pd.DataFrame, cal: pd.DatetimeIndex,
                   rules: Rules) -> pd.DataFrame:
    """Every (symbol, session) whose overnight return is not a clean overnight: a record, a vendor adjustment step, or a
    checked entry in the registry. Unique per symbol and session, with the first source that named it."""
    reg = registry[registry["effect"] == "overnight_unreliable"]
    reg = pd.DataFrame({"symbol": reg["symbol"], "date": _first_session_on_or_after(cal, reg["date"]),
                        "source": "registry: " + reg["kind"].astype(str)})
    parts = [x for x in (record_events(actions, cal), step_events(v, rules), reg) if len(x)]
    allv = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["symbol", "date", "source"])
    allv["date"] = pd.to_datetime(allv["date"])
    return allv.drop_duplicates(["symbol", "date"]).reset_index(drop=True)


def build_panel(sessions: pd.DataFrame, vendor: pd.DataFrame, actions: pd.DataFrame, registry: pd.DataFrame,
                symbols: list[str], rules: Rules = Rules(), start="2016-01-04", end="2026-10-12") -> pd.DataFrame:
    """The panel: every symbol on every NYSE session from `start` to `end`, with V, V_id, the close-to-close return, and the
    reason each is missing. `vendor` is vendor_frame(...); `sessions` the audit's sessions.parquet."""
    cal = nyse.sessions(start, end)
    p = pd.MultiIndex.from_product([sorted(symbols), cal], names=["symbol", "date"]).to_frame(index=False)
    p["half_day"] = np.asarray(nyse.is_half_day(p["date"]))
    v = vendor[(vendor["date"] >= cal[0]) & (vendor["date"] <= cal[-1])]
    p = p.merge(v, on=["symbol", "date"], how="left", indicator="has_row")
    p["has_row"] = p["has_row"] == "both"
    p["real"] = p["real"].eq(True)
    s = sessions[["symbol", "date", "bars", "slots", "high", "low", "dv_trailing63"]].rename(
        columns={"bars": "bars5", "slots": "slots5", "high": "high5", "low": "low5", "dv_trailing63": "dv63"})
    p = p.merge(s, on=["symbol", "date"], how="left")

    ohlc = ["open", "high", "low", "close"]
    finite = np.isfinite(p[ohlc + ["raw_high", "raw_low"]]).all(axis=1) & (p[ohlc + ["raw_high", "raw_low"]] > 0).all(axis=1)
    consistent = (p["high"] >= p[["open", "close", "low"]].max(axis=1)) & (p["low"] <= p[["open", "close", "high"]].min(axis=1)) \
        & (p["raw_high"] >= p["raw_low"])
    p["bar_ok"] = p["real"] & finite & consistent
    first = p[p["real"]].groupby("symbol")["date"].min()
    last = p[p["real"]].groupby("symbol")["date"].max()
    p["has_vendor"] = p["symbol"].isin(first.index)
    p["in_span"] = (p["date"] >= p["symbol"].map(first)) & (p["date"] <= p["symbol"].map(last))
    p["trunc"] = p["date"].map(market_gap_days(vendor, rules.market_gap_ratio, rules.market_gap_min_symbols)).eq(True)

    lr_v, lr_5 = np.log(p["raw_high"] / p["raw_low"]), np.log(p["high5"] / p["low5"])
    comparable = p["bar_ok"] & (lr_v > 0) & (lr_5 > 0) & (p["bars5"] > 0)
    p["range_d"] = np.where(comparable, np.log(lr_v / lr_5), np.nan)
    complete = p["bars5"] >= p["slots5"]
    # a complete 5-minute day must agree with the vendor's range; an incomplete one can only show the vendor too narrow
    p["range_conflict"] = comparable & np.where(complete, p["range_d"].abs() > rules.range_conflict,
                                                p["range_d"] < -rules.range_conflict)

    cal_idx = pd.DatetimeIndex(cal)
    brk = registry[registry["effect"] == "segment_break"]
    brk_dates = set(zip(brk["symbol"], _first_session_on_or_after(cal_idx, brk["date"])))
    p["is_break"] = [(a, b) in brk_dates for a, b in zip(p["symbol"], p["date"])]
    p["segment"] = p.groupby("symbol")["is_break"].cumsum()
    ev = event_sessions(actions, vendor, registry, cal_idx, rules)
    p = p.merge(ev.rename(columns={"source": "event_source"}), on=["symbol", "date"], how="left")
    p = p.sort_values(["symbol", "date"]).reset_index(drop=True)          # the shifts below are by session
    p["event"] = p["event_source"].notna()

    g = p.groupby("symbol", sort=False)
    p["prev_close"], p["prev_ok"], p["prev_trunc"] = g["close"].shift(1), g["bar_ok"].shift(1).eq(True), \
        g["trunc"].shift(1).eq(True)

    range_conds = [~p["has_vendor"], ~p["in_span"], ~p["has_row"], ~p["real"], ~p["bar_ok"], p["trunc"], p["range_conflict"]]
    night_conds = [~p["prev_ok"], p["prev_trunc"], p["is_break"], p["event"]]
    p["reason_id"] = np.select(range_conds, RANGE_REASONS, "")
    p["reason"] = np.select(range_conds + night_conds, RANGE_REASONS + OVERNIGHT_REASONS, "")
    p["rho"] = np.where(p["reason_id"] == "", PARKINSON * lr_v ** 2, np.nan)
    p["r_on"] = np.where(p["reason"] == "", np.log(p["open"] / p["prev_close"]), np.nan)
    p["V"] = p["r_on"] ** 2 + p["rho"]
    ret_ok = p["in_span"] & p["bar_ok"] & ~p["trunc"] & p["prev_ok"] & ~p["prev_trunc"] & ~p["is_break"] & ~p["event"]
    p["ret"] = np.where(ret_ok, np.log(p["close"] / p["prev_close"]), np.nan)
    p["session_index"] = p.groupby("symbol").cumcount()
    return p


def add_eligibility(p: pd.DataFrame, rules: Rules, warmup: int) -> pd.DataFrame:
    """Causal eligibility of the forecast origin t (the target is the session t+1). Everything but `target_ok` is known at
    the close of t. `warmup` is the window, in sessions, that must hold valid values for the longest feature (the
    permutation-entropy window plus its embedding); the HAR lags are the daily value and the weekly and monthly counts."""
    p = p.copy()
    key = [p["symbol"], p["segment"]]                       # windows never cross an identity change

    def count(valid: pd.Series, window: int) -> pd.Series:
        return valid.astype(float).groupby(key, sort=False).transform(lambda x: x.rolling(window, min_periods=1).sum())

    v_ok, r_ok = p["V"].notna(), p["ret"].notna()
    p["lag_ok"] = v_ok
    p["week_ok"] = count(v_ok, rules.weekly) >= rules.weekly_min
    p["month_ok"] = count(v_ok, rules.monthly) >= rules.monthly_min
    p["warm_ok"] = (count(r_ok, warmup) >= rules.coverage * warmup) & (count(v_ok, warmup) >= rules.coverage * warmup)
    p["liq_ok"] = (p["raw_close"] >= rules.min_raw_close) & (p["dv63"] >= rules.min_dollar_volume)
    g = p.groupby("symbol", sort=False)
    p["target_date"], p["V_next"] = g["date"].shift(-1), g["V"].shift(-1)
    p["V_id_next"], p["target_half_day"] = g["rho"].shift(-1), g["half_day"].shift(-1)
    p["target_in_span"] = g["in_span"].shift(-1).eq(True)
    p["target_ok"] = p["V_next"].notna()
    p["eligible"] = p["target_ok"] & p["liq_ok"] & p["lag_ok"] & p["week_ok"] & p["month_ok"] & p["warm_ok"]
    return p


def window_of(target_date: pd.Series, windows: dict = WINDOWS) -> pd.Series:
    out = pd.Series("", index=target_date.index, dtype=object)
    for name, (lo, hi) in windows.items():
        out[(target_date >= pd.Timestamp(lo)) & (target_date <= pd.Timestamp(hi))] = name
    return out


def sample_sizes(p: pd.DataFrame, rules: Rules, warmups=(63, 126, 252), windows: dict = WINDOWS) -> pd.DataFrame:
    """Stock-days by window (of the target date) and warm-up, as a waterfall: every row is kept in the panel; these
    are the counts that survive each requirement in turn."""
    rows = []
    for w in warmups:
        q = add_eligibility(p, rules, w)
        q["window"] = window_of(q["target_date"], windows)
        for name in windows:
            x = q[q["window"] == name]
            steps = {"listed stock-days": x["target_in_span"], "target has a value": x["target_ok"]}
            keep = x["target_ok"]
            for label, col in (("price and liquidity", "liq_ok"), ("daily lag", "lag_ok"), ("weekly window", "week_ok"),
                               ("monthly window", "month_ok"), ("warm-up", "warm_ok")):
                keep = keep & x[col]
                steps[f"+ {label}"] = keep
            row = {"warmup": w, "window": name, "sessions": int(x["target_date"].nunique())}
            row.update({k: int(s.sum()) for k, s in steps.items()})
            row["eligible"] = int(x["eligible"].sum())
            row["eligible, full-day targets"] = int((x["eligible"] & ~x["target_half_day"].eq(True)).sum())
            row["symbols with an eligible stock-day"] = int(x.loc[x["eligible"], "symbol"].nunique())
            rows.append(row)
    return pd.DataFrame(rows)


def reason_counts(p: pd.DataFrame, windows: dict = WINDOWS) -> pd.DataFrame:
    """Why the target (V of the session itself) is missing, by window of the session date: the missingness reasons."""
    q = p.assign(window=window_of(p["date"], windows))
    q = q[(q["window"] != "") & q["in_span"]]
    if q.empty:
        return pd.DataFrame()
    t = q.assign(reason=q["reason"].replace("", "(valid)")).groupby(["window", "reason"]).size().unstack(0).fillna(0).astype(int)
    return t.sort_values(t.columns[0], ascending=False)
