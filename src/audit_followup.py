"""Follow-up tables for the extended audit (read-only): what its first reading left open.

Everything here reads the audit's own outputs (sessions.parquet, large_moves.csv) and the reference data; nothing is fitted
and nothing is excluded. Each function returns tables, scripts/audit_followup.py writes them.

  * range_conflicts: the vendor's daily range (quoted prices) against the range rebuilt from the 5-minute cache, by size and direction;
  * truncation: dates on which the cache or the vendor's daily bar holds a fraction of the session;
  * adjustment_steps: where the vendor's adjustment factor changes, against the corporate-action records;
  * move_evidence: for each large move, the volume behind it and how the two sources' closes compare.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import nyse, volpanel

PARKINSON = volpanel.PARKINSON
BUCKETS = [-np.inf, -0.5, -0.25, -0.10, -0.02, -0.001, 0.001, 0.02, 0.10, 0.25, 0.5, np.inf]


def real_bars(vendor: pd.DataFrame) -> pd.DataFrame:
    return vendor[(vendor["volume"] > 0) & (vendor["trade_count"] > 0)]


def range_conflicts(sessions: pd.DataFrame, vendor: pd.DataFrame, threshold: float = float(np.log(1.25))) -> dict:
    """log(vendor range / rebuilt range) per session where both exist: buckets, the sessions beyond `threshold` by year, the
    worst on complete 5-minute days, and the mean Parkinson variance of each source with and without those sessions."""
    s = sessions[sessions["nyse_session"] & sessions["status"].isin(["ok", "sparse"])]
    v = real_bars(vendor)[["symbol", "date", "raw_high", "raw_low"]]
    m = s.merge(v, on=["symbol", "date"])
    m["lr_5"], m["lr_v"] = np.log(m["high"] / m["low"]), np.log(m["raw_high"] / m["raw_low"])
    m = m[(m["lr_5"] > 0) & (m["lr_v"] > 0)].copy()
    m["d"] = np.log(m["lr_v"] / m["lr_5"])
    m["complete"] = m["bars"] >= m["slots"]
    m["bucket"] = pd.cut(m["d"], BUCKETS)
    buckets = pd.crosstab(m["bucket"], m["complete"].map({True: "complete 5-minute day", False: "incomplete"}), margins=True)
    buckets["share"] = buckets["All"] / len(m)
    buckets = buckets.reset_index().astype({"bucket": str})
    big = m[m["d"].abs() > threshold].copy()
    big["direction"] = np.where(big["d"] > 0, "vendor wider", "vendor narrower")
    by_year = pd.crosstab([big["date"].dt.year.rename("year"), big["direction"]], big["complete"].map({True: "complete", False: "incomplete"})) \
        .reset_index()
    worst = m[m["complete"]].reindex(m.loc[m["complete"], "d"].abs().sort_values(ascending=False).index).head(25) \
        .rename(columns={"lr_5": "log_range_cache", "lr_v": "log_range_vendor"})
    means = {"sessions compared": int(len(m)), f"beyond {np.exp(threshold):.2f}x": int(len(big)),
             "mean Parkinson, vendor": float(PARKINSON * (m["lr_v"] ** 2).mean()),
             "mean Parkinson, rebuilt": float(PARKINSON * (m["lr_5"] ** 2).mean())}
    rest = m[m["d"].abs() <= threshold]
    means["mean Parkinson, vendor, without them"] = float(PARKINSON * (rest["lr_v"] ** 2).mean())
    means["mean Parkinson, rebuilt, without them"] = float(PARKINSON * (rest["lr_5"] ** 2).mean())
    return {"buckets": buckets, "by_year": by_year,
            "worst": worst[["symbol", "date", "bars", "slots", "log_range_cache", "log_range_vendor", "d"]], "means": means}


def trailing_ratio(v: pd.DataFrame, col: str) -> pd.Series:
    """The column over its own trailing 63-bar median (earlier bars only) per symbol."""
    r = real_bars(v).sort_values(["symbol", "date"])
    med = r.groupby("symbol")[col].transform(lambda x: x.shift(1).rolling(63, min_periods=40).median())
    return (r[col] / med).reindex(v.index)


def truncation(sessions: pd.DataFrame, vendor: pd.DataFrame, ratio: float = 0.05) -> dict:
    """Per full-day date: how much of the session the 5-minute cache holds (share of slots missing, share of stocks whose last
    bar starts before 14:55) and how much the vendor's daily bar holds (the cross-section's median trade count and volume over
    each stock's trailing median). Dates under `ratio` on trade count are the vendor's truncated days."""
    s = sessions[sessions["nyse_session"] & sessions["status"].isin(["ok", "sparse"]) & ~sessions["half_day"]].copy()
    s["missing"] = 1 - s["bars"] / s["slots"]
    s["early_stop"] = s["last_bar"] < 14 * 60 + 55
    cache = s.groupby("date").agg(stocks=("symbol", "size"), cache_missing_share=("missing", "mean"),
                                  cache_last_bar_before_1455=("early_stop", "mean"), cache_median_bars=("bars", "median"))
    v = vendor.reset_index(drop=True)
    v = v.assign(tc_ratio=trailing_ratio(v, "trade_count"), vol_ratio=trailing_ratio(v, "volume"))
    day = real_bars(v).groupby("date").agg(vendor_median_trade_count=("trade_count", "median"),
                                           vendor_trade_count_ratio=("tc_ratio", "median"),
                                           vendor_volume_ratio=("vol_ratio", "median"))
    t = cache.join(day, how="left")
    t["vendor_truncated"] = t["vendor_trade_count_ratio"] < ratio
    t["half_day"] = False
    flagged = t[t["vendor_truncated"] | (t["cache_last_bar_before_1455"] > 0.2)].sort_values("cache_missing_share", ascending=False)
    second = t.loc[~t["vendor_truncated"], "vendor_trade_count_ratio"].nsmallest(3)
    rel = v.loc[v["real"] & v["tc_ratio"].notna() & ~v["date"].isin(t.index[t["vendor_truncated"]]), "tc_ratio"]
    return {"flagged": flagged.reset_index(), "lowest_unflagged_trade_count_ratio": second.reset_index(),
            "stock_sessions_under_ratio_elsewhere": int((rel < ratio).sum()), "stock_sessions_with_a_ratio_elsewhere": int(len(rel))}


def adjustment_steps(vendor: pd.DataFrame, actions: pd.DataFrame, thresholds=(2e-4, 5e-4, 1e-3, 1e-2, 2e-2)) -> dict:
    """The vendor's adjustment steps (volpanel.vendor_frame) by size, the ones above the rounding noise matched with the
    split and spin-off records on the same session, and the records the vendor did not adjust for."""
    r = real_bars(vendor)
    a, c = r["step_split"].abs(), r["step_cash"].abs()
    by_size = pd.DataFrame([{"threshold": f"{t:g}", "split-type steps": int((a > t).sum()),
                             "dividend or spin-off steps": int((c > t).sum())} for t in thresholds])
    cal = pd.DatetimeIndex(sorted(vendor["date"].unique()))
    rec = volpanel.record_events(actions, cal)
    st = r.loc[a > 1e-3, ["symbol", "date", "step_split", "step_cash"]].merge(rec.drop_duplicates(["symbol", "date"]),
                                                                                on=["symbol", "date"], how="left")
    unmatched = st[st["source"].isna()].drop(columns="source")
    both = rec.merge(r[["symbol", "date", "step_split", "step_cash"]], on=["symbol", "date"], how="left")
    unadjusted = both[(both["step_split"].abs().fillna(0) <= 1e-2) & (both["step_cash"].abs().fillna(0) <= 1e-2)]
    big = r.loc[c > 0.03, ["symbol", "date", "step_split", "step_cash"]].sort_values("step_cash", key=abs, ascending=False)
    return {"by_size": by_size, "split_steps": int(len(st)), "matched": int(st["source"].notna().sum()), "unmatched": unmatched,
            "records": int(len(rec)), "records_without_a_vendor_step": unadjusted.sort_values("date"), "large_cash_steps": big}


def move_evidence(moves: pd.DataFrame, vendor: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """For each large move: the day's volume over the trailing median, the trade count, and the vendor's close against the
    rebuilt one (a large gap means the cache's last bars are missing)."""
    v = vendor.reset_index(drop=True)
    v = v.assign(vol_ratio=trailing_ratio(v, "volume"))
    out = moves.merge(v[["symbol", "date", "vol_ratio", "trade_count", "close"]].rename(columns={"close": "vendor_close_all"}),
                      on=["symbol", "date"], how="left")
    s = sessions[["symbol", "date", "close", "bars", "last_bar"]].rename(columns={"close": "rebuilt_close"})
    out = out.merge(s, on=["symbol", "date"], how="left")
    split = vendor  # the cache is split-adjusted: compare closes on that basis when the split series is present
    if "split_close" in split.columns:
        out = out.merge(split[["symbol", "date", "split_close"]], on=["symbol", "date"], how="left")
        out["close_gap_pct"] = 100 * np.log(out["split_close"] / out["rebuilt_close"])
    return out


def close_gaps(sessions: pd.DataFrame, vendor: pd.DataFrame, threshold: float = 0.01) -> dict:
    """The vendor's close against the cache's last bar (both split-adjusted): how often they differ by more than `threshold`, and
    whether the cache lacked its last slot or the whole day was there."""
    s = sessions[sessions["nyse_session"] & (sessions["bars"] > 0)][["symbol", "date", "close", "bars", "slots", "close_bar"]]
    m = s.merge(real_bars(vendor)[["symbol", "date", "split_close"]], on=["symbol", "date"])
    m["gap"] = np.log(m["split_close"] / m["close"])
    big = m[m["gap"].abs() > threshold]
    by_year = big.groupby(big["date"].dt.year).size().rename("sessions").reset_index().rename(columns={"date": "year"})
    huge = m[m["gap"].abs() > 0.10]
    return {"compared": int(len(m)), "beyond": int(len(big)), "share_without_last_bar": float(big["close_bar"].eq(False).mean()),
            "beyond_10pct": int(len(huge)), "beyond_10pct_with_last_bar": int(huge["close_bar"].eq(True).sum()),
            "beyond_10pct_by_year": huge.groupby(huge["date"].dt.year).size().to_dict(),
            "share_on_complete_days": float((big["bars"] >= big["slots"]).mean()),
            "within_half_bp": float((m["gap"].abs() <= 5e-5).mean()), "by_year": by_year}


def market_move_volume(ev: pd.DataFrame) -> pd.DataFrame:
    """The audit's 'market move (no record found)' days by the day's volume over its trailing median: news-driven moves bring volume,
    and so, it turned out, do spin-offs, so this supports a genuine move without proving one."""
    mm = ev[ev["class"].astype(str).str.startswith("market move")]
    bucket = pd.cut(mm["vol_ratio"], [0, 1.5, 3, 5, np.inf], labels=["under 1.5x", "1.5-3x", "3-5x", "5x or more"]).astype(object)
    bucket = bucket.where(mm["vol_ratio"].notna(), "no trailing median (new listing or early 2016)")
    return bucket.value_counts().rename("moves").reset_index().rename(columns={"vol_ratio": "volume over trailing median", "index": "volume over trailing median"})
