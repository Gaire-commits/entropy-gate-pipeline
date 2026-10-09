"""IEX bars against consolidated (SIP) bars for the same stock and days.

IEX prints a bar only when a trade happened on IEX, a few percent of the market, so its grid has gaps
and its volume is a sample. The consolidated tape has a bar whenever the stock traded anywhere. This
compares the two on the same symbol and days: how complete each grid is, how much more volume SIP
carries, and whether the prices agree where both have a bar (they should: IEX trades are real trades).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import regular_hours, to_eastern, to_session_grid


def gridded(raw: pd.DataFrame, minutes: int = 5) -> pd.DataFrame | None:
    """Raw downloaded bars -> the project's session grid (regular hours, NaN where there was no bar)."""
    if raw is None or raw.empty:
        return None
    df = regular_hours(to_eastern(raw))
    return None if df.empty else to_session_grid(df, minutes)


def compare_feeds(iex: pd.DataFrame | None, sip: pd.DataFrame | None, minutes: int = 5) -> dict:
    """Both are one symbol's gridded bars. Returns what each has and how they agree on shared bars."""
    out = {"iex_sessions": 0, "sip_sessions": 0, "iex_completeness": np.nan, "sip_completeness": np.nan,
           "common_bars": 0, "volume_ratio": np.nan, "close_diff_bps_median": np.nan, "close_diff_bps_p95": np.nan,
           "return_corr": np.nan}
    for name, df in (("iex", iex), ("sip", sip)):
        if df is not None and len(df):
            out[f"{name}_sessions"] = int(df["session_id"].nunique())
            out[f"{name}_completeness"] = float(df["close"].notna().mean())
    if iex is None or sip is None or not len(iex) or not len(sip):
        return out
    a = iex.droplevel("symbol") if "symbol" in iex.index.names else iex
    b = sip.droplevel("symbol") if "symbol" in sip.index.names else sip
    both = a.index.intersection(b.index)
    a, b = a.loc[both], b.loc[both]
    ok = a["close"].notna() & b["close"].notna()
    out["common_bars"] = int(ok.sum())
    if not ok.any():
        return out
    ratio = b.loc[ok, "volume"] / a.loc[ok, "volume"].replace(0, np.nan)
    out["volume_ratio"] = float(ratio.median())
    diff = 1e4 * (a.loc[ok, "close"] / b.loc[ok, "close"] - 1).abs()
    out["close_diff_bps_median"], out["close_diff_bps_p95"] = float(diff.median()), float(diff.quantile(0.95))
    ra, rb = np.log(a["close"]).diff(), np.log(b["close"]).diff()
    both_r = ra.notna() & rb.notna()
    if both_r.sum() > 10:
        out["return_corr"] = float(np.corrcoef(ra[both_r], rb[both_r])[0, 1])
    return out


def summarize(rows: list[dict]) -> dict:
    """Average the per-symbol comparisons of one period (symbols with no data on either side are skipped)."""
    frame = pd.DataFrame(rows)
    keys = ["iex_completeness", "sip_completeness", "volume_ratio", "close_diff_bps_median", "close_diff_bps_p95",
            "return_corr"]
    out = {k: float(frame[k].mean()) if k in frame and frame[k].notna().any() else np.nan for k in keys}
    out["symbols_iex"] = int((frame.get("iex_sessions", pd.Series(dtype=int)) > 0).sum()) if len(frame) else 0
    out["symbols_sip"] = int((frame.get("sip_sessions", pd.Series(dtype=int)) > 0).sum()) if len(frame) else 0
    out["symbols"] = int(len(frame))
    return out
