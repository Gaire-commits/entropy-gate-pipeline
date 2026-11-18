#!/usr/bin/env python3
"""EXPLORATORY, not a registered test: which simple daily features predict the next one or two days?

    python scripts/explore_signals.py --reference data/reference --official data/reference/official_prices.parquet

All daily data, 2016 to 2026-10-12 (the reserved period is already used). Each feature is known at the close of t. Four targets:
  ret_1d   close t to close t+1 (includes the bid-ask bounce, so reversal looks stronger than it trades)
  ret_oc   open t+1 to close t+1 (starts after the overnight, so it is closer to what could be traded)
  ret_2d   close t to close t+2
  vol_1d   ln V(t+1) minus the out-of-sample forecast of HAR plus the long-run level: variance beyond what volatility history predicts
For each feature and target: the daily cross-sectional rank correlation (IC) across eligible stocks, its Newey-West t, the same in each half of the period, and a
Benjamini-Hochberg q across every test run here. Every test is counted. Anything that survives is a lead, to be written as a rule and tested on days after 2026-11-17.
Daily bars only; prints everything, writes nothing.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import ROOT

from src import portfolio as pf, volforecast as vf

SPLIT = pd.Timestamp("2021-07-01")
LAGS = {"ret_1d": 1, "ret_oc": 1, "ret_2d": 2, "vol_1d": 5}


def wide(p: pd.DataFrame, col: str) -> pd.DataFrame:
    return p.pivot(index="date", columns="symbol", values=col)


def features(p: pd.DataFrame, returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each known at the close of t."""
    on = wide(p, "r_on")
    intraday = returns - on
    rho = wide(p, "rho").clip(lower=1e-10)
    V = wide(p, "V").clip(lower=1e-10)
    f = {"reversal_1d": returns, "reversal_5d": returns.rolling(5, min_periods=4).sum(), "momentum_21d": returns.rolling(21, min_periods=18).sum(),
         "overnight_gap": on, "intraday_move": intraday, "range_vs_63d": np.log(rho / rho.rolling(63, min_periods=40).median()),
         "vol_5d_vs_22d": np.log(V.rolling(5, min_periods=4).mean() / V.rolling(22, min_periods=18).mean()),
         "max_return_21d": returns.rolling(21, min_periods=18).max(), "pe": pf.rolling_pe(returns, 3, 1, 63)}
    if "raw_volume" in p:
        vol = wide(p, "raw_volume").where(lambda x: x > 0)
        f["volume_shock"] = np.log(vol / vol.shift(1).rolling(63, min_periods=40).median())
    return f


def targets(p: pd.DataFrame, returns: pd.DataFrame, rows: pd.DataFrame, cfg: vf.VolConfig, first, last) -> dict[str, pd.DataFrame]:
    intraday = returns - wide(p, "r_on")
    fc, _ = vf.walk_forward(rows, vf.LOG_HAR + ["long"], cfg, first, last)
    resid = fc.assign(e=fc["y"] - fc["f"]).pivot(index="date", columns="symbol", values="e")
    return {"ret_1d": returns.shift(-1), "ret_oc": intraday.shift(-1), "ret_2d": returns.shift(-1) + returns.shift(-2), "vol_1d": resid}


def daily_ic(x: pd.DataFrame, y: pd.DataFrame, mask: pd.DataFrame, min_names: int = 50) -> pd.Series:
    """Per date, the rank correlation across stocks where both are known and the stock is eligible."""
    x, y = x.reindex_like(mask).where(mask), y.reindex_like(mask).where(mask)
    both = x.notna() & y.notna()
    rx = x.where(both).rank(axis=1)
    ry = y.where(both).rank(axis=1)
    rx, ry = rx.sub(rx.mean(axis=1), axis=0), ry.sub(ry.mean(axis=1), axis=0)
    ic = (rx * ry).sum(axis=1) / np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
    return ic[both.sum(axis=1) >= min_names].dropna()


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def analyse(p: pd.DataFrame, rows: pd.DataFrame, cfg: vf.VolConfig, first, last, split=SPLIT) -> pd.DataFrame:
    from math import erfc, sqrt
    returns = wide(p, "ret")
    mask = rows.assign(ok=True).pivot(index="date", columns="symbol", values="ok").fillna(False).astype(bool)
    mask = mask[(mask.index >= pd.Timestamp(first)) & (mask.index <= pd.Timestamp(last))]
    feats, targs = features(p, returns), targets(p, returns, rows, cfg, first, last)
    out = []
    for fname, x in feats.items():
        for tname, y in targs.items():
            ic = daily_ic(x, y, mask)
            if len(ic) < 100:
                continue
            t = pf.newey_west_t(ic, lags=LAGS[tname])
            a, b = ic[ic.index < split], ic[ic.index >= split]
            out.append({"feature": fname, "target": tname, "days": len(ic), "ic_x100": 100 * ic.mean(), "t": t, "ic_first_half_x100": 100 * a.mean(), "ic_second_half_x100": 100 * b.mean(),
                        "same_sign_both_halves": bool(np.sign(a.mean()) == np.sign(b.mean())), "p": erfc(abs(t) / sqrt(2))})
    table = pd.DataFrame(out)
    table["q"] = bh(table["p"].to_numpy())
    return table.sort_values("t", key=abs, ascending=False).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/reference")
    ap.add_argument("--official", default=None)
    args = ap.parse_args()
    cfg = vf.VolConfig()
    p = vf.load_panel(None, args.reference, pd.Timestamp(cfg.test_end), args.official, ROOT)
    rows = vf.build_rows(p, cfg, pf.load_joined(ROOT / "registry" / "sp500_join_dates.csv"))
    table = analyse(p, rows, cfg, "2017-04-01", cfg.test_end)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)
    print(f"stock-days {len(rows):,}; split for the halves at {SPLIT.date()}. IC in percent; t Newey-West; q Benjamini-Hochberg over all {len(table)} tests.\n")
    print(table.drop(columns="p").round({"ic_x100": 2, "t": 2, "ic_first_half_x100": 2, "ic_second_half_x100": 2, "q": 4}).to_string(index=False))
    keep = table[(table["q"] < 0.01) & table["same_sign_both_halves"]]
    print(f"\nLEADS (q < 0.01 and the same sign in both halves): {len(keep)} of {len(table)} tests.")
    print(keep[["feature", "target", "ic_x100", "t"]].round(2).to_string(index=False) if len(keep) else "none")
    print(f"\nTESTS RUN: {len(table)}. Exploratory: confirm only on trading days after 2026-11-17.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
