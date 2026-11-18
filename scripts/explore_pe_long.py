#!/usr/bin/env python3
"""EXPLORATORY, not a registered test: does PE of daily returns predict next-day variance once the stock's long-run level is in the model?

    python scripts/explore_pe_long.py --reference data/reference --official data/reference/official_prices.parquet

The lead comes from amendment 3's robustness line on the reserved period (Clark-West t 2.71, a negative PE coefficient on every refit). This looks at it on all the
daily data, 2016 to 2026-10-12, reserved period included, which is already used. Nothing here confirms anything: whatever survives is written as a rule and tested on
trading days after 2026-11-17. Every variant tried is counted and printed, so the search can be disclosed. Daily bars only; prints everything, writes nothing.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import ROOT

from src import portfolio as pf, volforecast as vf

BASE = ["d", "w", "m", "long"]
START, END = "2017-04-01", "2026-10-12"     # the first year is training only


def add_variants(rows: pd.DataFrame, returns: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Alternative predictors, each through t: other PE settings, PE purged of the long-run level within each date, and lag-1 return autocorrelation."""
    out = rows.copy()
    def put(name, wide):
        nonlocal out
        out = out.merge(wide.stack().rename(name).reset_index(), on=["date", "symbol"], how="left")
    put("pe_w21", pf.rolling_pe(returns, 3, 1, 21))
    put("pe_w126", pf.rolling_pe(returns, 3, 1, 126))
    put("pe_m4_w126", pf.rolling_pe(returns, 4, 1, 126))
    put("ac1", returns.rolling(63, min_periods=57).corr(returns.shift(1)))
    def purge(f):
        b = np.polyfit(f["long"], f["pe"], 1) if len(f) > 10 else (0.0, f["pe"].mean())
        return f["pe"] - (b[0] * f["long"] + b[1])
    out["pe_purged"] = out.groupby("date", group_keys=False)[["pe", "long"]].apply(purge)
    variants = {"pe (registered: order 3, 63 patterns)": "pe", "pe, 21 patterns": "pe_w21", "pe, 126 patterns": "pe_w126", "pe, order 4, 126 patterns": "pe_m4_w126",
                "pe purged of the long-run level": "pe_purged", "lag-1 return autocorrelation (not PE)": "ac1"}
    return out, variants


def oos(rows: pd.DataFrame, cfg: vf.VolConfig, extra: str, first=START, last=END) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keep = rows.dropna(subset=[extra])
    small, _ = vf.walk_forward(keep, BASE, cfg, first, last)
    large, coefs = vf.walk_forward(keep, BASE + [extra], cfg, first, last)
    return small, large, coefs


def summary(small, large, coefs, extra, cfg) -> dict:
    cw = vf.clark_west(small, large)
    a, b = vf.daily_loss(small), vf.daily_loss(large)
    ll, q = vf.gap(a["mse_log"], b["mse_log"], cfg), vf.gap(a["qlike"], b["qlike"], cfg)
    c = coefs[extra]
    return {"clark_west_t": pf.newey_west_t(cw["clark_west"], lags=cfg.hac_lags), "log_loss_gap_x1e5": 1e5 * ll["gap"], "lo_x1e5": 1e5 * ll["lo"], "hi_x1e5": 1e5 * ll["hi"],
            "qlike_gap_x1e5": 1e5 * q["gap"], "qlike_lo_x1e5": 1e5 * q["lo"], "qlike_hi_x1e5": 1e5 * q["hi"], "coef_mean": float(c.mean()), "same_sign": float((np.sign(c) == np.sign(c.mean())).mean())}


def analyse(rows: pd.DataFrame, returns: pd.DataFrame, cfg: vf.VolConfig, first=START, last=END) -> None:
    rows, variants = add_variants(rows, returns)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"rows {len(rows):,}, {rows['date'].min().date()} to {rows['target_date'].max().date()}, stocks {rows['symbol'].nunique()}")

    print("\n1. How PE relates to volatility, pooled correlations:")
    print(rows[["pe", "long", "m", "d", "ac1"]].assign(m_minus_long=rows["m"] - rows["long"]).corr().round(3).to_string())

    print("\n2. Out of sample, the long-run-level baseline with and without the registered PE, by target year (gaps in units of 1e-5; negative means PE helps):")
    small, large, coefs = oos(rows, cfg, "pe", first, last)
    whole = summary(small, large, coefs, "pe", cfg)
    years = {}
    for y in sorted(small["target_date"].dt.year.unique()):
        s, l = small[small["target_date"].dt.year == y], large[large["target_date"].dt.year == y]
        cw = vf.clark_west(s, l)
        years[int(y)] = {"clark_west_t": pf.newey_west_t(cw["clark_west"], lags=cfg.hac_lags), "log_loss_gap_x1e5": 1e5 * (vf.daily_loss(l)["mse_log"] - vf.daily_loss(s)["mse_log"]).mean(),
                         "qlike_gap_x1e5": 1e5 * (vf.daily_loss(l)["qlike"] - vf.daily_loss(s)["qlike"]).mean()}
    print(pd.DataFrame(years).T.round(2).to_string())
    print("   whole span:", {k: round(v, 3) for k, v in whole.items()})
    print("   PE coefficient by refit year, mean:", coefs["pe"].groupby(pd.DatetimeIndex(coefs.index).year).mean().round(4).to_dict())

    print("\n3. Is it PE? Each predictor added to the long-run-level baseline, out of sample over the whole span:")
    table = {name: summary(*oos(rows, cfg, col, first, last), col, cfg) for name, col in variants.items()}
    print(pd.DataFrame(table).T.round(3).to_string())

    print("\n4. Where it lives, the registered PE: out-of-sample log-loss and QLIKE gaps (1e-5) by group:")
    m = small.merge(large[["symbol", "date", "f", "h"]], on=["symbol", "date"], suffixes=("_0", "_1")).merge(rows[["symbol", "date", "m", "long", "vm"]], on=["symbol", "date"])
    m["log_gap"] = (m["y"] - m["f_1"]) ** 2 - (m["y"] - m["f_0"]) ** 2
    q = lambda h: m["next"] / h - np.log(m["next"] / h) - 1
    m["qlike_gap"] = q(m["h_1"]) - q(m["h_0"])
    m["vol now against its long run"] = pd.qcut(m["m"] - m["long"], 3, labels=["below", "near", "above"])
    m["long-run level"] = pd.qcut(m["long"], 3, labels=["low", "middle", "high"])
    for g in ("vol now against its long run", "long-run level"):
        print(m.groupby(g, observed=True)[["log_gap", "qlike_gap"]].mean().mul(1e5).round(2).to_string())
    print(f"\nVARIANTS TRIED: {len(variants)} predictors, each one out-of-sample fit on the whole span, plus the year and group splits of the registered PE. Exploratory: confirm only on new data.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/reference")
    ap.add_argument("--official", default=None)
    args = ap.parse_args()
    cfg = vf.VolConfig()
    p = vf.load_panel(None, args.reference, pd.Timestamp(cfg.test_end), args.official, ROOT)
    rows = vf.build_rows(p, cfg, pf.load_joined(ROOT / "registry" / "sp500_join_dates.csv"))
    analyse(rows, p.pivot(index="date", columns="symbol", values="ret"), cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
