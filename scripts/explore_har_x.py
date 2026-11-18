#!/usr/bin/env python3
"""EXPLORATORY, not a registered test: a better next-day variance forecast, HAR plus the features the signal search found.

    python scripts/explore_har_x.py --reference data/reference --official data/reference/official_prices.parquet

The signal search (RESULTS.md, 2026-11-17) found variance falling back faster than HAR expects after a wide day, and lower after rises (the leverage effect). This
adds those features to HAR plus the long-run level, one block at a time, and measures each out of sample, 2017 to 2026-10-12: the same recursive pooled fit,
refit every 21 sessions, losses on ln V and QLIKE. The features were chosen on all of this data, so it overstates what a forward test will show. The model
that holds up is to be frozen and tested on trading days after 2026-11-17. Daily bars only; prints everything, writes nothing.

5-day against 22-day variance is left out: in logs it is w - m, already in the model.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import ROOT

import explore_signals
from src import portfolio as pf, volforecast as vf

BASE = ["d", "w", "m", "long"]
CLIP = {"range_ratio": 4.0, "ret_1d": 0.2, "ret_5d": 0.4, "mom_21d": 0.6, "max_ret_21d": 0.4, "volume_shock": 3.0}   # fixed, so no future data sets them
BLOCKS = {"HAR + long-run level (baseline)": [],
          "+ range against its 63-day median": ["range_ratio"],
          "+ recent returns (leverage)": ["range_ratio", "ret_1d", "ret_5d", "mom_21d", "neg_ret_1d"],
          "+ largest return, volume shock (full)": ["range_ratio", "ret_1d", "ret_5d", "mom_21d", "neg_ret_1d", "max_ret_21d", "volume_shock"]}
START, END = "2017-04-01", "2026-10-12"


def add_features(rows: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    returns = explore_signals.wide(p, "ret")
    f = explore_signals.features(p, returns)
    cols = {"range_ratio": f["range_vs_63d"], "ret_1d": f["reversal_1d"], "ret_5d": f["reversal_5d"], "mom_21d": f["momentum_21d"], "max_ret_21d": f["max_return_21d"]}
    if "volume_shock" in f:
        cols["volume_shock"] = f["volume_shock"]
    out = rows
    for name, w in cols.items():
        out = out.merge(w.clip(-CLIP[name], CLIP[name]).stack().rename(name).reset_index(), on=["date", "symbol"], how="left")
    out["neg_ret_1d"] = out["ret_1d"].clip(upper=0.0)
    return out


def compare(rows: pd.DataFrame, cfg: vf.VolConfig, first=START, last=END) -> None:
    blocks = {k: [c for c in v if c in rows] for k, v in BLOCKS.items()}
    used = sorted({c for v in blocks.values() for c in v})
    rows = rows.dropna(subset=BASE + used)
    print(f"rows {len(rows):,}, stocks {rows['symbol'].nunique()}; every model on the same rows and folds")
    fc, coefs = {}, {}
    for name, extra in blocks.items():
        fc[name], coefs[name] = vf.walk_forward(rows, BASE + extra, cfg, first, last)
    base = list(blocks)[0]
    a = vf.daily_loss(fc[base])
    table = {}
    for name in blocks:
        b = vf.daily_loss(fc[name])
        row = {"log_loss": b["mse_log"].mean(), "qlike": b["qlike"].mean(), "realized_over_forecast": vf.calibration(fc[name])}
        if name != base:
            ll, q = vf.gap(a["mse_log"], b["mse_log"], cfg), vf.gap(a["qlike"], b["qlike"], cfg)
            cw = vf.clark_west(fc[base], fc[name])
            row.update({"log_gap": ll["gap"], "log_lo": ll["lo"], "log_hi": ll["hi"], "qlike_gap": q["gap"], "qlike_lo": q["lo"], "qlike_hi": q["hi"],
                        "clark_west_t": pf.newey_west_t(cw["clark_west"], lags=cfg.hac_lags), "log_gain_pct": -100 * ll["gap"] / a["mse_log"].mean(), "qlike_gain_pct": -100 * q["gap"] / a["qlike"].mean()})
        table[name] = row
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 30)
    print("\n1. Out of sample, 2017 to 2026, against the baseline (gaps: model minus baseline, negative is better; intervals 95%, blocks of 21 days):")
    print(pd.DataFrame(table).T.round(4).to_string())
    full = list(blocks)[-1]
    b = vf.daily_loss(fc[full])
    by = pd.DataFrame({"log_gap": b["mse_log"] - a["mse_log"], "qlike_gap": b["qlike"] - a["qlike"]})
    yr = by.groupby(by.index.year).mean()
    yr["log_gain_pct"] = -100 * yr["log_gap"] / a["mse_log"].groupby(a.index.year).mean()
    print(f"\n2. The full model by target year (negative gap is better):")
    print(yr.round(4).to_string())
    c = coefs[full]
    print("\n3. The full model's coefficients over the refits: mean, standard deviation, share with the mean's sign:")
    print(pd.DataFrame({"mean": c.mean(), "std": c.std(), "same_sign": (np.sign(c) == np.sign(c.mean())).mean()}).round(4).to_string())
    print(f"\nMODELS TRIED: {len(blocks) - 1} additions to the baseline, built in fixed blocks. Exploratory: the features were picked on this data; confirm only on days after 2026-11-17.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/reference")
    ap.add_argument("--official", default=None)
    args = ap.parse_args()
    cfg = vf.VolConfig()
    p = vf.load_panel(None, args.reference, pd.Timestamp(cfg.test_end), args.official, ROOT)
    rows = vf.build_rows(p, cfg, pf.load_joined(ROOT / "registry" / "sp500_join_dates.csv"))
    compare(add_features(rows, p), cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
