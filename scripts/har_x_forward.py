#!/usr/bin/env python3
"""The HAR-X forward test (protocols/har_x_forward_v1.yaml). Prints everything, writes nothing.

    python scripts/har_x_forward.py --sp500 /content/reserved/reference --sp400 /content/midcap/reference

Each run fits and forecasts from 2016 to the latest day fetched, recursively, so a forecast only ever uses data through its origin, and scores the target dates from
2026-11-19 on. Before 500 target dates it is a progress report; at 500 it gives the one verdict.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from _common import ROOT

import explore_har_x
import har_x_replication
from src import portfolio as pf, volforecast as vf

PROTOCOL = ROOT / "protocols" / "har_x_forward_v1.yaml"


def load(path=PROTOCOL) -> tuple[dict, dict]:
    with open(path) as f:
        doc = yaml.safe_load(f)
    return doc, har_x_replication.load(ROOT / doc["model_protocol"])


def daily_gaps(rows: pd.DataFrame, model: dict, cfg: vf.VolConfig, first, last) -> pd.DataFrame:
    """Per target date in [first, last]: HAR-X minus baseline, in QLIKE and in log loss, and the Clark-West difference."""
    rows = rows.dropna(subset=model["base"] + model["extra"])
    small, _ = vf.walk_forward(rows, model["base"], cfg, first, last)
    large, _ = vf.walk_forward(rows, model["base"] + model["extra"], cfg, first, last)
    a, b = vf.daily_loss(small), vf.daily_loss(large)
    return pd.DataFrame({"qlike": b["qlike"] - a["qlike"], "log": b["mse_log"] - a["mse_log"], "base_qlike": a["qlike"], "base_log": a["mse_log"],
                         "clark_west": vf.clark_west(small, large)["clark_west"]})


def judge(g: dict[str, pd.DataFrame], doc: dict, cfg: vf.VolConfig) -> dict:
    both = pd.concat({k: v["qlike"] for k, v in g.items()}, axis=1).dropna()
    avg = both.mean(axis=1)
    n = len(avg)
    stage = "VERDICT" if n >= doc["decision_at_target_dates"] else "INTERIM (not decisive)"
    if n >= doc["decision_at_target_dates"]:
        avg = avg.iloc[:doc["decision_at_target_dates"]]
    q = vf.gap(pd.Series(0.0, index=avg.index), avg, cfg) if len(avg) >= 2 * cfg.block else {"gap": float(avg.mean()) if n else float("nan"), "lo": float("nan"), "hi": float("nan"), "t": float("nan")}
    out = {"target_dates": n, "stage": stage, "qlike_gap": q}
    if stage == "VERDICT":
        out["verdict"] = "CONFIRMED" if q["hi"] < 0 else "NOT CONFIRMED"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp500", required=True)
    ap.add_argument("--sp400", required=True)
    args = ap.parse_args()
    doc, model = load()
    cfg = vf.VolConfig()
    print(f"HAR-X forward test, registered {doc['registered']}; target dates from {doc['first_target']}. Daily bars only.")
    gaps = {}
    for name, ref in (("sp500", args.sp500), ("sp400", args.sp400)):
        raw = pd.read_parquet(Path(ref) / "daily" / "daily_sip_raw.parquet", columns=["date"])
        end = pd.to_datetime(raw["date"]).max().normalize()
        p = har_x_replication.panel(ref, Path(ref) / "official_prices.parquet", doc["universes"][name], end)
        rows = explore_har_x.add_features(vf.build_rows(p, cfg, None), p)
        gaps[name] = daily_gaps(rows, model, cfg, doc["first_target"], end)
        g = gaps[name]
        print(f"\n{name}: data to {end.date()}; {len(g)} target dates scored")
        if len(g):
            print(f"  QLIKE gap {g['qlike'].mean():+.5f} ({-100 * g['qlike'].mean() / g['base_qlike'].mean():.2f}% gain); log-loss gap {g['log'].mean():+.5f} "
                  f"({-100 * g['log'].mean() / g['base_log'].mean():.2f}%); Clark-West t {pf.newey_west_t(g['clark_west'], lags=cfg.hac_lags):.2f}")
            q = g.groupby(pd.PeriodIndex(g.index, freq="Q"))[["qlike", "log"]].mean()
            print("  by quarter (gaps; negative is better):", q.round(5).to_dict("index"))
    r = judge(gaps, doc, cfg)
    q = r["qlike_gap"]
    print(f"\n{r['stage']}: {r['target_dates']} of {doc['decision_at_target_dates']} target dates. QLIKE gap, both universes averaged: {q['gap']:+.5f} [{q['lo']:+.5f}, {q['hi']:+.5f}]")
    if "verdict" in r:
        print("VERDICT:", r["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
