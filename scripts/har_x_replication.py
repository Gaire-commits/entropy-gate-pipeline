#!/usr/bin/env python3
"""The HAR-X replication on mid caps (protocols/har_x_replication_v1.yaml), run once. Prints everything, writes nothing.

    python scripts/har_x_replication.py --reference /content/midcap/reference --official /content/midcap/reference/official_prices.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from _common import ROOT

import explore_har_x
from src import audit, portfolio as pf, volforecast as vf, volpanel

PROTOCOL = ROOT / "protocols" / "har_x_replication_v1.yaml"


def load(path=PROTOCOL) -> dict:
    with open(path) as f:
        doc = yaml.safe_load(f)
    full = explore_har_x.BLOCKS["+ largest return, volume shock (full)"]
    assert doc["base"] == explore_har_x.BASE and doc["extra"] == full and doc["clip"] == explore_har_x.CLIP, "the protocol and the large-cap model disagree"
    return doc


def panel(reference, official, universe, end) -> pd.DataFrame:
    """As volforecast.load_panel, from daily bars only, for any universe."""
    end = pd.Timestamp(end)
    def upto(frame):
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        return frame[frame["date"] <= end]
    daily = {t: upto(pd.read_parquet(Path(reference) / "daily" / f"daily_sip_{t}.parquet")) for t in ("raw", "split", "all")}
    vendor = volpanel.vendor_frame(daily["raw"], daily["split"], daily["all"])
    actions = audit.normalize_actions(pd.read_parquet(Path(reference) / "corporate_actions.parquet"))
    registry = volpanel.load_registry(ROOT / "registry" / "vol_registry.csv")
    symbols = sorted(pd.read_csv(ROOT / universe)["symbol"])
    return volpanel.build_panel(None, vendor, actions, registry, symbols, volpanel.Rules(), end=end, official=upto(pd.read_parquet(official)))


def verdict(log_gap: dict, qlike_gap: dict) -> str:
    hits = (log_gap["hi"] < 0) + (qlike_gap["hi"] < 0)
    return {2: "REPLICATES", 1: "PARTLY REPLICATES", 0: "DOES NOT REPLICATE"}[int(hits)]


def run(rows: pd.DataFrame, doc: dict, cfg: vf.VolConfig) -> dict:
    rows = rows.dropna(subset=doc["base"] + doc["extra"])
    small, _ = vf.walk_forward(rows, doc["base"], cfg, doc["first_target"], doc["last_target"])
    large, coefs = vf.walk_forward(rows, doc["base"] + doc["extra"], cfg, doc["first_target"], doc["last_target"])
    a, b = vf.daily_loss(small), vf.daily_loss(large)
    ll, q = vf.gap(a["mse_log"], b["mse_log"], cfg), vf.gap(a["qlike"], b["qlike"], cfg)
    cw = vf.clark_west(small, large)
    yr = pd.DataFrame({"log_gap": b["mse_log"] - a["mse_log"], "qlike_gap": b["qlike"] - a["qlike"]})
    by_year = yr.groupby(yr.index.year).mean()
    by_year["log_gain_pct"] = -100 * by_year["log_gap"] / a["mse_log"].groupby(a.index.year).mean()
    signs = {k: {"mean": float(coefs[k].mean()), "same_sign_as_large_caps": float((np.sign(coefs[k]) == s).mean())} for k, s in doc["large_cap_signs"].items()}
    return {"rows": len(rows), "stocks": rows["symbol"].nunique(), "baseline_log_loss": float(a["mse_log"].mean()), "baseline_qlike": float(a["qlike"].mean()),
            "log": ll, "qlike": q, "clark_west_t": pf.newey_west_t(cw["clark_west"], lags=cfg.hac_lags), "by_year": by_year, "signs": signs, "verdict": verdict(ll, q)}


def report(r: dict) -> None:
    pd.set_option("display.width", 200)
    print(f"rows {r['rows']:,}, stocks {r['stocks']}; baseline log loss {r['baseline_log_loss']:.4f}, QLIKE {r['baseline_qlike']:.4f}")
    for name, g, base in (("log loss", r["log"], r["baseline_log_loss"]), ("QLIKE", r["qlike"], r["baseline_qlike"])):
        print(f"  {name}, HAR-X minus baseline: {g['gap']:+.5f} [{g['lo']:+.5f}, {g['hi']:+.5f}], t {g['t']:.2f}; gain {-100 * g['gap'] / base:.2f}%")
    print(f"  Clark-West t {r['clark_west_t']:.2f}")
    print("\nby target year:")
    print(r["by_year"].round(4).to_string())
    print("\ncoefficients against their large-cap signs:", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in r["signs"].items()})
    print(f"\nVERDICT: {r['verdict']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--official", required=True)
    args = ap.parse_args()
    doc, cfg = load(), vf.VolConfig()
    print(f"HAR-X replication on mid caps, protocol registered {doc['registered']}. Daily bars only.")
    p = panel(args.reference, args.official, doc["universe"], doc["last_target"])
    rows = vf.build_rows(p, cfg, None)
    print("official-open share of rows:", round(float((rows["open_source"] == "official").mean()), 4))
    report(run(explore_har_x.add_features(rows, p), doc, cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
