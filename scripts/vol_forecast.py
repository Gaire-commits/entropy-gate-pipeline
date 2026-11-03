#!/usr/bin/env python3
"""The volatility protocol, by stage (protocols/volatility_v1.yaml; prints everything, writes nothing).

    python scripts/vol_forecast.py --stage baseline --audit outputs/data_audit_v2 --reference data/reference
    python scripts/vol_forecast.py --stage reserved --score-reserved --audit ... --reference ... --official data/reference/official_prices.parquet [--rv data/reference/rv5.parquet]

`baseline` chooses the HAR form on development and confirms it on validation. It uses no PE and loads nothing after the lock.
`rv` builds the realized-variance robustness target from the 5-minute cache. `reserved` runs the one registered comparison, once,
on the reserved test window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT

from src import portfolio as pf, volforecast as vf


def baseline_stage(args, cfg: vf.VolConfig) -> int:
    end = vf.lock_end(cfg)
    p = vf.load_panel(Path(args.audit) / "sessions.parquet", args.reference, end, args.official, ROOT)
    print(f"panel through {p['date'].max().date()} (locked); open: {'official where printed' if args.official else 'the vendor open'}")
    joined = pf.load_joined(ROOT / "registry" / "sp500_join_dates.csv")
    rows = vf.build_rows(p, cfg, joined)
    print(vf.sample_sizes(rows, cfg).to_string(index=False))
    spans = {"development": ("2016-01-01", "2021-12-31"), "validation": (cfg.validation_start, "2023-12-29")}
    fc = {}
    for name, spec in vf.CANDIDATES.items():
        for w, (a, b) in spans.items():
            fc[name, w] = vf.walk_forward(rows, spec["names"], cfg, a, b, spec.get("per_stock", False), spec.get("level", False))[0]
    naive = {w: vf.naive_forecast(rows, a, b) for w, (a, b) in spans.items()}
    table = []
    for name in list(vf.CANDIDATES) + ["naive monthly mean"]:
        row = {"model": name}
        for w in spans:
            fr = naive[w] if name.startswith("naive") else fc[name, w]
            # on the rows and dates the walk-forward scored, so every model is compared on the same sessions
            fr = fr[fr["target_date"].isin(fc["log_har", w]["target_date"])]
            d = vf.daily_loss(fr)
            row.update({f"{w[:3]}_qlike": d["qlike"].mean(), f"{w[:3]}_mse_log": d["mse_log"].mean(), f"{w[:3]}_mse_level_x1e6": d["mse_level"].mean() * 1e6})
        table.append(row)
    table = pd.DataFrame(table)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n" + table.round(4).to_string(index=False))
    chosen = vf.choose_baseline({r["model"]: r["dev_qlike"] for _, r in table.iterrows() if not r["model"].startswith("naive")})
    print(f"\nchosen on development QLIKE (ties within 0.5% to the simplest): {chosen}")
    coefs = vf.walk_forward(rows, vf.CANDIDATES[chosen]["names"], cfg, *spans["development"], vf.CANDIDATES[chosen].get("per_stock", False), vf.CANDIDATES[chosen].get("level", False))[1]
    print("its pooled coefficients over the development refits, mean and std:", coefs.mean().round(3).to_dict(), coefs.std().round(3).to_dict())
    print("\nthe validation gate: the chosen baseline must beat the monthly mean carried forward, on the same rows")
    keep = fc[chosen, "validation"][["symbol", "date"]]
    nv = naive["validation"].merge(keep, on=["symbol", "date"])
    a, b = vf.daily_loss(nv), vf.daily_loss(fc[chosen, "validation"])
    ok = True
    for loss in ("qlike", "mse_log"):
        g = vf.gap(a[loss], b[loss], cfg.seed)
        print(f"  {loss}: {b[loss].mean():.4f} against {a[loss].mean():.4f}; gap {g['gap']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}], t {g['t']:.1f}")
        ok = ok and g["hi"] < 0
    print("GATE PASSED" if ok else "GATE FAILED: stop; do not run the comparison on this baseline")
    return 0 if ok else 2


def reserved_stage(args, cfg: vf.VolConfig) -> int:
    import yaml
    if not args.score_reserved:
        print("STOPPED: this scores the reserved window. It runs once, after the freeze, with --score-reserved.")
        return 2
    with open(ROOT / "protocols" / "volatility_v1.yaml") as f:
        doc = yaml.safe_load(f)
    end = vf.lock_end(cfg, True)
    if not args.official:
        print("STOPPED: the registered target uses the official open: pass --official.")
        return 2
    p = vf.load_panel(Path(args.audit) / "sessions.parquet", args.reference, end, args.official, ROOT)
    joined = pf.load_joined(ROOT / "registry" / "sp500_join_dates.csv")
    rows = vf.build_rows(p, cfg, joined)
    print(vf.sample_sizes(rows, cfg).to_string(index=False))
    r = vf.pe_test(rows, doc["baseline"]["form"], cfg, cfg.test_start, cfg.test_end)
    print("\nPRIMARY: baseline against baseline + PE, log variance, Clark-West")
    print({k: v for k, v in r.items() if k != "by_year"})
    print(pd.DataFrame(r["by_year"]).T.round(5).to_string())
    print("\nVERDICT:", vf.verdict(r))
    print("\nROBUSTNESS (reported, not decisive)")
    idr = vf.build_rows(p, cfg, joined, col="rho")
    x = vf.pe_test(idr, doc["baseline"]["form"], cfg, cfg.test_start, cfg.test_end)
    print("intraday only (V_id):", {k: x[k] for k in ("rows", "clark_west_t", "qlike", "pe_coef_mean")})
    if args.rv:
        rv = pd.read_parquet(args.rv)
        q = p.merge(rv[["symbol", "date", "rv"]], on=["symbol", "date"], how="left")
        q["V_rv"] = q["r_on"] ** 2 + q["rv"]
        x = vf.pe_test(vf.build_rows(q, cfg, joined, col="V_rv"), doc["baseline"]["form"], cfg, cfg.test_start, cfg.test_end)
        print("realized variance from 5-minute bars plus the overnight:", {k: x[k] for k in ("rows", "clark_west_t", "qlike", "pe_coef_mean")})
    return 0


def rv_stage(args) -> int:
    """Realized variance from the 5-minute SIP cache for every stock (about 15 minutes in Colab), for the robustness target."""
    from src import audit
    from src.data import cache_path
    frames = []
    for i, sym in enumerate(pd.read_csv(ROOT / "universe" / "sp500.csv")["symbol"]):
        bars = audit.read_bars(cache_path(args.sip, sym, "5Min"))
        if bars is not None and not bars.empty:
            frames.append(vf.realized_variance(bars, sym))
        if (i + 1) % 100 == 0:
            print(f"{i + 1} symbols")
    pd.concat(frames, ignore_index=True).to_parquet(args.rv, index=False)
    print(f"wrote {args.rv}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["baseline", "rv", "reserved"], required=True)
    ap.add_argument("--sip", default="data/bars_sip")
    ap.add_argument("--audit", default="outputs/data_audit_v2")
    ap.add_argument("--reference", default="data/reference")
    ap.add_argument("--official", default=None)
    ap.add_argument("--rv", default=None)
    ap.add_argument("--score-reserved", action="store_true")
    args = ap.parse_args()
    cfg = vf.VolConfig()
    return {"baseline": baseline_stage, "rv": lambda a, c: rv_stage(a), "reserved": reserved_stage}[args.stage](args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
