#!/usr/bin/env python3
"""The volatility protocol, by stage (protocols/volatility_v2.yaml; prints everything, writes nothing unless told to).

    python scripts/vol_forecast.py --stage baseline --sessions sessions.parquet --reference data/reference --official data/reference/official_prices.parquet
    python scripts/vol_forecast.py --stage rv --sip data/bars_sip --rv data/reference/rv5.parquet
    python scripts/vol_forecast.py --stage reserved --score-reserved --sessions ... --reference ... --official ... --rv ...

`baseline` applies the protocol's rule for the HAR form on development and its gate on validation. No model in it uses PE, and it loads nothing
after the lock. `rv` builds the realized-variance robustness target. `reserved` runs the one registered comparison, once, on the reserved test window.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import ROOT

from src import portfolio as pf, volforecast as vf, volpanel

SPANS = {"development": ("2016-01-01", "2021-12-31"), "validation": ("2022-01-04", "2023-12-29")}
PROTOCOL = ROOT / "protocols" / "volatility_v2.yaml"


def gate(rows: pd.DataFrame, form: str, cfg: vf.VolConfig) -> bool:
    """On validation, the baseline's QLIKE and log loss must each be lower than the monthly mean carried forward, on the same rows, with
    the 95% interval of each gap below zero."""
    spec = vf.CANDIDATES[form]
    fc, _ = vf.walk_forward(rows, spec["names"], cfg, *SPANS["validation"], spec.get("per_stock", False), spec.get("level", False))
    nv = vf.naive_forecast(rows, *SPANS["validation"]).merge(fc[["symbol", "date"]], on=["symbol", "date"])
    a, b = vf.daily_loss(nv), vf.daily_loss(fc)
    ok = True
    for loss in ("qlike", "mse_log"):
        g = vf.gap(a[loss], b[loss], cfg)
        print(f"  {loss}: {b[loss].mean():.4f} against {a[loss].mean():.4f}; gap {g['gap']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}], t {g['t']:.1f}")
        ok = ok and g["hi"] < 0
    print(f"  conversion from logs: mean smearing factor {fc['smear'].mean():.3f}; mean realized over mean forecast variance {vf.calibration(fc):.3f}")
    print("GATE PASSED" if ok else "GATE FAILED: stop; the comparison does not run on this baseline")
    return ok


def baseline_stage(args, cfg: vf.VolConfig) -> int:
    end = vf.lock_end(cfg)
    p = vf.load_panel(args.sessions, args.reference, end, args.official, ROOT)
    print(f"panel through {p['date'].max().date()} (locked). The open: {'official, and no V where it is missing' if args.official else 'THE VENDOR OPEN, a stand-in: this is not the registered target'}")
    joined = pf.load_joined(ROOT / "registry" / "sp500_join_dates.csv")
    rows = vf.build_rows(p, cfg, joined)
    print(vf.sample_sizes(rows, cfg).to_string(index=False))
    named = p[(p["reason"] != "") & p["in_span"] & (p["date"] >= "2016-01-05")]
    print("\nwhy V is missing, stock-sessions through the lock:", named["reason"].value_counts().to_dict())
    fc = {(name, w): vf.walk_forward(rows, spec["names"], cfg, a, b, spec.get("per_stock", False), spec.get("level", False))[0]
          for name, spec in vf.CANDIDATES.items() for w, (a, b) in SPANS.items()}
    naive = {w: vf.naive_forecast(rows, a, b) for w, (a, b) in SPANS.items()}
    table = []
    for name in list(vf.CANDIDATES) + ["naive monthly mean"]:
        row = {"model": name}
        for w in SPANS:
            fr = naive[w] if name.startswith("naive") else fc[name, w]
            fr = fr[fr["target_date"].isin(fc["log_har", w]["target_date"])]         # the same sessions for every model
            d = vf.daily_loss(fr)
            row.update({f"{w[:3]}_qlike": d["qlike"].mean(), f"{w[:3]}_log_loss": d["mse_log"].mean(), f"{w[:3]}_realized_over_forecast": vf.calibration(fr)})
        table.append(row)
    table = pd.DataFrame(table)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n" + table.round(4).to_string(index=False))
    chosen = vf.choose_baseline({r["model"]: r["dev_qlike"] for _, r in table.iterrows() if not r["model"].startswith("naive")})
    print(f"\nthe rule's choice on development QLIKE (ties within 0.5% to the simplest): {chosen}")
    spec = vf.CANDIDATES[chosen]
    coefs = vf.walk_forward(rows, spec["names"], cfg, *SPANS["development"], spec.get("per_stock", False), spec.get("level", False))[1]
    print("its pooled coefficients over the development refits, mean and std:", coefs.mean().round(3).to_dict(), coefs.std().round(3).to_dict())
    import yaml
    with open(PROTOCOL) as f:
        form = yaml.safe_load(f)["baseline"]["form"]
    if chosen != form:
        print(f"STOP: the protocol's baseline is {form}, but its rule now picks {chosen}. File an amendment, with the reason, before going on.")
        return 2
    print("\nthe validation gate:")
    return 0 if gate(rows, form, cfg) else 2


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


def report(r: dict) -> None:
    c = r["log_loss"]
    q = r["qlike"]
    print(f"  rows {r['rows']:,}, target dates {r['dates']}, {r['first']} to {r['last']}")
    t, five, holm = r["clark_west_t"], r["clark_west_critical"], r["clark_west_critical_holm"]
    print(f"  1. Clark-West (is there incremental predictive content?): mean {r['clark_west_mean']:+.5f}, t {t:.2f} with {vf.VolConfig().hac_lags} Newey-West lags; "
          f"clears {five} (5%): {'yes' if t >= five else 'no'}; clears {holm} (2.5%, for Holm): {'yes' if t >= holm else 'no'}")
    print(f"     of which the adjustment for estimation noise: {r['adjustment_mean']:.6f}")
    print(f"  2. Unadjusted paired log loss, PE minus baseline (did errors actually fall?): {c['gap']:+.5f} [{c['lo']:+.5f}, {c['hi']:+.5f}], t {c['t']:.2f}; baseline {r['baseline_log_loss']:.4f}")
    print(f"  3. QLIKE, PE minus baseline: {q['gap']:+.5f} [{q['lo']:+.5f}, {q['hi']:+.5f}], t {q['t']:.2f}; baseline {r['baseline_qlike']:.4f}")
    print(f"     level forecasts, realized over forecast: baseline {r['calibration_baseline']:.3f}, with PE {r['calibration_pe']:.3f}")
    print(f"  PE coefficient: mean {r['pe_coef_mean']:+.4f}, same sign on {r['pe_coef_same_sign']:.0%} of refits")


def reserved_stage(args, cfg: vf.VolConfig) -> int:
    import yaml
    if not args.score_reserved:
        print("STOPPED: this scores the reserved window. It runs once, after the freeze, with --score-reserved.")
        return 2
    if not args.official:
        print("STOPPED: the registered target uses the official open: pass --official.")
        return 2
    with open(PROTOCOL) as f:
        doc = yaml.safe_load(f)
    form = doc["baseline"]["form"]
    locked = vf.build_rows(vf.load_panel(args.sessions, args.reference, vf.lock_end(cfg), args.official, ROOT), cfg, pf.load_joined(ROOT / "registry" / "sp500_join_dates.csv"))
    print("the validation gate first:")
    if not gate(locked, form, cfg):
        print("STOPPED: the gate failed, so the reserved window was not scored.")
        return 2
    end = vf.lock_end(cfg, True)
    p = vf.load_panel(args.sessions, args.reference, end, args.official, ROOT)
    joined = pf.load_joined(ROOT / "registry" / "sp500_join_dates.csv")
    rows = vf.build_rows(p, cfg, joined)
    print(vf.sample_sizes(rows, cfg).to_string(index=False))
    r = vf.pe_test(rows, form, cfg, cfg.test_start, cfg.test_end)
    print("\nPRIMARY: baseline against baseline + PE, on the reserved window")
    report(r)
    print(pd.DataFrame(r["by_year"]).T.round(5).to_string())
    print("\nVERDICT:", vf.verdict(r))
    print("\nROBUSTNESS (reported, not decisive)")
    print("the range alone (V_id):")
    report(vf.pe_test(vf.build_rows(p, cfg, joined, col="rho"), form, cfg, cfg.test_start, cfg.test_end))
    print("the vendor's open where the official open is missing:")
    report(vf.pe_test(vf.build_rows(vf.load_panel(args.sessions, args.reference, end, args.official, ROOT, open_fallback=True), cfg, joined), form, cfg, cfg.test_start, cfg.test_end))
    if args.rv:
        q = p.merge(pd.read_parquet(args.rv)[["symbol", "date", "rv"]], on=["symbol", "date"], how="left")
        q["V_rv"] = q["r_on"] ** 2 + q["rv"]
        print("realized variance from 5-minute bars plus the overnight:")
        report(vf.pe_test(vf.build_rows(q, cfg, joined, col="V_rv"), form, cfg, cfg.test_start, cfg.test_end))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["baseline", "rv", "reserved"], required=True)
    ap.add_argument("--sip", default="data/bars_sip")
    ap.add_argument("--sessions", default="sessions.parquet")
    ap.add_argument("--reference", default="data/reference")
    ap.add_argument("--official", default=None)
    ap.add_argument("--rv", default=None)
    ap.add_argument("--score-reserved", action="store_true")
    args = ap.parse_args()
    cfg = vf.VolConfig()
    return {"baseline": baseline_stage, "rv": lambda a, c: rv_stage(a), "reserved": reserved_stage}[args.stage](args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
