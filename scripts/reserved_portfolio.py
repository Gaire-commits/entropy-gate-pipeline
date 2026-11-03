#!/usr/bin/env python3
"""The portfolio protocol (protocols/portfolio_v1.yaml), on the frozen rules. Prints everything, writes nothing.

    python scripts/reserved_portfolio.py --reference data/reference                     development and validation; nothing later is loaded
    python scripts/reserved_portfolio.py --reference data/reference --score-reserved    once, after the freeze, the reserved window too

It stops if the benchmarks disagree with the independently recorded returns in the protocol.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from _common import ROOT

import portfolio_rules as rules
from src import portfolio as pf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/reference")
    ap.add_argument("--benchmarks", default=None, help="a saved benchmark file; by default fetched from Alpaca")
    ap.add_argument("--score-reserved", action="store_true")
    args = ap.parse_args()
    cfg, doc = pf.load_protocol(ROOT / "protocols" / "portfolio_v1.yaml")
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    if args.score_reserved:
        print("SCORING THE RESERVED WINDOW. Protocol frozen on", doc["registered"], "- this period is reserved for this specification but previously "
              "exposed through related research (protocols/reserved_period_disclosure.md).\n")
    m = rules.load_market(args.reference, cfg, args.score_reserved)
    index = m.close.index
    end = pf.scored_end(index, cfg, args.score_reserved)
    print(f"data through {index.max().date()}; {m.close.shape[1]} symbols; look-ahead check:", pf.no_lookahead_check(m, cfg, index[252 + cfg.min_train_sessions + 250]))

    path = args.benchmarks or f"{args.reference}/benchmarks_to_{pf.lock_date(cfg, args.score_reserved).date()}.parquet"
    bench = pf.fetch_benchmarks(path, "2016-01-04", str(pf.lock_date(cfg, args.score_reserved).date()))
    guard = pf.check_benchmarks(bench, doc["benchmark_reference"], m.truncated)
    print("\nbenchmarks against the recorded total returns (largest gap "
          f"{(guard['this_series'] - guard['reference']).abs().max():.2%}):", "PASS" if guard["ok"].all() else "FAIL")
    if not guard["ok"].all():
        print(guard[~guard["ok"]].to_string(index=False))
        print("STOPPED: the benchmark series is not the one recorded in the protocol. Nothing was scored.")
        return 2
    rf = pf.benchmark_returns(bench, "BIL", cfg.execution, m.truncated)
    spy = pf.benchmark_returns(bench, "SPY", cfg.execution, m.truncated)

    feats, eligible = pf.build_features(m, cfg)
    target = pf.forward_returns(m, cfg)
    dates = pf.decision_dates(index, cfg, args.score_reserved)
    print(f"decisions {len(dates)}, {dates[0].date()} to {dates[-1].date()}, every {pf.step(cfg)} sessions; returns scored to {end.date()}")
    res = {n: pf.run_model(n, feats, eligible, target, m, dates, cfg, end) for n in pf.MODELS}

    print("\nthe target against the book, per tranche:")
    for k, tr in enumerate(pf.tranche_targets(res["base"]["forecasts"], m, cfg)):
        c = pf.reconcile_target(tr, m, cfg, end)
        print(f"  tranche {k}: {len(c)} holding periods, largest gap {c['gap'].abs().max():.1e}, names with no fill {int(c['target'].isna().sum())}")

    ew = pf.simulate(pf.equal_weight_targets(eligible, dates), m, cfg, end)
    series = {n: res[n]["daily"] for n in res}
    series["equal-weight"] = ew
    series["market (SPY)"] = spy
    print("\n" + pf.scorecard(series, rf, cfg, {n: res[n]["ic"] for n in res}).to_string(index=False))

    base = res["base"]["daily"]
    print(f"\nbase: median names {base['names'].median():.0f}; cost per dollar traded {(base['cost'].sum() / (2 * base['turnover'].sum())) * 1e4:.1f} bp; largest trade {base['max_participation'].max():.2%} of daily dollar volume")
    top = base.reindex(base["top_contribution"].abs().sort_values(ascending=False).index).head(5)
    print("largest single-stock days in the base book:", [(str(d.date()), s, round(c, 4)) for d, s, c in zip(top.index, top["top_symbol"], top["top_contribution"])])
    print("\nthe base model's coefficients:")
    print(pf.coefficient_table(res["base"]["coefs"]).to_string())

    v, h = pd.Timestamp(cfg.validation_start), pd.Timestamp(cfg.holdout_start)
    spans = {"development": (None, v - pd.Timedelta(days=1)), "validation": (v, h - pd.Timedelta(days=1))}
    if args.score_reserved:
        spans["reserved"] = (h, None)
    rows = []
    for name in res:
        for span, (a, b) in spans.items():
            d = res[name]["daily"]["net"].loc[a:b]
            active = (d - ew["net"].reindex(d.index).fillna(0.0)).to_numpy()
            lo, hi = pf.block_bootstrap(active, lambda x: float(np.mean(x) * 252), seed=cfg.seed)
            rows.append({"model": name, "window": span, "active_ann": float(active.mean() * 252), "lo": lo, "hi": hi, "information_ratio": float(active.mean() / active.std() * np.sqrt(252))})
    print("\nnet active return against equal-weight, annual, with its 95% interval. The rule was the best of 24 candidates seen, on development and validation:")
    print(pd.DataFrame(rows).to_string(index=False))
    rows = []
    for name in ("base_pe", "base_pe_x"):
        for span, (a, b) in spans.items():
            c = pf.paired_comparison(res["base"], res[name], target, eligible, rf, a, b, cfg.seed)
            rows.append({"model": name, "window": span, **{k: c[k] for k in ("dates", "ic_small", "ic_large", "ic_gap", "ic_gap_lo", "ic_gap_hi", "clark_west_t", "net_return_gap_ann", "net_gap_lo", "net_gap_hi", "sharpe_gap", "sharpe_gap_lo", "sharpe_gap_hi")}, "verdict": pf.verdict(c)})
    comparison = pd.DataFrame(rows)
    print("\nPE against base (intervals 95%, block bootstrap):")
    print(comparison.drop(columns="verdict").to_string(index=False))
    print()
    for r in rows:
        print(f"{r['model']:10s} {r['window']:12s} {r['verdict']}")
    print("\nPE terms in the largest model:")
    print(pf.coefficient_table(res["base_pe_x"]["coefs"]).loc[pf.PE_FEATURE + pf.INTERACTIONS].to_string())

    start = h if args.score_reserved else v
    variants = {"0 bp": replace(cfg, cost_bps=0, impact=0), "5 bp, no impact": replace(cfg, impact=0), "as frozen": cfg, "10 bp": replace(cfg, cost_bps=10), "20 bp": replace(cfg, cost_bps=20), "$100M": replace(cfg, aum=100e6)}
    rows = []
    for label, c in variants.items():
        for name in res:
            p = pf.performance(pf.resimulate(res[name], m, c)["net"].loc[start:], rf)
            rows.append({"costs": label, "model": name, "return_ann": p["return_ann"], "sharpe": p["sharpe"]})
        p = pf.performance(pf.simulate(pf.equal_weight_targets(eligible, dates), m, c, end)["net"].loc[start:], rf)
        rows.append({"costs": label, "model": "equal-weight", "return_ann": p["return_ann"], "sharpe": p["sharpe"]})
    s = pd.DataFrame(rows)
    print(f"\nunder other costs, from {start.date()}, net annual return, then Sharpe:")
    print(s.pivot(index="costs", columns="model", values="return_ann").reindex(variants).to_string())
    print(s.pivot(index="costs", columns="model", values="sharpe").reindex(variants).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
