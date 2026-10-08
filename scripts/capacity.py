#!/usr/bin/env python3
"""E2: model or signal? A learning curve, a capacity ladder, and planted edges, judged by fixed rules.

    python scripts/capacity.py --config configs/e2_capacity.yaml --part curve      # E2a (5 cells)
    python scripts/capacity.py --config configs/e2_capacity.yaml --part planted    # E2b (8 cells)
    python scripts/capacity.py --config configs/e2_capacity.yaml --last-folds 4    # a quick look; cannot be judged

Each cell is a full walk-forward run of the tree model, saved fold by fold under
outputs/e2_capacity/cells/<cell>/ (interrupt and re-run to resume). Writes capacity.md and planted.md.
The rules are in the config; the method is in src/capacity.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import output_dir

from src.capacity import (
    calibrate_edge, capacity_verdict, cell_config, cell_name, paired_ic_difference, plant, planted_signal,
    planted_verdict, trim_train_days,
)
from src.config import load_config
from src.crosssection import cross_sectional_ic
from src.data import load_universe, resolve_universe
from src.dataset import build_dataset, walk_forward_splits
from src.experiment import KEY, load_predictions, run_arch
from src.holdout import apply_target, resolve_roles, restrict_folds


def fmt(x, spec="+.4f") -> str:
    return "—" if x is None or not np.isfinite(x) else format(x, spec)


def ic_cell(s: dict) -> str:
    return f"{fmt(s.get('ic'))} [{fmt(s.get('ic_lo'))}, {fmt(s.get('ic_hi'))}] · {fmt(s.get('t'), '+.1f')}"


def quiet(_):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e2_capacity.yaml")
    parser.add_argument("--part", choices=["curve", "planted", "all"], default="all")
    parser.add_argument("--last-folds", type=int, default=None, help="only the last N test quarters (not judged)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = output_dir(cfg)
    n_boot = int(cfg.evaluation.bootstrap)
    load, train, trade = resolve_roles(cfg.data)
    t0 = time.time()
    bars = load_universe(cfg.data.cache_dir, load, cfg.data.timeframe)
    if not bars:
        print(f"no cached bars under {cfg.data.cache_dir} — run scripts/fetch_data.py first")
        return 1
    data = build_dataset(bars, None, cfg)
    del bars
    min_rel = int(cfg.features.relative_min_peers)
    base = dict(data)
    keep = apply_target(base, "relative", train, trade, min_rel)
    folds = restrict_folds(walk_forward_splits(base["date"], cfg.validation), base["symbol"], train, trade, keep)
    if args.last_folds:
        folds = folds[-args.last_folds:]
    cells_dir = out / ("cells" if not args.last_folds else f"cells_last{args.last_folds}")
    judged = not args.last_folds
    print(f"{len(base['y']):,} samples, {len(folds)} folds, test {str(folds[0]['test_start'])[:10]} -> "
          f"{str(folds[-1]['test_end'])[:10]} ({time.time() - t0:.0f}s)")

    if args.part in ("curve", "all"):
        curve(cfg, base, folds, cells_dir, out, n_boot, judged, args.last_folds)
    if args.part in ("planted", "all"):
        planted(cfg, data, train, trade, min_rel, folds, cells_dir, out, n_boot, judged)
    return 0


def run_cell(name, data, folds, cfg, cells_dir) -> pd.DataFrame:
    t = time.time()
    ran = run_arch(data, folds, "gbm", 0, cfg, cells_dir / name, log=quiet)
    pred = load_predictions(cells_dir / name)
    print(f"  {name}: {ran} folds trained ({time.time() - t:.0f}s), {len(pred):,} predictions")
    return pred


def curve(cfg, base, folds, cells_dir, out, n_boot, judged, last_folds):
    e = cfg.capacity
    default = cell_name(e.default.train_days, e.default.leaves)
    cells = [(e.default.train_days, e.default.leaves)] + [(c.train_days, c.leaves) for c in e.cells]
    primary = set(resolve_universe(e.primary_universe))
    secondary = set(resolve_universe(e.secondary_universe))
    print("E2a: learning curve and capacity ladder")
    preds = {}
    for days, leaves in cells:
        name = cell_name(days, leaves)
        preds[name] = run_cell(name, base, trim_train_days(folds, base["date"], days), cell_config(cfg, leaves), cells_dir)

    rows, summ, paired = [], {}, {}
    for (days, leaves), (name, pred) in zip(cells, preds.items()):
        p1, p2 = pred[pred["symbol"].isin(primary)], pred[pred["symbol"].isin(secondary)]
        summ[name] = cross_sectional_ic(p1, min_peers=e.min_peers, n_boot=n_boot)
        if name != default:
            d = preds[default]
            paired[name] = paired_ic_difference(p1, d[d["symbol"].isin(primary)], e.min_peers, n_boot, level=e.ci_level)
        per_fold = pred.groupby("fold").agg(trees=("best_epoch", "first"), leaves=("n_params", "first"))
        rows.append({"cell": name, "train_days": days, "leaves": leaves, "primary": summ[name],
                     "secondary": cross_sectional_ic(p2, min_peers=e.min_peers, n_boot=n_boot),
                     "trees": float(per_fold["trees"].mean() + 1), "total_leaves": float(per_fold["leaves"].mean())})
    verdict = capacity_verdict(summ, paired, default, e.t_bar)

    check = "not checked (saved replication predictions not found, or a --last-folds run)"
    rep = Path(e.replication_dir)
    if not last_folds and (rep / "predictions" / "gbm").exists():
        saved = load_predictions(rep)
        saved = saved[saved["arch"] == "gbm"][KEY + ["prob"]]
        both = preds[default][KEY + ["prob"]].merge(saved, on=KEY, suffixes=("", "_saved"))
        diff = float((both["prob"] - both["prob_saved"]).abs().max()) if len(both) else np.nan
        check = (f"{len(both):,} of {len(preds[default]):,} predictions matched to the replication's saved `gbm`; "
                 f"largest difference {diff:.1e}" + (" (identical)" if diff < 1e-9 else " — NOT IDENTICAL"))

    lines = [f"# {cfg.experiment}: does more data or a bigger model find more? (E2a)", "",
             f"**Verdict: {verdict['level'].upper()}** — {verdict['reason']}." if judged else
             "**Not judged**: a --last-folds run covers only part of the test quarters.", "",
             f"Tree model on the relative target, trained on the 418 stocks outside the Nasdaq-100. IC is the "
             f"cross-sectional rank IC on the 416 training stocks scored out of time (primary) and on the 85 Nasdaq-100 "
             f"stocks (reported). **vs default** is the IC difference from the default cell ({default}) on the same "
             f"moments, with a {e.ci_level:.0%} day-block interval. **trees**: chosen on validation, averaged over folds.", "",
             "| cell | training days | leaves per tree | trees | IC on 416 [95%] · t | vs default [99%] | IC on 85 [95%] · t |",
             "|---|---:|---:|---:|---|---|---|"]
    for r in rows:
        p = paired.get(r["cell"])
        vs = "default" if p is None else f"{fmt(p['diff'])} [{fmt(p['lo'])}, {fmt(p['hi'])}]"
        lines.append(f"| {r['cell']} | {r['train_days']} | {r['leaves']} | {r['trees']:.0f} | {ic_cell(r['primary'])} | "
                     f"{vs} | {ic_cell(r['secondary'])} |")
    lines += ["", f"Default cell reproduction: {check}.", "",
              f"The data could detect an IC of about {fmt(summ[default].get('ic_min_detectable'), '.4f')} on the 416 "
              "(2.8 standard errors). A book needed about 0.0138 there to pay 2 bps."]
    text = "\n".join(lines) + "\n"
    (out / "capacity.md").write_text(text)
    pd.DataFrame([{**{k: v for k, v in r.items() if not isinstance(v, dict)},
                   **{f"primary_{k}": v for k, v in r["primary"].items()},
                   **{f"vs_default_{k}": v for k, v in paired.get(r["cell"], {}).items()}} for r in rows]).to_csv(
        out / "capacity.csv", index=False)
    print("\n" + text)


def planted(cfg, data, train, trade, min_rel, folds, cells_dir, out, n_boot, judged):
    pl, e = cfg.planted, cfg.capacity
    channels = list(cfg.features.channels)
    test = np.concatenate([f["test"] for f in folds])
    print("E2b: planted edges")
    rows = []
    for shape in pl.shapes:
        s = planted_signal(data, shape, channels)
        for target in pl.targets:
            lam = calibrate_edge(data, s, float(target), min_peers=pl.min_peers)
            d = plant(data, s, lam)
            apply_target(d, "relative", train, trade, min_rel)
            name = f"planted_{shape}_{target}"
            pred = run_cell(name, d, folds, cell_config(cfg, e.default.leaves), cells_dir)
            ic = cross_sectional_ic(pred, min_peers=pl.min_peers, n_boot=n_boot)
            oracle = cross_sectional_ic(pd.DataFrame({"date": d["date"][test], "bar_index": d["bar_index"][test],
                                                      "symbol": d["symbol"][test], "prob": s[test], "ret": d["ret"][test]}),
                                        min_peers=pl.min_peers, n_boot=n_boot)
            rows.append({"shape": shape, "target": float(target), "lambda_bps": lam, "oracle_ic": oracle.get("ic"),
                         "ic": ic.get("ic"), "ic_lo": ic.get("ic_lo"), "ic_hi": ic.get("ic_hi"), "t": ic.get("t"),
                         "captured": ic.get("ic", np.nan) / oracle.get("ic", np.nan)})
            print(f"    {shape} {target}: edge {lam:.2f} bps per sd, oracle IC {fmt(oracle.get('ic'))}, "
                  f"recovered {fmt(ic.get('ic'))} (t {fmt(ic.get('t'), '+.1f')})")
    verdict = planted_verdict(rows, float(pl.break_even), pl.t_bar, pl.share_of_oracle)

    real = cells_dir / cell_name(e.default.train_days, e.default.leaves)
    real_ic = cross_sectional_ic(load_predictions(real), min_peers=pl.min_peers, n_boot=n_boot) if real.exists() else {}
    lines = [f"# {cfg.experiment}: can the pipeline find an edge of break-even size? (E2b)", ""]
    if judged:
        for shape, v in verdict.items():
            lines.append(f"- **{shape}: {'ADEQUATE' if v['adequate'] else 'NOT ADEQUATE'}** at the break-even IC "
                         f"{pl.break_even}; smallest planted IC recovered: {v['smallest_recovered'] or 'none'}.")
    else:
        lines.append("**Not judged**: a --last-folds run covers only part of the test quarters.")
    lines += ["", "The real returns plus an edge built from the model's own inputs, sized so the edge itself scores the "
              "target IC (**oracle**: the edge's own IC on the test rows). **recovered**: the tree model's IC, trained and "
              f"scored exactly as on the real data, on every stock. Recovered means t >= {pl.t_bar:g} and at least "
              f"{pl.share_of_oracle:.0%} of the oracle IC.", "",
              "| edge | target IC | edge size (bps per sd) | oracle IC | recovered IC [95%] | t | share of oracle |",
              "|---|---:|---:|---:|---|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['shape']} | {r['target']} | {r['lambda_bps']:.2f} | {fmt(r['oracle_ic'])} | "
                     f"{fmt(r['ic'])} [{fmt(r['ic_lo'])}, {fmt(r['ic_hi'])}] | {fmt(r['t'], '+.1f')} | "
                     f"{fmt(r['captured'], '.0%')} |")
    lines += ["", f"**Real data, no edge planted** (E2a's default cell, every stock): IC {ic_cell(real_ic)}." if real_ic else
              "Real data: run `--part curve` for the zero-edge comparison.", "",
              "A planted edge lives in the model's inputs by construction. This shows the models and the measurement "
              "would find an edge of that size there; it cannot speak for information the model never sees."]
    text = "\n".join(lines) + "\n"
    (out / "planted.md").write_text(text)
    pd.DataFrame(rows).to_csv(out / "planted.csv", index=False)
    print("\n" + text)


if __name__ == "__main__":
    raise SystemExit(main())
