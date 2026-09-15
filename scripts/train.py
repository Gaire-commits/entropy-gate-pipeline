#!/usr/bin/env python3
"""Walk-forward training.

Experiment 1 (does the gate help?):
    python scripts/train.py --arch resnet1d --gate
    python scripts/train.py --arch resnet1d --no-gate

Experiment 3 (does depth beat a linear model?):
    python scripts/train.py --arch logreg --gate
    python scripts/train.py --arch inceptiontime --gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import load_universe
from src.dataset import build_dataset, walk_forward_splits
from src.models import count_parameters
from src.training import evaluate, predict, train_fold


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--arch", default=None)
    parser.add_argument("--gate", dest="gate", action="store_true", default=True)
    parser.add_argument("--no-gate", dest="gate", action="store_false")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    arch = args.arch or cfg.model.arch

    bars = load_universe(cfg.data.cache_dir, cfg.data.universe, cfg.data.timeframe)
    if not bars:
        print(f"no cached bars in {cfg.data.cache_dir} — run scripts/fetch_data.py first")
        return 1

    screen_path = Path(args.out) / "screen.parquet"
    gated = None
    if args.gate:
        if not screen_path.exists():
            print(f"{screen_path} missing — run scripts/run_screen.py first, or pass --no-gate")
            return 1
        gated = pd.read_parquet(screen_path)

    data = build_dataset(bars, gated, cfg, use_gate=args.gate)
    folds = walk_forward_splits(data["date"], cfg.validation)
    if not folds:
        print("not enough sessions for one walk-forward fold — shorten train/val/test in config")
        return 1

    print(f"arch={arch}  gate={'on' if args.gate else 'off'}  "
          f"samples={len(data['y']):,}  shape={data['X'].shape[1:]}  folds={len(folds)}")
    print(f"class balance: {data['y'].mean():.3f} up\n")

    rows = []
    for i, fold in enumerate(folds):
        tr, va, te = fold["train"], fold["val"], fold["test"]
        if min(len(tr), len(va), len(te)) < 50:
            continue

        model, info = train_fold(
            data["X"][tr], data["y"][tr], data["X"][va], data["y"][va],
            cfg.model, arch=arch, verbose=args.verbose,
        )
        metrics = evaluate(
            predict(model, data["X"][te]), data["y"][te], data["ret"][te],
            threshold=args.threshold, cost_bps=args.cost_bps,
            trading_days=len(np.unique(data["date"][te])),
        )
        metrics |= {
            "fold": i,
            "test_start": str(fold["test_start"])[:10],
            "test_end": str(fold["test_end"])[:10],
            "best_epoch": info["best_epoch"],
        }
        rows.append(metrics)
        print(f"fold {i:2d} [{metrics['test_start']}..{metrics['test_end']}]  "
              f"acc {metrics['accuracy']:.3f}  hit {metrics['hit_rate']:.3f}  "
              f"net {metrics['mean_return_bps_net']:+7.2f}bps  trades {metrics['n_trades']:,}")

    if not rows:
        print("every fold was too small to train")
        return 1

    results = pd.DataFrame(rows)
    tag = f"{arch}_{'gated' if args.gate else 'ungated'}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / f"walkforward_{tag}.csv", index=False)

    summary = {
        "arch": arch,
        "gate": args.gate,
        "cost_bps": args.cost_bps,
        "n_params": count_parameters(model),
        "folds": len(results),
        "accuracy_mean": results["accuracy"].mean(),
        "accuracy_std": results["accuracy"].std(),
        "hit_rate_mean": results["hit_rate"].mean(),
        "net_bps_mean": results["mean_return_bps_net"].mean(),
        "gross_bps_mean": results["mean_return_bps_gross"].mean(),
        "breakeven_cost_bps": results["breakeven_cost_bps"].mean(),
        "folds_profitable_net": int((results["mean_return_bps_net"] > 0).sum()),
    }
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"\n{'='*60}\n{arch}  gate={'on' if args.gate else 'off'}  ({summary['n_params']:,} params)")
    print(f"accuracy      {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    print(f"gross return  {summary['gross_bps_mean']:+.2f} bps/trade")
    print(f"net return    {summary['net_bps_mean']:+.2f} bps/trade  (after {args.cost_bps} bps round trip)")
    print(f"breakeven at  {summary['breakeven_cost_bps']:.2f} bps — costs above this erase the edge")
    print(f"profitable in {summary['folds_profitable_net']}/{len(results)} folds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
