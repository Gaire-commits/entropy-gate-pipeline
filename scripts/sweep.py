#!/usr/bin/env python3
"""Walk-forward sweep: every arch x seed on every fold. Safe to interrupt and re-run.

    python scripts/sweep.py --config configs/spy_gao.yaml
    python scripts/sweep.py --config configs/etf_intraday.yaml --archs logreg resnet1d --seeds 0
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from _common import DEFAULT_CONFIG, output_dir

from src.baselines import is_rule
from src.config import load_config
from src.data import load_universe
from src.dataset import build_dataset, walk_forward_splits
from src.experiment import run_arch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--archs", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = output_dir(cfg)
    archs = args.archs or cfg.sweep.archs
    seeds = args.seeds if args.seeds is not None else cfg.sweep.seeds

    bars = load_universe(cfg.data.cache_dir, cfg.data.universe, cfg.data.timeframe)
    if not bars:
        print(f"no cached bars in {cfg.data.cache_dir} — run scripts/fetch_data.py first")
        return 1
    screen_path = out / "screen.parquet"
    screen = pd.read_parquet(screen_path) if screen_path.exists() else None
    if screen is None:
        print(f"note: {screen_path} missing, so predictions carry no gate columns and Q1 is skipped")

    data = build_dataset(bars, screen, cfg)
    folds = walk_forward_splits(data["date"], cfg.validation)
    if not folds:
        print("not enough trading days for one walk-forward fold — shorten the validation windows")
        return 1
    print(f"{cfg.experiment}: {len(data['y']):,} samples  shape {data['X'].shape[1:]}  "
          f"{len(folds)} folds  test {str(folds[0]['test_start'])[:10]} -> {str(folds[-1]['test_end'])[:10]}")

    for arch in archs:
        for seed in ([seeds[0]] if is_rule(arch) else seeds):
            started = time.time()
            print(f"\n{arch}  seed {seed}")
            ran = run_arch(data, folds, arch, seed, cfg, out)
            done = "all folds already done" if ran == 0 else f"{ran} folds in {time.time() - started:.0f}s"
            print(f"  {done}")
    print("\nnext: python scripts/summarize.py --config", args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
