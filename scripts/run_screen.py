#!/usr/bin/env python3
"""Run the gate over the cached universe.

    python scripts/run_screen.py --config configs/spy_gao.yaml
    python scripts/run_screen.py --config configs/sp500_multihour.yaml --workers 8

Writes outputs/<experiment>/screen.parquet. Every reading carries all three gate
statistics, so they can be compared later without screening again.
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from _common import DEFAULT_CONFIG, output_dir

from src.config import load_config
from src.data import load_universe, resolve_universe
from src.screening import apply_gate, capacity_diagnostic, screen_universe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=1, help="use os.cpu_count() on a large universe")
    args = parser.parse_args()
    if args.workers == 0:
        args.workers = os.cpu_count() or 1

    cfg = load_config(args.config)
    s = cfg.screening
    universe = resolve_universe(cfg.data.universe)
    bars = load_universe(cfg.data.cache_dir, universe, cfg.data.timeframe)
    if not bars:
        print(f"no cached bars in {cfg.data.cache_dir} — run scripts/fetch_data.py first")
        return 1
    if len(bars) < len(universe):
        print(f"note: {len(universe) - len(bars)} of {len(universe)} symbols have no cached data, skipping them")

    verbose = len(bars) <= 20
    seen = 0

    def progress(sym: str, table) -> None:
        nonlocal seen
        seen += 1
        if verbose:
            print(f"  {sym:6s} {len(table):5,} readings")
        elif seen % 25 == 0 or seen == len(bars):
            print(f"  ... {seen}/{len(bars)} symbols screened")

    print(f"screening {len(bars)} symbols  m={s.embedding_dim}  window={s.window}  "
          f"shuffles={s.n_surrogates}  workers={args.workers}")
    started = time.time()
    table = screen_universe(bars, s, progress=progress, workers=args.workers)
    if table.empty:
        print("screen produced no rows — window may exceed available history")
        return 1

    gated = apply_gate(table, s.gate_statistic, s.alpha, s.max_tie_fraction, s.min_ticks_per_bar)
    out = output_dir(cfg)
    gated.to_parquet(out / "screen.parquet")
    capacity_diagnostic(gated).to_csv(out / "capacity.csv", index=False)

    print(f"\n{len(gated):,} readings in {time.time() - started:.0f}s")
    print(f"refused: {gated['stale'].mean():.1%} stale, {gated['tick_limited'].mean():.1%} tick-limited")
    print(f"\nshare of readings that pass (pure noise would pass about {s.alpha:.0%}):")

    by_symbol = gated.groupby("symbol").agg(
        trend=("pass_trend", "mean"), entropy=("pass_entropy", "mean"),
        entropy_weighted=("pass_entropy_weighted", "mean"),
        pe_unweighted=("pe_unweighted", "mean"), pe_weighted=("pe_weighted", "mean"),
        ticks=("ticks_per_bar", "median"), dollar_volume=("dollar_volume", "median"),
    )
    header = f"  {'symbol':6s} {'trend':>6s} {'entropy':>8s} {'weighted':>9s}   {'mean PE':>8s} {'weighted':>9s}  {'cents/bar':>9s}"
    if verbose:
        print(header)
        for sym, r in by_symbol.iterrows():
            print(f"  {sym:6s} {r.trend:6.1%} {r.entropy:8.1%} {r.entropy_weighted:9.1%}   "
                  f"{r.pe_unweighted:8.3f} {r.pe_weighted:9.3f}  {r.ticks:9.1f}")
    else:
        print(f"  ({len(by_symbol)} symbols — most and least ordered by mean permutation entropy)")
        print(header)
        ranked = by_symbol.sort_values("pe_unweighted")
        for sym, r in pd.concat([ranked.head(8), ranked.tail(8)]).iterrows():
            print(f"  {sym:6s} {r.trend:6.1%} {r.entropy:8.1%} {r.entropy_weighted:9.1%}   "
                  f"{r.pe_unweighted:8.3f} {r.pe_weighted:9.3f}  {r.ticks:9.1f}")

    print(f"  {'all':6s} {gated['pass_trend'].mean():6.1%} {gated['pass_entropy'].mean():8.1%} "
          f"{gated['pass_entropy_weighted'].mean():9.1%}")

    capacity = capacity_diagnostic(gated)
    if len(capacity):
        print(f"\ncapacity: gate-selected names carry {capacity['dv_ratio'].median():.2f}x the "
              f"universe's own median dollar volume (1.0 = no systematic size skew)")
    print(f"\nwrote {out / 'screen.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
