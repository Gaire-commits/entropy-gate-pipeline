#!/usr/bin/env python3
"""Run the gate over the cached universe.

    python scripts/run_screen.py --config configs/spy_gao.yaml

Writes outputs/<experiment>/screen.parquet. Every reading carries all three gate
statistics, so they can be compared later without screening again.
"""

from __future__ import annotations

import argparse
import time

from _common import DEFAULT_CONFIG, output_dir

from src.config import load_config
from src.data import load_universe
from src.screening import apply_gate, capacity_diagnostic, screen_universe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_config(args.config)
    s = cfg.screening
    bars = load_universe(cfg.data.cache_dir, cfg.data.universe, cfg.data.timeframe)
    if not bars:
        print(f"no cached bars in {cfg.data.cache_dir} — run scripts/fetch_data.py first")
        return 1

    print(f"screening {len(bars)} symbols  m={s.embedding_dim}  window={s.window}  shuffles={s.n_surrogates}")
    started = time.time()
    table = screen_universe(bars, s, progress=lambda sym, t: print(f"  {sym:5s} {len(t):5,} readings"))
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
        ticks=("ticks_per_bar", "median"),
    )
    print(f"  {'symbol':6s} {'trend':>6s} {'entropy':>8s} {'weighted':>9s}   {'mean PE':>8s} {'weighted':>9s}  {'cents/bar':>9s}")
    for sym, r in by_symbol.iterrows():
        print(f"  {sym:6s} {r.trend:6.1%} {r.entropy:8.1%} {r.entropy_weighted:9.1%}   "
              f"{r.pe_unweighted:8.3f} {r.pe_weighted:9.3f}  {r.ticks:9.1f}")
    print(f"  {'all':6s} {gated['pass_trend'].mean():6.1%} {gated['pass_entropy'].mean():8.1%} "
          f"{gated['pass_entropy_weighted'].mean():9.1%}")
    print(f"\nwrote {out / 'screen.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
