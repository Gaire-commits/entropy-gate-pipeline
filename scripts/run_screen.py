#!/usr/bin/env python3
"""Run the entropy gate over the cached universe.

    python scripts/run_screen.py

Writes outputs/screen.parquet (every reading) and outputs/capacity.csv
(whether the gate is quietly selecting thinner names).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import load_universe
from src.screening import apply_gate, capacity_diagnostic, screen_universe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bars = load_universe(cfg.data.cache_dir, cfg.data.universe, cfg.data.timeframe)
    if not bars:
        print(f"no cached bars in {cfg.data.cache_dir} — run scripts/fetch_data.py first")
        return 1
    print(f"screening {len(bars)} symbols  m={cfg.screening.embedding_dim} window={cfg.screening.window}")

    table = screen_universe(bars, cfg.screening)
    if table.empty:
        print("screen produced no rows — window may exceed available history")
        return 1

    gated = apply_gate(table, cfg.screening.select_quantile)
    capacity = capacity_diagnostic(gated)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gated.to_parquet(out / "screen.parquet")
    capacity.to_csv(out / "capacity.csv", index=False)

    n_pass = int(gated["passed"].sum())
    print(f"\n{len(gated):,} readings over {gated['session_date'].nunique()} sessions")
    print(f"passed gate: {n_pass:,} ({n_pass/len(gated):.1%})   stale-rejected: {int(gated['stale'].sum()):,}")
    print(f"\nPE across universe: mean {gated['pe'].mean():.4f}  min {gated['pe'].min():.4f}  max {gated['pe'].max():.4f}")

    print("\nmost ordered symbols (lowest mean PE):")
    ranked = gated.groupby("symbol").agg(
        mean_pe=("pe", "mean"), mean_complexity=("complexity", "mean"),
        pass_rate=("passed", "mean"), median_dv=("dollar_volume", "median"),
    ).sort_values("mean_pe")
    for sym, row in ranked.head(8).iterrows():
        print(f"  {sym:6s} PE {row.mean_pe:.4f}  C {row.mean_complexity:.4f}  "
              f"pass {row.pass_rate:5.1%}  ${row.median_dv/1e6:8.1f}M")

    if not capacity.empty:
        ratio = capacity["dv_ratio"].median()
        print(f"\ncapacity check: selected names carry {ratio:.2f}x the universe median dollar volume")
        if ratio < 0.8:
            print("  -> gate leans toward thinner names; capacity ceiling is set by selection, not sizing")
    print(f"\nwrote {out/'screen.parquet'} and {out/'capacity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
