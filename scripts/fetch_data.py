#!/usr/bin/env python3
"""Download Alpaca bars into the local parquet cache, skipping what is already there.

    python scripts/fetch_data.py
    python scripts/fetch_data.py --config configs/spy_gao.yaml
    python scripts/fetch_data.py --refresh          # re-download everything

An explicit `end` date is inclusive. On repeat runs only the days since the last
cached day are downloaded, so extending `end` (or setting it to "today") tops the
cache up instead of starting over.
"""

from __future__ import annotations

import argparse

import pandas as pd

from _common import DEFAULT_CONFIG

from src.config import load_config
from src.data import (
    completeness, fetch_symbol, get_client, load_raw, load_symbol, merge_bars, plan_fetch, regular_hours,
    resolve_end, save_symbol, to_eastern,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    d = load_config(args.config).data
    print(f"{len(d.universe)} symbols  {d.start} -> {d.end} (through {resolve_end(d.end) - pd.Timedelta(seconds=1):%Y-%m-%d})"
          f"  @{d.timeframe}  feed={d.feed}")
    client = None
    for symbol in d.universe:
        plan = None if args.refresh else plan_fetch(d.cache_dir, symbol, d.timeframe, d.start, d.end, d.feed, d.adjustment)
        if args.refresh:
            plan = (pd.Timestamp(d.start), resolve_end(d.end))
        if plan is None:
            status = "cached"
        else:
            client = client or get_client()
            first, stop = plan
            raw = fetch_symbol(client, symbol, str(first), str(stop), d.timeframe, d.feed, d.adjustment)
            cached = None if args.refresh else load_raw(d.cache_dir, symbol, d.timeframe)
            fresh = regular_hours(to_eastern(raw)) if not raw.empty else raw
            if fresh.empty and cached is None:
                print(f"  {symbol:5s} no bars returned")
                continue
            before = 0 if cached is None else len(cached)
            merged = merge_bars(cached, fresh)
            save_symbol(merged, d.cache_dir, symbol, d.timeframe, d.start, d.end, d.feed, d.adjustment)
            status = f"+{len(merged) - before:,} bars" if cached is not None else "downloaded"

        grid = load_symbol(d.cache_dir, symbol, d.timeframe)
        ts = grid.index.get_level_values("timestamp")
        print(f"  {symbol:5s} {status:14s} {ts[0].date()} -> {ts[-1].date()}  "
              f"{grid['session_id'].nunique():5,} sessions  {completeness(grid):6.1%} of 5-min slots have a bar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
