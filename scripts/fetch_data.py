#!/usr/bin/env python3
"""Download Alpaca bars into the local parquet cache, skipping what is already there.

    python scripts/fetch_data.py
    python scripts/fetch_data.py --config configs/spy_gao.yaml
    python scripts/fetch_data.py --config configs/sp500_multihour.yaml --workers 6
    python scripts/fetch_data.py --refresh          # re-download everything

An explicit `end` date is inclusive. On repeat runs only the days since the last
cached day are downloaded, so extending `end` (or setting it to "today") tops the
cache up instead of starting over.

`--workers` fetches multiple symbols concurrently (each on its own client, since
sharing one HTTP session across threads isn't guaranteed safe). Past a handful of
workers you're mostly testing Alpaca's rate limit rather than your network, so
raise it gradually on a large universe rather than jumping straight to a big number.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from _common import DEFAULT_CONFIG

from src.config import load_config
from src.data import (
    completeness, fetch_symbol, get_client, load_raw, load_symbol, merge_bars, plan_fetch, regular_hours,
    resolve_end, resolve_universe, save_symbol, to_eastern,
)


def fetch_one(symbol: str, d, refresh: bool) -> str:
    """Do the network work for one symbol; returns the status string to print."""
    plan = (pd.Timestamp(d.start), resolve_end(d.end)) if refresh else plan_fetch(
        d.cache_dir, symbol, d.timeframe, d.start, d.end, d.feed, d.adjustment
    )
    if plan is None:
        return "cached"

    first, stop = plan
    raw = fetch_symbol(get_client(), symbol, str(first), str(stop), d.timeframe, d.feed, d.adjustment)
    cached = None if refresh else load_raw(d.cache_dir, symbol, d.timeframe)
    fresh = regular_hours(to_eastern(raw)) if not raw.empty else raw
    if fresh.empty and cached is None:
        return "no bars returned"

    before = 0 if cached is None else len(cached)
    merged = merge_bars(cached, fresh)
    save_symbol(merged, d.cache_dir, symbol, d.timeframe, d.start, d.end, d.feed, d.adjustment)
    return f"+{len(merged) - before:,} bars" if cached is not None else "downloaded"


def report(symbol: str, status: str, cache_dir, timeframe) -> None:
    if "no bars" in status:
        print(f"  {symbol:6s} {status}")
        return
    grid = load_symbol(cache_dir, symbol, timeframe)
    ts = grid.index.get_level_values("timestamp")
    print(f"  {symbol:6s} {status:14s} {ts[0].date()} -> {ts[-1].date()}  "
          f"{grid['session_id'].nunique():5,} sessions  {completeness(grid):6.1%} of 5-min slots have a bar")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    d = load_config(args.config).data
    universe = resolve_universe(d.universe)
    print(f"{len(universe)} symbols  {d.start} -> {d.end} (through {resolve_end(d.end) - pd.Timedelta(seconds=1):%Y-%m-%d})"
          f"  @{d.timeframe}  feed={d.feed}  workers={args.workers}")

    if args.workers <= 1:
        for symbol in universe:
            report(symbol, fetch_one(symbol, d, args.refresh), d.cache_dir, d.timeframe)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_one, s, d, args.refresh): s for s in universe}
            for i, future in enumerate(as_completed(futures), 1):
                symbol = futures[future]
                try:
                    report(symbol, future.result(), d.cache_dir, d.timeframe)
                except Exception as e:
                    print(f"  {symbol:6s} FAILED: {e}")
                if i % 50 == 0:
                    print(f"  ... {i}/{len(universe)} done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
