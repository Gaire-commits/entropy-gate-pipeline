#!/usr/bin/env python3
"""Download Alpaca bars into the local parquet cache, skipping what is already there.

    python scripts/fetch_data.py
    python scripts/fetch_data.py --config configs/spy_gao.yaml
    python scripts/fetch_data.py --refresh          # re-download everything
"""

from __future__ import annotations

import argparse

from _common import DEFAULT_CONFIG

from src.config import load_config
from src.data import (
    completeness, fetch_symbol, get_client, is_cached, load_symbol, regular_hours, save_symbol, to_eastern,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    d = load_config(args.config).data
    print(f"{len(d.universe)} symbols  {d.start} -> {d.end}  @{d.timeframe}  feed={d.feed}")
    client = None
    for symbol in d.universe:
        if not args.refresh and is_cached(d.cache_dir, symbol, d.timeframe, d.start, d.end, d.feed, d.adjustment):
            status = "cached"
        else:
            client = client or get_client()
            raw = fetch_symbol(client, symbol, d.start, d.end, d.timeframe, d.feed, d.adjustment)
            if raw.empty:
                print(f"  {symbol:5s} no bars returned")
                continue
            save_symbol(regular_hours(to_eastern(raw)), d.cache_dir, symbol, d.timeframe,
                        d.start, d.end, d.feed, d.adjustment)
            status = "downloaded"

        grid = load_symbol(d.cache_dir, symbol, d.timeframe)
        ts = grid.index.get_level_values("timestamp")
        print(f"  {symbol:5s} {status:10s} {ts[0].date()} -> {ts[-1].date()}  "
              f"{grid['session_id'].nunique():5,} sessions  {completeness(grid):6.1%} of 5-min slots have a bar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
