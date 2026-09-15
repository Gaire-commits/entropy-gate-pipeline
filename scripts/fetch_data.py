#!/usr/bin/env python3
"""Download Alpaca bars into the local parquet cache.

    python scripts/fetch_data.py
    python scripts/fetch_data.py --symbols AAPL MSFT --timeframe 1Min
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import add_session_id, fetch_bars, regular_hours, save_symbol, to_eastern


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config).data
    symbols = args.symbols or cfg.universe
    timeframe = args.timeframe or cfg.timeframe

    print(f"fetching {len(symbols)} symbols  {args.start or cfg.start} -> {args.end or cfg.end}  @{timeframe}")
    df = fetch_bars(
        symbols,
        start=args.start or cfg.start,
        end=args.end or cfg.end,
        timeframe=timeframe,
        feed=cfg.feed,
        adjustment=cfg.adjustment,
    )
    if df.empty:
        print("no bars returned — check credentials, date range, and feed entitlement")
        return 1

    df = to_eastern(df)
    if cfg.regular_hours_only:
        before = len(df)
        df = regular_hours(df)
        print(f"regular-hours filter: {before:,} -> {len(df):,} bars")

    for symbol in df.index.get_level_values("symbol").unique():
        sym_df = add_session_id(df.xs(symbol, level="symbol", drop_level=False))
        path = save_symbol(sym_df, cfg.cache_dir, symbol, timeframe)
        sessions = sym_df["session_id"].nunique()
        print(f"  {symbol:6s} {len(sym_df):>7,} bars  {sessions:>4} sessions  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
