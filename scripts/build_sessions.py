#!/usr/bin/env python3
"""The 5-minute cache's session table, all the volatility panel needs from it (read-only; about 15 minutes in Colab):

    python scripts/build_sessions.py --sip data/bars_sip --out sessions.parquet --until 2024-01-02

Nothing after `until` is read, so the gate can run with the reserved period absent. It prints counts only: no return, volatility or
gap summary of any period. It replaces the full data audit (scripts/data_audit_v2.py) for this purpose, whose report covers every year.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from _common import ROOT, reachable

from src import audit, nyse
from src.data import cache_path, resolve_universe

COLUMNS = ["symbol", "date", "bars", "slots", "high", "low", "half_day", "nyse_session", "status"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sip", default="data/bars_sip")
    ap.add_argument("--universe", default=str(ROOT / "universe" / "sp500.csv"))
    ap.add_argument("--start", default="2016-01-04")
    ap.add_argument("--until", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not reachable(args.sip, args.out):
        return 1
    until = pd.Timestamp(args.until)
    calendar = nyse.sessions(args.start, until)
    t0, tables = time.time(), []
    for i, symbol in enumerate(resolve_universe(args.universe)):
        bars = audit.read_bars(cache_path(args.sip, symbol, "5Min"))
        if bars is None or bars.empty:
            continue
        bars = bars[bars.index < until + pd.Timedelta(days=1)]
        if len(bars):
            table = audit.session_table(bars, symbol, calendar)
            if len(table):
                tables.append(table[COLUMNS])
        if (i + 1) % 100 == 0:
            print(f"{i + 1} symbols ({time.time() - t0:.0f}s)")
    if not tables:
        print(f"no cached bars under {args.sip}")
        return 1
    out = pd.concat(tables, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"{len(out):,} symbol-sessions for {out['symbol'].nunique()} symbols through {out['date'].max().date()}; written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
