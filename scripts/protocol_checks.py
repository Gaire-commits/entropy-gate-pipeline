#!/usr/bin/env python3
"""The checks that need Alpaca, run before the reserved window is scored. They use no forecast and score nothing.

    python scripts/protocol_checks.py --reference data/reference

1. Alpaca's SPY and BIL against the independently recorded total returns in the protocol, 2017-2023.
2. The official open: how many vendor bars have one, by year (needs official_prices.parquet).
3. Quoted half-spreads just before the close on sampled development and validation days, against the frozen cost.
Prints PASS or FAIL for each. A failure means amend the protocol, with the reason, before anything is scored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT

from src import nyse, official, portfolio as pf
from src.alpaca_rest import AlpacaREST, rfc3339

MIN_COVERAGE = 0.95


def spreads(client, symbols, days, seconds: int = 30) -> pd.DataFrame:
    """Median quoted half-spread, in bp of the mid, per stock-day over the `seconds` ending one minute before the close."""
    rows = []
    for day in days:
        close = pd.Timestamp(day) + nyse.close_time(pd.Timestamp(day))
        lo, hi = close - pd.Timedelta(seconds=60 + seconds), close - pd.Timedelta(seconds=60)
        q = client.quotes(list(symbols), rfc3339(lo), rfc3339(hi))
        q = q[(q["bid"] > 0) & (q["ask"] > q["bid"])]
        q = q.assign(half=1e4 * (q["ask"] - q["bid"]) / (q["ask"] + q["bid"]))
        rows.append(q.groupby("symbol")["half"].median().rename("half_spread_bp").reset_index().assign(date=pd.Timestamp(day)))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/reference")
    args = ap.parse_args()
    ref = Path(args.reference)
    cfg, doc = pf.load_protocol(ROOT / "protocols" / "portfolio_v1.yaml")
    pd.set_option("display.width", 200)
    results = {}
    client = AlpacaREST()

    lock = pf.lock_date(cfg)
    bench = pf.fetch_benchmarks(ref / f"benchmarks_to_{lock.date()}.parquet", "2016-01-04", str(lock.date()), client=client)
    guard = pf.check_benchmarks(bench, doc["benchmark_reference"])
    print("1. Alpaca's benchmarks against the recorded total returns:")
    print(guard.round(4).to_string(index=False))
    r = pf.benchmark_returns(bench, "BIL")
    print(f"   BIL daily: largest move {r.abs().max():.2%}; SPY sessions {bench[bench['symbol'] == 'SPY']['date'].nunique()}")
    results["benchmarks"] = bool(guard["ok"].all())

    print("\n2. The official open:")
    path = ref / "official_prices.parquet"
    if path.exists():
        op = pd.read_parquet(path)
        table = official.summary(op)
        print(table.round(4).to_string())
        results["official open"] = bool((table["with_official_open"] >= MIN_COVERAGE).all())
    else:
        print("   official_prices.parquet is missing: run the auctions fetch and scripts/official_prices.py first.")
        results["official open"] = False

    print("\n3. Quoted half-spreads before the close, development and validation only:")
    rng = np.random.default_rng(0)
    universe = pd.read_csv(ROOT / "universe" / "sp500.csv")["symbol"].tolist()
    symbols = sorted(rng.choice(universe, 30, replace=False).tolist())
    cal = nyse.sessions("2019-01-04", "2023-12-29")
    days = sorted(rng.choice(cal[~cal.isin(nyse.HALF_DAYS)], 12, replace=False))
    try:
        s = spreads(client, symbols, days)
        by_year = s.groupby(s["date"].dt.year)["half_spread_bp"].describe()[["count", "50%", "75%", "max"]]
        print(by_year.round(2).to_string())
        median = float(s["half_spread_bp"].median())
        print(f"   median {median:.2f} bp against the frozen {cfg.cost_bps:.1f} bp one-way")
        results["costs"] = median <= cfg.cost_bps
    except Exception as exc:  # noqa: BLE001  (not every plan serves quotes; the cost stays the frozen one, with this gap stated)
        print(f"   not available: {type(exc).__name__}: {str(exc)[:200]}")
        results["costs"] = None

    print()
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'NOT CHECKED' if ok is None else 'FAIL'}: {name}")
    failed = [n for n, ok in results.items() if ok is False]
    print("\n" + ("PRE-CHECKS PASSED: the reserved window may be scored when you decide." if not failed else f"STOP: {failed}. Amend the protocol, with the reason, before anything is scored."))
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
