#!/usr/bin/env python3
"""The official open and close for every symbol-day, from the auction prints (read-only, about a minute): see src/official.py.

    python scripts/official_prices.py --reference data/reference

Reads data/reference/auctions.parquet (fetch_reference.py --parts auctions) and the vendor's raw daily bars; writes
data/reference/official_prices.parquet and prints a short summary to paste into the chat. Nothing is downloaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import reachable

from src import official

TRUNCATED = ["2021-04-19", "2021-10-25", "2022-03-08"]   # the vendor's own truncated days (RESULTS.md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/reference")
    args = ap.parse_args()
    ref = Path(args.reference)
    if not reachable(ref):
        return 1
    need = [ref / "auctions.parquet", ref / "daily" / "daily_sip_raw.parquet"]
    if [p for p in need if not p.exists()]:
        print("STOPPED: missing", ", ".join(str(p) for p in need if not p.exists()),
              "- run section 35, then the auctions fetch in section 39.")
        return 1
    prints = pd.read_parquet(need[0])
    op = official.official_prices(prints, pd.read_parquet(need[1]))
    op.to_parquet(ref / "official_prices.parquet", index=False)

    pd.set_option("display.width", 200)
    print(f"{len(prints):,} auction prints for {prints['symbol'].nunique()} symbols; {len(op):,} vendor bars with a trade, "
          f"{op['listing'].notna().mean():.2%} with a listing exchange")
    years = op.assign(year=op["date"].dt.year).drop_duplicates(["symbol", "year"])
    print("\nlisting exchanges (symbol-years):", years["listing"].value_counts(dropna=False).to_dict())
    print("\nby year (shares of vendor bars; 'equal' within 0.5 bp, among bars with an official price):")
    print(official.summary(op).round(4).to_string())
    print("\nofficial open taken from:", op["open_print"].value_counts(dropna=False).to_dict(),
          "| official close taken from:", op["close_print"].value_counts(dropna=False).to_dict())
    t = op[op["date"].isin(pd.to_datetime(TRUNCATED))]
    print("\nthe vendor's truncated days: is its close the official close?")
    print(t.groupby(t["date"].dt.date).agg(bars=("symbol", "size"), with_official_close=("official_close", "count"),
                                            close_equal=("close_gap_bp", lambda s: (s.abs() <= 0.5).mean())).to_string())
    miss = op[op["official_open"].isna()]
    print(f"\nvendor bars with no official open: {len(miss):,}; symbols with the most:",
          miss["symbol"].value_counts().head(8).to_dict())
    big = op.reindex(op["open_gap_bp"].abs().sort_values(ascending=False).index).head(10)
    print("\nlargest open gaps (vendor over official, bp):")
    print(big[["symbol", "date", "listing", "vendor_open", "official_open", "open_print", "open_gap_bp"]]
          .round(2).to_string(index=False))
    print(f"\nwritten {ref / 'official_prices.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
