#!/usr/bin/env python3
"""Build a static universe file from a public constituent list.

    python scripts/build_universe.py --index sp500

Writes universe/<index>.csv (symbol, name, sector, fetched_at) and is meant to be
re-run occasionally, not on every pipeline run: a config points at the saved file,
so every experiment run against it uses the exact same universe until you rebuild it.

Survivorship bias: this is today's constituent list. Screening 2020-2026 history
against it silently omits every name that was in the index at some point in that
window and later got removed (acquired, bankrupt, demoted) -- exactly the names a
market-wide predictability claim most needs to include. Fine for a first look at
whether the gate behaves differently at this scale; a real point-in-time
constituent history (available from WRDS/CRSP) is needed before this can be a
survivorship-bias-free result.
"""

from __future__ import annotations

import argparse
import ssl
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SOURCES = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
}


def fetch_html(url: str) -> str:
    """Plain urllib with certifi's CA bundle -- the stock ssl module has no trust
    store on some Python.org macOS installs, which makes urlopen fail outright."""
    import certifi

    context = ssl.create_default_context(cafile=certifi.where())
    # Wikipedia's servers 403 requests with no descriptive User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": "entropy-gate-pipeline/1.0 (research use)"})
    with urllib.request.urlopen(request, context=context, timeout=20) as resp:
        return resp.read().decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", choices=sorted(SOURCES), default="sp500")
    parser.add_argument("--out", default="universe")
    args = parser.parse_args()

    import io

    print(f"fetching {args.index} constituents from {SOURCES[args.index]}")
    tables = pd.read_html(io.StringIO(fetch_html(SOURCES[args.index])))
    raw = tables[0]

    df = pd.DataFrame(
        {
            "symbol": raw["Symbol"].str.strip(),
            "name": raw["Security"],
            "sector": raw["GICS Sector"],
        }
    ).drop_duplicates(subset="symbol")
    df["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    dotted = df[df["symbol"].str.contains(r"\.")]
    if len(dotted):
        print(f"note: {len(dotted)} dual-class tickers keep their dot (e.g. BRK.B) -- "
              f"drop or rewrite them first if your data vendor wants a different format")

    out = Path(args.out) / f"{args.index}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} symbols -> {out}")
    print("\nsurvivorship bias: this is today's list. See this file's module docstring before "
          "treating a historical backtest against it as free of survivorship bias.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
