#!/usr/bin/env python3
"""Build a static universe file from a public constituent list.

    python scripts/build_universe.py --index sp500
    python scripts/build_universe.py --index nasdaq100
    python scripts/build_universe.py --holdout nasdaq100     # needs both lists; see below

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

--holdout splits the S&P 500 list against another list for configs/ndx_holdout.yaml: the S&P 500 without
the other index's members (training universe) and the members of both (held-out universe, with the
S&P 500's GICS sectors so both universes use one classification). The other index's members outside the
S&P 500 are left out: there is no comparable history for them in the S&P 500 data.
"""

from __future__ import annotations

import argparse
import ssl
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import _common  # noqa: F401  (puts the repo root on the path for src.holdout)

SOURCES = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "nasdaq100": "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
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


def _column(table: pd.DataFrame, *names: str) -> str:
    """The first column whose name (footnote markers like "[1]" ignored) matches one of `names`."""
    clean = {c: str(c).split("[")[0].strip().lower() for c in table.columns}
    for want in names:
        for c, label in clean.items():
            if label == want:
                return c
    raise KeyError(f"none of {names} in columns {list(table.columns)}")


def parse_sp500(tables: list[pd.DataFrame]) -> pd.DataFrame:
    raw = tables[0]
    return pd.DataFrame({"symbol": raw["Symbol"].astype(str).str.strip(), "name": raw["Security"],
                         "sector": raw["GICS Sector"]})


def parse_nasdaq100(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """The constituents table is the one with a Ticker column. Its sector is the ICB industry, a different
    classification from the S&P 500's GICS sectors."""
    raw = next(t for t in tables if any(str(c).split("[")[0].strip().lower() == "ticker" for c in t.columns))
    return pd.DataFrame({"symbol": raw[_column(raw, "ticker")].astype(str).str.strip(),
                         "name": raw[_column(raw, "company")],
                         "sector": raw[_column(raw, "icb industry", "gics sector", "industry", "sector")]})


PARSERS = {"sp500": parse_sp500, "nasdaq100": parse_nasdaq100}


def write_holdout(other: str, out_dir: Path) -> int:
    from src.holdout import holdout_split

    base_path, other_path = out_dir / "sp500.csv", out_dir / f"{other}.csv"
    for p in (base_path, other_path):
        if not p.exists():
            print(f"missing {p}: run --index {p.stem} first")
            return 1
    base, members = pd.read_csv(base_path), pd.read_csv(other_path)
    train, held = holdout_split(base, members)
    left_out = sorted(set(members["symbol"]) - set(base["symbol"]))
    train.to_csv(out_dir / f"sp500_ex_{other}.csv", index=False)
    held.to_csv(out_dir / f"{other}_in_sp500.csv", index=False)
    print(f"training universe: {len(train)} S&P 500 stocks outside {other} -> {out_dir / f'sp500_ex_{other}.csv'}")
    print(f"held-out universe: {len(held)} of {len(members)} {other} members that are in the S&P 500 -> "
          f"{out_dir / f'{other}_in_sp500.csv'}")
    print(f"left out (not in the S&P 500 list, so no comparable history): {len(left_out)} {left_out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", choices=sorted(SOURCES), default="sp500")
    parser.add_argument("--holdout", choices=sorted(set(SOURCES) - {"sp500"}), default=None,
                        help="split the saved S&P 500 list against this saved list (no download)")
    parser.add_argument("--out", default="universe")
    args = parser.parse_args()
    if args.holdout:
        return write_holdout(args.holdout, Path(args.out))

    import io

    print(f"fetching {args.index} constituents from {SOURCES[args.index]}")
    tables = pd.read_html(io.StringIO(fetch_html(SOURCES[args.index])))
    df = PARSERS[args.index](tables).drop_duplicates(subset="symbol")
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
