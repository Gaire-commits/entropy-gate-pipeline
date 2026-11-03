#!/usr/bin/env python3
"""Which rule for the listing exchange fixes the 2021 shortfall of official opens? Read-only: writes nothing and scores nothing.

    python scripts/official_listing_candidates.py --reference data/reference

STEP 1 failed its coverage check for 2021 (91.08% of usable bars). A diagnostic showed the prints exist (99.99% on some exchange): src/official.py picks one
exchange per stock per year, Nasdaq's prints change code from T to Q during April to June 2021, flipping between them, and stocks can change listing mid-year.

The rule's pick became src/official.py (protocols/amendment_2.md), so on a rerun "as run" and "per month, T and Q as one" match. This stays as the record of the choice.

THE RULE, committed before this ran. Each candidate takes, for a stock and period, the exchange with the most closing prints equal to the vendor's close, each day counted once. Among those whose
prices agree with the current rule's on at least 99.9% of the bars where both have an official open before the lock, it picks the highest lowest-year coverage of usable bars, 2016-2023. Candidates within 0.1
point count as tied, and a tie goes to the earliest in CANDIDATES, the simplest first. If none agrees that well, nothing is picked. For the reserved period this prints coverage counts only; agreement,
gaps and prices stop at the lock.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT

from src import official, portfolio as pf

CANDIDATES = {"per year, T and Q as one": ("year", True), "per month": ("month", False), "per month, T and Q as one": ("month", True)}
LOCK = pd.Timestamp("2024-01-03")
MIN_AGREEMENT = 0.999
TIE = 0.001


def candidate_opens(m: pd.DataFrame, period: str, merged: bool) -> pd.DataFrame:
    """The official open under one candidate, from official.with_vendor's frame."""
    x = m.assign(venue=m["exchange"].replace({"T": "Q"}) if merged else m["exchange"],
                 period=m["date"].dt.year.astype(str) if period == "year" else m["date"].dt.strftime("%Y-%m"))
    c = x[(x["side"] == "close") & x["condition"].isin(official.CLOSE_CODES) & (x["gap_bp"].abs() <= official.TOL_BP)].drop_duplicates(["symbol", "date", "venue"])
    best = c.groupby(["symbol", "period", "venue"]).size().rename("n").reset_index() \
        .sort_values(["symbol", "period", "n", "venue"], ascending=[True, True, False, True]).drop_duplicates(["symbol", "period"])
    o = x[(x["side"] == "open") & x["condition"].isin(official.OPEN_CODES)].merge(best[["symbol", "period", "venue"]], on=["symbol", "period", "venue"])
    o = o.assign(rank=o["condition"].map({c: i for i, c in enumerate(official.OPEN_CODES)})).sort_values(["rank", "t"]).drop_duplicates(["symbol", "date"])
    return o[["symbol", "date", "price", "condition"]].rename(columns={"price": "official_open", "condition": "open_print"})


def agreement(cand: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Before the lock, where both rules have an official open: how often they agree within tolerance."""
    x = cand[cand["date"] < LOCK].merge(current[current["date"] < LOCK].dropna(subset=["official_open"])[["symbol", "date", "official_open"]], on=["symbol", "date"], suffixes=("", "_now"))
    gap = 1e4 * np.log(x["official_open"] / x["official_open_now"])
    return {"both": int(len(x)), "agree": float((gap.abs() <= official.TOL_BP).mean()) if len(x) else float("nan"), "differ_over_25bp": int((gap.abs() > 25).sum())}


def choose(table: pd.DataFrame) -> str | None:
    ok = table[table["agree"] >= MIN_AGREEMENT]
    if ok.empty:
        return None
    top = ok["lowest_year"].max()
    return next(n for n in CANDIDATES if n in ok.index and ok.loc[n, "lowest_year"] >= top - TIE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/reference")
    args = ap.parse_args()
    ref = Path(args.reference)
    pd.set_option("display.width", 200)
    prints = pd.read_parquet(ref / "auctions.parquet")
    raw = pd.read_parquet(ref / "daily" / "daily_sip_raw.parquet")
    current = pd.read_parquet(ref / "official_prices.parquet")
    current["date"] = pd.to_datetime(current["date"]).dt.normalize()
    prints["date"] = pd.to_datetime(prints["date"]).dt.tz_localize(None)
    joined = pf.load_joined(ROOT / "registry" / "sp500_join_dates.csv")
    m = official.with_vendor(prints, raw)

    cover = {"as run": official.eligible_coverage(current, raw, joined)["with_official_open"]}
    rows, opens = {}, {}
    for name, (period, merged) in CANDIDATES.items():
        opens[name] = candidate_opens(m, period, merged)
        cover[name] = official.eligible_coverage(opens[name], raw, joined)["with_official_open"]
        rows[name] = {**agreement(opens[name], current), "lowest_year": float(cover[name].loc[:2023].min())}
    print("A. Official-open coverage of usable bars, by year (counts only):")
    print(pd.DataFrame(cover).round(4).to_string())
    table = pd.DataFrame(rows).T
    print("\nB. Against the current rule before the lock, where both have an official open:")
    print(table.round(5).to_string())
    pick = choose(table)
    print(f"\nTHE RULE PICKS: {pick}")

    print("\nC. The opening codes, before the lock, on the same stock, day and exchange:")
    o = m[(m["side"] == "open") & (m["date"] < LOCK) & m["condition"].isin(official.OPEN_CODES)]
    q = o[o["condition"] == "Q"].drop_duplicates(["symbol", "date", "exchange"])
    t = o[o["condition"] == "O"].drop_duplicates(["symbol", "date", "exchange"])
    both = q.merge(t, on=["symbol", "date", "exchange"], suffixes=("_q", "_o"))
    print(f"   first Q print {q['date'].min().date()}; bars with both {len(both):,}; equal within 0.5 bp {(1e4 * np.log(both['price_q'] / both['price_o'])).abs().le(official.TOL_BP).mean():.4f}")
    print("   Q prints by year:", q.groupby(q["date"].dt.year).size().to_dict())

    if pick:
        c = opens[pick]
        v = current[current["date"] < LOCK][["symbol", "date"]].merge(c, on=["symbol", "date"], how="left")
        miss = v[v["official_open"].isna()]
        print(f"\nD. Under the pick, before the lock: {len(miss):,} vendor bars still have no official open; most by symbol and year:",
              miss.groupby(["symbol", miss["date"].dt.year]).size().sort_values(ascending=False).head(10).to_dict())

    print("\nE. 2023-01-24, when the NYSE skipped most opening auctions. Stocks whose current listing is N; distance from the previous close, in bp:")
    r = raw.assign(date=pd.to_datetime(raw["date"]).dt.normalize()).sort_values(["symbol", "date"])
    r = r[(r["volume"] > 0) & (r["trade_count"] > 0) & (r["date"] < LOCK)]
    r["prev"] = r.groupby("symbol")["close"].shift(1)
    d = pd.Timestamp("2023-01-24")
    x = r[r["date"] == d][["symbol", "open", "high", "low", "prev"]].merge(current[(current["date"] == d) & (current["listing"] == "N")][["symbol", "official_open", "open_print"]], on="symbol")
    far = lambda s: (1e4 * np.log(s / x["prev"])).abs()
    print(f"   stocks {len(x)}; print used {x['open_print'].value_counts(dropna=False).to_dict()}; vendor open: median {far(x['open']).median():.0f}, largest {far(x['open']).max():.0f}; "
          f"official open: median {far(x['official_open']).median():.0f}, largest {far(x['official_open']).max():.0f}; official outside the day's range {int(((x['official_open'] > x['high']) | (x['official_open'] < x['low'])).sum())}")
    print("\nThis scored nothing and wrote nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
