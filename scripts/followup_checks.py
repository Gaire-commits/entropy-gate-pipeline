#!/usr/bin/env python3
"""Three small checks left open by amendment 2, before the reserved period is scored. Read-only: writes nothing, scores nothing, and reads nothing on or after 2024-01-03.

    python scripts/followup_checks.py --reference data/reference

1. The bars whose official open moved by more than 25 bp when the listing exchange became per month: which price looks right.
2. 2023-01-24, when the NYSE skipped most opening auctions: were the day's high and low set by trades later cancelled, which would inflate the range in V?
3. Quoted half-spreads on a larger sample than the protocol check's (which was 12 days and 30 stocks), by liquidity, against the 5 bp assumed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT

from src import nyse, official
from src.alpaca_rest import AlpacaREST, rfc3339

LOCK = pd.Timestamp("2024-01-03")
GLITCH = pd.Timestamp("2023-01-24")


def old_rule_opens(prints: pd.DataFrame, raw: pd.DataFrame, min_prints: int = 50) -> pd.DataFrame:
    """The official open as version 2 chose it, before amendment 2: one exchange per stock and year, the one whose closing prints most often equal the vendor's close."""
    m = official.with_vendor(prints, raw)
    m["year"] = m["date"].dt.year
    c = m[(m["side"] == "close") & m["condition"].isin(official.CLOSE_CODES)]
    rate = c.groupby(["symbol", "year", "exchange"])["gap_bp"].agg(prints="size", close_equal=lambda x: float((x.abs() <= official.TOL_BP).mean())).reset_index()
    best = rate[rate["prints"] >= min_prints].sort_values(["symbol", "year", "close_equal"], ascending=[True, True, False]).drop_duplicates(["symbol", "year"])
    m = m.merge(best[["symbol", "year", "exchange"]].rename(columns={"exchange": "old_listing"}), on=["symbol", "year"])
    o = official._pick(m[m["exchange"] == m["old_listing"]], "open", official.OPEN_CODES)
    return o.merge(m[["symbol", "date", "old_listing"]].drop_duplicates(["symbol", "date"]), on=["symbol", "date"]).rename(columns={"price": "old_open", "condition": "old_print"})


def changed_bars(new: pd.DataFrame, old: pd.DataFrame, raw: pd.DataFrame, bp: float = 25.0) -> pd.DataFrame:
    """Bars before the lock where both rules have an official open and they differ by more than `bp`, with the day's vendor bar to judge them by."""
    r = raw.assign(date=pd.to_datetime(raw["date"]).dt.normalize())[["symbol", "date", "open", "high", "low"]]
    x = new[new["date"] < LOCK].dropna(subset=["official_open"]).merge(old, on=["symbol", "date"]).merge(r, on=["symbol", "date"])
    x["gap_bp"] = 1e4 * np.log(x["official_open"] / x["old_open"])
    x = x[x["gap_bp"].abs() > bp].copy()
    inside = lambda p: (p >= x["low"] - 1e-9) & (p <= x["high"] + 1e-9)
    x["new_in_range"], x["old_in_range"] = inside(x["official_open"]), inside(x["old_open"])
    x["closer_to_vendor_open"] = np.where((x["official_open"] - x["open"]).abs() <= (x["old_open"] - x["open"]).abs(), "new", "old")
    return x.sort_values("gap_bp", key=abs, ascending=False)


def range_from_cancelled(trades: pd.DataFrame, bars: pd.DataFrame, tick: float = 0.005) -> pd.DataFrame:
    """Per stock: was the day's high or low printed by a trade later cancelled or corrected (a non-empty update), and by nothing else at that price?"""
    t = trades.assign(bad=trades["update"].fillna("").astype(str).str.len() > 0)
    out = []
    for sym, b in bars.set_index("symbol").iterrows():
        x = t[t["symbol"] == sym]
        good, bad = x[~x["bad"]]["price"], x[x["bad"]]["price"]
        high_bad = bool(len(bad) and bad.max() >= b["high"] - tick and not (len(good) and good.max() >= b["high"] - tick))
        low_bad = bool(len(bad) and bad.min() <= b["low"] + tick and not (len(good) and good.min() <= b["low"] + tick))
        out.append({"symbol": sym, "cancelled_trades": int(len(bad)), "high_from_cancelled": high_bad, "low_from_cancelled": low_bad})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/reference")
    args = ap.parse_args()
    ref = Path(args.reference)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 80)
    raw = pd.read_parquet(ref / "daily" / "daily_sip_raw.parquet")
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw = raw[(raw["date"] < LOCK) & (raw["volume"] > 0) & (raw["trade_count"] > 0)]
    prints = pd.read_parquet(ref / "auctions.parquet")
    prints["date"] = pd.to_datetime(prints["date"]).dt.tz_localize(None).dt.normalize()
    prints = prints[prints["date"] < LOCK]
    new = pd.read_parquet(ref / "official_prices.parquet")
    new["date"] = pd.to_datetime(new["date"]).dt.normalize()
    new = new[new["date"] < LOCK]

    print("1. Official opens that moved by over 25 bp under amendment 2, before the lock:")
    x = changed_bars(new, old_rule_opens(prints, raw), raw)
    print(f"   {len(x)} bars; new price inside the day's range {int(x['new_in_range'].sum())}, old {int(x['old_in_range'].sum())}; "
          f"closer to the vendor's open: new {int((x['closer_to_vendor_open'] == 'new').sum())}, old {int((x['closer_to_vendor_open'] == 'old').sum())}")
    print("   by year:", x.groupby(x["date"].dt.year).size().to_dict(), "| old exchange to new:", x.groupby(["old_listing", "listing"]).size().to_dict())
    print(x[["symbol", "date", "old_listing", "listing", "old_open", "official_open", "open", "low", "high", "gap_bp", "new_in_range", "old_in_range"]]
          .head(60).round(2).to_string(index=False))

    print(f"\n2. {GLITCH.date()}: were the day's high and low set by trades later cancelled? NYSE-listed stocks, trades 09:30 to 10:00:")
    on = new[(new["date"] == GLITCH) & (new["listing"] == "N")]
    bars = raw[raw["date"] == GLITCH].merge(on[["symbol", "official_open"]], on="symbol")
    client = AlpacaREST()
    lo, hi = pd.Timestamp(GLITCH) + nyse.OPEN, pd.Timestamp(GLITCH) + nyse.OPEN + pd.Timedelta(minutes=30)
    lo, hi = lo.tz_localize("America/New_York"), hi.tz_localize("America/New_York")
    symbols = sorted(bars["symbol"])
    trades = pd.concat([client.trades(symbols[i:i + 50], rfc3339(lo), rfc3339(hi)) for i in range(0, len(symbols), 50)], ignore_index=True)
    flags = range_from_cancelled(trades, bars[["symbol", "high", "low"]]).merge(bars[["symbol", "official_open"]], on="symbol")
    with_v = flags[flags["official_open"].notna()]
    print(f"   stocks {len(flags)}; trades fetched {len(trades):,}, of which flagged cancelled or corrected {int((trades['update'].fillna('').astype(str).str.len() > 0).sum()):,}; updates seen {trades['update'].dropna().astype(str).value_counts().head(5).to_dict()}")
    print(f"   high set only by a cancelled trade: {int(flags['high_from_cancelled'].sum())}; low: {int(flags['low_from_cancelled'].sum())}. "
          f"Among the {len(with_v)} with an official open, which get a V that day: high {int(with_v['high_from_cancelled'].sum())}, low {int(with_v['low_from_cancelled'].sum())}")
    prev = raw[(raw["date"] < GLITCH) & (raw["date"] >= GLITCH - pd.Timedelta(days=45))]
    rho = lambda f: np.log(f["high"] / f["low"]) ** 2 / (4 * np.log(2))
    ratio = (bars.set_index("symbol").pipe(rho) / prev.assign(r=rho(prev)).groupby("symbol")["r"].median()).dropna()
    print(f"   the range that day over each stock's median of the 30 sessions before: median {ratio.median():.2f}, 90th percentile {ratio.quantile(0.9):.1f}, largest {ratio.max():.1f}")

    print("\n3. Quoted half-spreads one minute before the close, 2019-2023, a larger sample (seed 1; the protocol check used seed 0):")
    import protocol_checks
    rng = np.random.default_rng(1)
    liq = raw[raw["date"] >= "2019-01-01"].assign(dv=lambda f: f["close"] * f["volume"]).groupby("symbol")["dv"].median()
    third = pd.qcut(liq, 3, labels=["least liquid third", "middle third", "most liquid third"])
    symbols = sorted(rng.choice(liq.index.to_numpy(), 90, replace=False).tolist())
    cal = nyse.sessions("2019-01-04", "2023-12-29")
    days = sorted(rng.choice(cal[~cal.isin(nyse.HALF_DAYS)], 30, replace=False))
    s = protocol_checks.spreads(client, symbols, days)
    s["third"] = s["symbol"].map(third)
    table = s.groupby("third", observed=True)["half_spread_bp"].describe(percentiles=[0.5, 0.75, 0.9])[["count", "50%", "75%", "90%", "max"]]
    print(table.round(2).to_string())
    print(f"   all: median {s['half_spread_bp'].median():.2f} bp, mean {s['half_spread_bp'].mean():.2f}; stock-days above 5 bp {(s['half_spread_bp'] > 5).mean():.1%}")
    print("\nThis scored nothing and wrote nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
