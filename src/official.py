"""The official open and close: each symbol's listing exchange's auction prints (read-only).

The vendor's daily close is the official close, but its open is the first regular trade on any venue, which on a late
opening auction can be a stale price far from where the auction cleared (RESULTS.md, data audit part 2). The official
open is the listing exchange's official open print (condition Q), or its opening trade (O) where there is no Q; the
official close its official close (M), or its closing trade (6). The listing exchange of each symbol and year is the
one whose closing prints equal the vendor's close most often; Nasdaq listings print under T until 2020 and under Q
from 2021, and a symbol can move (LIN, N to Q in 2024).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.volpanel import trailing_dollar_volume

OPEN_CODES = ("Q", "O")
CLOSE_CODES = ("M", "6")
TOL_BP = 0.5


def _real(raw: pd.DataFrame) -> pd.DataFrame:
    r = raw.copy()
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()
    return r[(r["volume"] > 0) & (r["trade_count"] > 0)]


def with_vendor(prints: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Every print with the vendor's raw open (close) of its day and the gap in basis points, vendor over print. Both are
    unadjusted, so equality is exact."""
    a = prints.copy()
    a["date"] = pd.to_datetime(a["date"]).dt.normalize()
    v = _real(raw)[["symbol", "date", "open", "close"]].rename(columns={"open": "vendor_open", "close": "vendor_close"})
    m = a.merge(v, on=["symbol", "date"])
    m["gap_bp"] = 1e4 * np.log(np.where(m["side"] == "open", m["vendor_open"], m["vendor_close"]) / m["price"])
    m["year"] = m["date"].dt.year
    return m


def listing_exchange(m: pd.DataFrame, min_prints: int = 50, tol_bp: float = TOL_BP) -> pd.DataFrame:
    """Per symbol and year: the exchange whose closing prints equal the vendor's close most often, and how often."""
    c = m[(m["side"] == "close") & m["condition"].isin(CLOSE_CODES)]
    rate = c.groupby(["symbol", "year", "exchange"])["gap_bp"].agg(
        prints="size", close_equal=lambda x: float((x.abs() <= tol_bp).mean())).reset_index()
    rate = rate[rate["prints"] >= min_prints]
    best = rate.sort_values(["symbol", "year", "close_equal"], ascending=[True, True, False]).drop_duplicates(["symbol", "year"])
    return best.rename(columns={"exchange": "listing"}).reset_index(drop=True)


def _pick(m: pd.DataFrame, side: str, codes: tuple) -> pd.DataFrame:
    """The day's first print of the first code present, under the listing exchange."""
    x = m[(m["side"] == side) & m["condition"].isin(codes)].copy()
    x["rank"] = x["condition"].map({c: i for i, c in enumerate(codes)})
    x = x.sort_values(["rank", "t"]).drop_duplicates(["symbol", "date"])
    return x[["symbol", "date", "price", "condition"]]


def official_prices(prints: pd.DataFrame, raw: pd.DataFrame, min_prints: int = 50) -> pd.DataFrame:
    """One row per vendor bar with a trade: the vendor's raw open and close, the official ones where the listing exchange
    printed them (empty where it did not), which print each is, and the gaps in basis points (vendor over official)."""
    m = with_vendor(prints, raw)
    listing = listing_exchange(m, min_prints)
    m = m.merge(listing[["symbol", "year", "listing"]], on=["symbol", "year"])
    m = m[m["exchange"] == m["listing"]]
    o = _pick(m, "open", OPEN_CODES).rename(columns={"price": "official_open", "condition": "open_print"})
    c = _pick(m, "close", CLOSE_CODES).rename(columns={"price": "official_close", "condition": "close_print"})
    v = _real(raw)[["symbol", "date", "open", "close", "high", "low"]].rename(columns={"open": "vendor_open", "close": "vendor_close"})
    v = v[v["symbol"].isin(prints["symbol"].unique())]
    v["year"] = v["date"].dt.year
    out = v.merge(listing[["symbol", "year", "listing"]], on=["symbol", "year"], how="left") \
        .merge(o, on=["symbol", "date"], how="left").merge(c, on=["symbol", "date"], how="left")
    out["open_gap_bp"] = 1e4 * np.log(out["vendor_open"] / out["official_open"])
    out["close_gap_bp"] = 1e4 * np.log(out["vendor_close"] / out["official_close"])
    out["open_outside_range"] = (out["official_open"] > out["high"]) | (out["official_open"] < out["low"])
    return out.drop(columns=["year", "high", "low"]).sort_values(["symbol", "date"]).reset_index(drop=True)


def coverage(op: pd.DataFrame) -> pd.DataFrame:
    """By year: how many vendor bars have an official open and close. Counts only, so it is safe to print for any period."""
    g = op.groupby(op["date"].dt.year)
    return pd.DataFrame({"bars": g.size(), "with_listing": g["listing"].apply(lambda s: s.notna().mean()),
                         "with_official_open": g["official_open"].apply(lambda s: s.notna().mean()),
                         "with_official_close": g["official_close"].apply(lambda s: s.notna().mean())})


def eligible_coverage(op: pd.DataFrame, raw: pd.DataFrame, joined: pd.Series | None = None, min_close: float = 5.0, min_dollar_volume: float = 10e6) -> pd.DataFrame:
    """The same among the bars the protocols can use: at least $5, liquid by trailing_dollar_volume(), in the index by then."""
    r = _real(raw).sort_values(["symbol", "date"])
    r["dv63"] = (r["close"] * r["volume"]).groupby(r["symbol"]).transform(trailing_dollar_volume)
    ok = r[(r["close"] >= min_close) & (r["dv63"] >= min_dollar_volume)]
    if joined is not None:
        ok = ok[~(ok["date"] < ok["symbol"].map(joined))]
    x = ok[["symbol", "date"]].merge(op[["symbol", "date", "official_open"]], on=["symbol", "date"], how="left")
    g = x.groupby(x["date"].dt.year)
    return pd.DataFrame({"usable_bars": g.size(), "with_official_open": g["official_open"].apply(lambda s: s.notna().mean())})


def gap_summary(op: pd.DataFrame, before, tol_bp: float = TOL_BP) -> pd.DataFrame:
    """By year: how the vendor's open and close compare with the official ones, on dates before `before` only. Gap sizes describe
    the data, so the reserved period's are not computed here, let alone printed."""
    x = op[op["date"] < pd.Timestamp(before)]
    x = x.assign(year=x["date"].dt.year)
    g = x.groupby("year")
    a_open, a_close = x["open_gap_bp"].abs(), x["close_gap_bp"].abs()
    return pd.DataFrame({
        "open_equal": (a_open <= tol_bp).astype(float).where(x["official_open"].notna()).groupby(x["year"]).mean(),
        "open_over_25bp": (a_open > 25).astype(float).where(x["official_open"].notna()).groupby(x["year"]).mean(),
        "open_p99_bp": a_open.groupby(x["year"]).quantile(0.99),
        "close_equal": (a_close <= tol_bp).astype(float).where(x["official_close"].notna()).groupby(x["year"]).mean(),
        "open_outside_range": x["open_outside_range"].where(x["official_open"].notna()).astype(float).groupby(x["year"]).mean(),
        "bars": g.size()})


def missing_on_extreme_days(op: pd.DataFrame, raw: pd.DataFrame, adj: pd.DataFrame, before, top: float = 0.01) -> dict:
    """Are sessions with no official open concentrated among the largest moves? On dates before `before` only: the share missing
    overall, and among the top `top` of sessions by V on the vendor's open."""
    keys = ["symbol", "date"]
    a = adj.copy()
    a["date"] = pd.to_datetime(a["date"]).dt.normalize()
    a = a[(a["volume"] > 0) & (a["trade_count"] > 0) & (a["date"] < pd.Timestamp(before))].sort_values(keys)
    r = _real(raw)[keys + ["high", "low"]]
    m = a[keys + ["open", "close"]].merge(r, on=keys).merge(op[keys + ["official_open"]], on=keys, how="left")
    m["V"] = np.log(m["open"] / m.groupby("symbol")["close"].shift(1)) ** 2 + np.log(m["high"] / m["low"]) ** 2 / (4 * np.log(2))
    m = m.dropna(subset=["V"])
    cut = m["V"].quantile(1 - top)
    return {"sessions": int(len(m)), "missing_overall": float(m["official_open"].isna().mean()), "missing_on_top": float(m.loc[m["V"] >= cut, "official_open"].isna().mean())}
