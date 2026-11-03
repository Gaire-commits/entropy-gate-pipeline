"""The official open and close from the listing exchange's auction prints."""

import numpy as np
import pandas as pd

from src import official


def frames():
    """AAA lists on T in 2020 and on Q in 2021 (Nasdaq's code change); BBB on N. Other venues print too."""
    days = list(pd.bdate_range("2020-12-01", "2021-01-29"))
    raw, prints = [], []
    for sym, early, late in (("AAA", "T", "Q"), ("BBB", "N", "N")):
        for i, d in enumerate(days):
            lst = early if d.year == 2020 else late
            o, c = 100.0 + i, 101.0 + i
            vendor_open = o + (0.42 if i == 3 else 0.0)                       # day 3: a trade before the cross
            raw.append({"symbol": sym, "date": d, "open": vendor_open, "high": c + 1, "low": o - 1, "close": c,
                        "volume": 1000, "trade_count": 50})
            t = pd.Timestamp(d).tz_localize("America/New_York")
            if i != 5:                                                         # day 5: no opening print at all
                prints += [(sym, d, "open", "O", o, 100, t + pd.Timedelta(hours=9, minutes=30, seconds=2), lst),
                           (sym, d, "open", "O", o + 0.3, 1, t + pd.Timedelta(hours=9, minutes=30, seconds=1), "P")]
                if i % 2 == 0:
                    prints.append((sym, d, "open", "Q", o, 0, t + pd.Timedelta(hours=9, minutes=30, seconds=3), lst))
            prints += [(sym, d, "close", "6", c, 100, t + pd.Timedelta(hours=16), lst),
                       (sym, d, "close", "M", c, 0, t + pd.Timedelta(hours=16, seconds=1), lst),
                       (sym, d, "close", "M", c - 0.2, 0, t + pd.Timedelta(hours=16, seconds=1), "P")]
    a = pd.DataFrame(prints, columns=["symbol", "date", "side", "condition", "price", "size", "t", "exchange"])
    a["t"] = a["t"].dt.tz_convert("UTC")
    return a, pd.DataFrame(raw)


def test_the_listing_exchange_is_per_month_and_nasdaqs_two_codes_are_one_exchange():
    a, raw = frames()
    lst = official.listing_exchange(official.with_vendor(a, raw))
    got = {(r.symbol, r.month): r.listing for r in lst.itertuples()}
    assert got == {("AAA", "2020-12"): "Q", ("AAA", "2021-01"): "Q", ("BBB", "2020-12"): "N", ("BBB", "2021-01"): "N"}


def test_a_code_change_or_a_listing_move_inside_a_year_loses_no_prints():
    days = pd.bdate_range("2021-01-04", "2021-12-31")
    raw = pd.DataFrame([{"symbol": s, "date": d, "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1000, "trade_count": 50} for s in ("NAS", "HON") for d in days])
    rows = []
    for s, d in raw[["symbol", "date"]].itertuples(index=False):
        flip = pd.Timestamp("2021-04-01") <= d < pd.Timestamp("2021-05-01") and d.day % 2 == 0     # flipping between the codes
        ex = ("T" if d < pd.Timestamp("2021-06-01") and not flip else "Q") if s == "NAS" else ("N" if d < pd.Timestamp("2021-05-17") else "Q")
        rows += [(s, d, "open", "O", 100.0, d, ex), (s, d, "close", "6", 101.0, d, ex)]
    a = pd.DataFrame(rows, columns=["symbol", "date", "side", "condition", "price", "t", "exchange"])
    op = official.official_prices(a, raw)
    nas, hon = op[op["symbol"] == "NAS"], op[op["symbol"] == "HON"]
    assert nas["official_open"].notna().all() and nas["official_close"].notna().all()
    assert hon["official_open"].notna().sum() >= len(hon) - 10              # at most the days of the move's own month are lost


def test_the_code_is_the_rule_that_was_picked():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import official_listing_candidates as script
    a, raw = frames()
    picked = script.candidate_opens(official.with_vendor(a, raw), "month", True).sort_values(["symbol", "date"]).reset_index(drop=True)
    op = official.official_prices(a, raw).dropna(subset=["official_open"]).reset_index(drop=True)
    assert (op["symbol"] == picked["symbol"]).all() and (op["date"] == picked["date"]).all()
    assert np.allclose(op["official_open"], picked["official_open"]) and (op["open_print"] == picked["open_print"]).all()


def test_official_open_prefers_q_then_o_and_leaves_days_without_a_print_empty():
    a, raw = frames()
    op = official.official_prices(a, raw)
    assert len(op) == len(raw)
    days = sorted(op["date"].unique())
    aaa = op[op["symbol"] == "AAA"].set_index("date")
    assert aaa.loc[days[0], "open_print"] == "Q" and aaa.loc[days[1], "open_print"] == "O"
    assert np.isnan(aaa.loc[days[5], "official_open"])                       # no print: no official open, nothing invented
    assert aaa.loc[days[3], "open_gap_bp"] > 4                              # the vendor's open is the earlier trade
    others = aaa.drop(index=[days[3], days[5]])
    assert (others["open_gap_bp"].abs() < 1e-9).all()                       # the other venue's print is never used
    assert (op["close_print"] == "M").all() and (op["close_gap_bp"].abs() < 1e-9).all()
    c = official.coverage(op)
    assert c.loc[2020, "with_official_open"] < 1 == c.loc[2021, "with_official_open"]
    assert list(c.columns) == ["bars", "with_listing", "with_official_open", "with_official_close"]   # counts only
    g = official.gap_summary(op, "2021-01-01")
    assert list(g.index) == [2020] and g["close_equal"].eq(1).all()              # nothing from on or after the cutoff is even computed


def test_coverage_among_usable_bars_and_missing_prints_on_extreme_days():
    a, raw = frames()
    op = official.official_prices(a, raw)
    e = official.eligible_coverage(op, raw, min_dollar_volume=1.0)
    assert e["usable_bars"].sum() > 0 and e["with_official_open"].between(0, 1).all()
    assert official.eligible_coverage(op, raw, min_dollar_volume=1e15).empty
    adj = raw.copy()
    m = official.missing_on_extreme_days(op, raw, adj, "2022-01-01")
    assert m["sessions"] > 0 and 0 < m["missing_overall"] < 1
