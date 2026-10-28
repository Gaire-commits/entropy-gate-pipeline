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


def test_the_listing_exchange_follows_the_vendor_close_and_its_code_change():
    a, raw = frames()
    lst = official.listing_exchange(official.with_vendor(a, raw), min_prints=10)
    got = {(r.symbol, r.year): r.listing for r in lst.itertuples()}
    assert got == {("AAA", 2020): "T", ("AAA", 2021): "Q", ("BBB", 2020): "N", ("BBB", 2021): "N"}


def test_official_open_prefers_q_then_o_and_leaves_days_without_a_print_empty():
    a, raw = frames()
    op = official.official_prices(a, raw, min_prints=10)
    assert len(op) == len(raw)
    days = sorted(op["date"].unique())
    aaa = op[op["symbol"] == "AAA"].set_index("date")
    assert aaa.loc[days[0], "open_print"] == "Q" and aaa.loc[days[1], "open_print"] == "O"
    assert np.isnan(aaa.loc[days[5], "official_open"])                       # no print: no official open, nothing invented
    assert aaa.loc[days[3], "open_gap_bp"] > 4                              # the vendor's open is the earlier trade
    others = aaa.drop(index=[days[3], days[5]])
    assert (others["open_gap_bp"].abs() < 1e-9).all()                       # the other venue's print is never used
    assert (op["close_print"] == "M").all() and (op["close_gap_bp"].abs() < 1e-9).all()
    s = official.summary(op)
    assert s["close_equal"].eq(1).all() and s.loc[2020, "with_official_open"] < 1 == s.loc[2021, "with_official_open"]
