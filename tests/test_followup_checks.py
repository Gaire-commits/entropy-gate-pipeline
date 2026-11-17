"""The follow-up checks: the old listing rule as it was, and the test for a range set by cancelled trades."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]

import followup_checks as script
from test_official import frames


def test_the_old_rule_is_one_exchange_per_year():
    a, raw = frames()
    old = script.old_rule_opens(a, raw, min_prints=10)
    assert set(map(tuple, old[["symbol", "old_listing"]].assign(y=old["date"].dt.year)[["symbol", "y", "old_listing"]].drop_duplicates().values)) == \
        {("AAA", 2020, "T"), ("AAA", 2021, "Q"), ("BBB", 2020, "N"), ("BBB", 2021, "N")}


def test_changed_bars_keeps_only_large_moves_before_the_lock():
    new = pd.DataFrame({"symbol": "A", "date": pd.to_datetime(["2023-01-24", "2023-01-25", "2024-02-01"]), "official_open": [100.0, 100.0, 100.0], "listing": "N"})
    old = new[["symbol", "date"]].assign(old_open=[110.0, 100.1, 120.0], old_print="O", old_listing="N")
    raw = new[["symbol", "date"]].assign(open=[109.0, 100.0, 100.0], high=[105.0, 101.0, 121.0], low=[95.0, 99.0, 99.0])
    x = script.changed_bars(new, old, raw)
    assert list(x["date"]) == [pd.Timestamp("2023-01-24")] and bool(x["new_in_range"].iloc[0]) and not bool(x["old_in_range"].iloc[0])


def test_a_range_counts_as_cancelled_only_when_nothing_else_reached_it():
    trades = pd.DataFrame({"symbol": ["A", "A", "A", "B", "B"], "price": [120.0, 100.0, 90.0, 120.0, 120.0], "update": ["canceled", None, "", "canceled", None]})
    bars = pd.DataFrame({"symbol": ["A", "B"], "high": [120.0, 120.0], "low": [90.0, 100.0]})
    f = script.range_from_cancelled(trades, bars).set_index("symbol")
    assert f.loc["A", "high_from_cancelled"] and not f.loc["A", "low_from_cancelled"]
    assert not f.loc["B", "high_from_cancelled"]
