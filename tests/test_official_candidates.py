"""Choosing the listing exchange: a mid-year code change must not lose prints, and the rule must follow what it says."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import official_listing_candidates as script


def frame():
    rows = []
    for d in pd.bdate_range("2021-01-04", "2021-12-31"):
        ex = "T" if d < pd.Timestamp("2021-04-15") else "Q"
        for side, cond in (("open", "O"), ("close", "6")):
            rows.append(("AAA", d, side, cond, 100.0, d, ex, 0.0))
        rows.append(("AAA", d, "close", "6", 101.0, d, "P", 99.0))
    return pd.DataFrame(rows, columns=["symbol", "date", "side", "condition", "price", "t", "exchange", "gap_bp"])


def test_treating_t_and_q_as_one_exchange_loses_no_day_where_the_others_do():
    m = frame()
    days = m["date"].nunique()
    assert len(script.candidate_opens(m, "year", True)) == days
    assert len(script.candidate_opens(m, "month", True)) == days
    assert len(script.candidate_opens(m, "month", False)) < days        # April holds both codes


def test_the_rule_needs_agreement_then_coverage_with_ties_to_the_simplest():
    t = pd.DataFrame({"agree": [0.9995, 0.9999, 0.9999], "lowest_year": [0.9900, 0.9905, 0.9950]}, index=list(script.CANDIDATES))
    assert script.choose(t) == "per month, T and Q as one"
    assert script.choose(t.assign(lowest_year=[0.9950, 0.9905, 0.9955])) == "per year, T and Q as one"
    assert script.choose(t.assign(agree=[0.99, 0.99, 0.9995])) == "per month, T and Q as one"
    assert script.choose(t.assign(agree=0.9)) is None


def test_agreement_stops_at_the_lock():
    cand = pd.DataFrame({"symbol": "A", "date": pd.to_datetime(["2023-12-29", "2024-06-03"]), "official_open": [10.0, 10.0]})
    now = cand.assign(official_open=[10.0, 50.0])
    a = script.agreement(cand, now)
    assert a["both"] == 1 and a["agree"] == 1.0 and a["differ_over_25bp"] == 0
