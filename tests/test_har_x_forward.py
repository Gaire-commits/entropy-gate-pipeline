"""The forward test scores only target dates on or after its first, and gives a verdict only at the registered length."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]

import explore_har_x
import har_x_forward as script
from src import volforecast as vf
from test_volforecast import CFG, DAYS, panel


def test_the_protocol_points_at_the_frozen_model():
    doc, model = script.load()
    assert doc["first_target"] == "2026-11-19" and doc["decision_at_target_dates"] == 500 and len(model["extra"]) == 7


def test_only_dates_from_the_first_target_are_scored():
    p = panel(n=40, seed=7)
    rows = explore_har_x.add_features(vf.build_rows(p, CFG), p)
    _, model = script.load()
    model = {**model, "extra": [c for c in model["extra"] if c in rows]}
    g = script.daily_gaps(rows, model, CFG, DAYS[500], DAYS[-1])
    assert len(g) and g.index.min() >= DAYS[500]


def test_interim_until_the_registered_length_then_one_verdict():
    doc, _ = script.load()
    days = pd.bdate_range("2026-11-19", periods=600)
    better = {"a": pd.DataFrame({"qlike": -0.01 + 0.001 * np.random.default_rng(0).normal(size=600)}, index=days)}
    early = script.judge({"a": better["a"].iloc[:200]}, doc, CFG)
    assert early["stage"].startswith("INTERIM") and "verdict" not in early
    final = script.judge(better, doc, CFG)
    assert final["stage"] == "VERDICT" and final["verdict"] == "CONFIRMED" and final["target_dates"] == 600
    worse = script.judge({"a": -better["a"]}, doc, CFG)
    assert worse["verdict"] == "NOT CONFIRMED"
