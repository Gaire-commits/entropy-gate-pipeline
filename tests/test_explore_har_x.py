"""The HAR-X comparison runs on a synthetic panel, clips its features, and counts its models."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]

import explore_har_x as script
from src import volforecast as vf
from test_volforecast import CFG, DAYS, panel


def test_the_comparison_runs_and_counts_its_models(capsys):
    p = panel(n=40, seed=5)
    rows = script.add_features(vf.build_rows(p, CFG), p)
    assert {"range_ratio", "ret_1d", "neg_ret_1d", "mom_21d"} <= set(rows.columns)
    assert (rows["ret_1d"].abs() <= script.CLIP["ret_1d"]).all() and (rows["neg_ret_1d"] <= 0).all()
    script.compare(rows, CFG, DAYS[300], DAYS[-1])
    out = capsys.readouterr().out
    assert "1. Out of sample" in out and "2. The full model by target year" in out and "MODELS TRIED: 3" in out
