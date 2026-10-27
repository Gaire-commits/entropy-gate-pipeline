"""The entropy-effect comparison: paired on the same rows and days."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from entropy_effect import paired


def frame(prob, ret, dates):
    return pd.DataFrame({"date": dates, "prob": prob, "ret": ret, "y": (ret > 0).astype(int)})


def test_identical_models_differ_by_exactly_zero_and_a_better_one_is_found():
    rng = np.random.default_rng(0)
    n, d = 20_000, 200
    dates = np.repeat(pd.date_range("2024-01-01", periods=d), n // d)
    ret = rng.normal(0, 0.003, n)
    signal = (ret > 0).astype(float) * 0.2 + 0.4            # a model that leans the right way
    days = np.unique(dates)
    same = paired(frame(signal, ret, dates), frame(signal, ret, dates), days, n_boot=200)
    assert same["gross_bps"]["diff"] == 0 and same["gross_bps"]["lo"] == 0 == same["gross_bps"]["hi"]
    coin = rng.choice([0.4, 0.6], n)
    better = paired(frame(coin, ret, dates), frame(signal, ret, dates), days, n_boot=200)
    assert better["gross_bps"]["lo"] > 0 and better["accuracy"]["diff"] > 0
