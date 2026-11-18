"""The signal search: ICs are right on a known case, q-values follow Benjamini-Hochberg, and a planted signal is found."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]

import explore_signals as script
from src import volforecast as vf
from test_volforecast import CFG, DAYS, panel


def test_daily_ic_is_a_rank_correlation_and_respects_the_mask():
    d = pd.date_range("2020-01-01", periods=3)
    x = pd.DataFrame(np.arange(180).reshape(3, 60) % 60, index=d, columns=[f"S{i}" for i in range(60)], dtype=float)
    mask = pd.DataFrame(True, index=d, columns=x.columns)
    assert np.allclose(script.daily_ic(x, x ** 3, mask), 1.0) and np.allclose(script.daily_ic(x, -x, mask), -1.0)
    assert script.daily_ic(x, x, mask.iloc[:, :10].reindex(columns=x.columns, fill_value=False)).empty      # too few names


def test_benjamini_hochberg():
    q = script.bh(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.5])


def test_a_planted_next_day_signal_is_found_and_counted():
    p = panel(n=60, seed=4)
    rng = np.random.default_rng(0)
    p = p.sort_values(["symbol", "date"])
    e = pd.Series(0.01 * rng.normal(size=len(p)), index=p.index)
    p["ret"] = e - 0.5 * e.groupby(p["symbol"]).shift(1).fillna(0.0)        # no drift, and a strong one-day reversal
    rows = vf.build_rows(p, CFG)
    table = script.analyse(p, rows, CFG, DAYS[200], DAYS[-1], split=DAYS[450])
    hit = table[(table["feature"] == "reversal_1d") & (table["target"] == "ret_1d")].iloc[0]
    assert hit["ic_x100"] < 0 and hit["t"] < -5 and hit["q"] < 0.01 and hit["same_sign_both_halves"]
    assert len(table) == 9 * 4 and table["q"].between(0, 1).all()
