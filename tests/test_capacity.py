"""E2: cells must differ only in what they claim to, and a planted edge must be planted, sized and found."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capacity import (
    calibrate_edge, capacity_verdict, cell_config, mean_rank_ic, paired_ic_difference, plant, planted_signal,
    planted_verdict, trim_train_days, zscore_moments,
)
from src.crosssection import cross_sectional_ic
from src.dataset import build_dataset, walk_forward_splits
from src.experiment import load_predictions, run_arch
from src.holdout import apply_target, restrict_folds
from src.ml import predict_gbm, train_gbm
from src.synthetic import ar_returns, bars_from_returns

CHANNELS = ["log_return", "session_return", "hl_range", "volume_z"]


def _cfg(leaves=None):
    model = SimpleNamespace(epochs=5, patience=3, batch_size=256, lr=1e-3, weight_decay=1e-4, dropout=0.2)
    if leaves is not None:
        model.gbm_leaves = leaves
    return SimpleNamespace(
        features=SimpleNamespace(channels=CHANNELS, window=12, horizon=30, embargo=1, stride=12, allow_overnight=False,
                                 normalize="fold_standardize", label_deadzone=0.0),
        model=model)


def _data(n=40, sessions=220, seed=0):
    bars = {f"S{i:02d}": bars_from_returns(f"S{i:02d}", ar_returns(sessions, 0.02, seed=seed * 100 + i), seed=i,
                                           start="2020-01-02") for i in range(n)}
    return build_dataset(bars, None, _cfg())


def _folds(data, train, trade, keep):
    split = SimpleNamespace(train_days=100, val_days=20, test_days=40, step_days=40)
    return restrict_folds(walk_forward_splits(data["date"], split), data["symbol"], train, trade, keep)


def test_trimming_keeps_the_most_recent_training_days_and_leaves_val_and_test_alone():
    data = _data(n=6, sessions=120)
    folds = walk_forward_splits(data["date"], SimpleNamespace(train_days=60, val_days=20, test_days=20, step_days=20))
    short = trim_train_days(folds, data["date"], 15)
    for f, s in zip(folds, short):
        kept = np.unique(data["date"][s["train"]])
        assert len(kept) == 15 and kept.max() == np.unique(data["date"][f["train"]]).max()
        assert np.array_equal(f["val"], s["val"]) and np.array_equal(f["test"], s["test"])


def test_tree_size_defaults_reproduce_every_earlier_run_and_a_setting_changes_it():
    rng = np.random.default_rng(0)
    F, Fv = rng.normal(size=(4000, 5)), rng.normal(size=(1000, 5))
    y, yv = (F[:, 0] + rng.normal(size=4000) > 0).astype(int), (Fv[:, 0] + rng.normal(size=1000) > 0).astype(int)
    m0, i0 = train_gbm(F, y, Fv, yv)
    m1, i1 = train_gbm(F, y, Fv, yv, max_leaf_nodes=15, max_iter=300)
    assert np.array_equal(predict_gbm(m0, Fv, i0), predict_gbm(m1, Fv, i1))
    m4, i4 = train_gbm(F, y, Fv, yv, max_leaf_nodes=4)
    assert i4["n_leaves"] <= 4 * (i4["best_epoch"] + 1)

    data = _data(n=12, sessions=200)
    everyone = set(data["symbol"])
    folds = _folds(data, everyone, everyone, np.ones(len(data["y"]), bool))[:1]
    assert folds
    probs = []
    for cfg in (_cfg(), _cfg(leaves=15), _cfg(leaves=4)):
        with tempfile.TemporaryDirectory() as tmp:
            run_arch(data, folds, "gbm", 0, cfg, Path(tmp), log=lambda _: None)
            probs.append(load_predictions(Path(tmp))["prob"].to_numpy())
    assert np.array_equal(probs[0], probs[1]) and not np.array_equal(probs[0], probs[2])
    base = _cfg()
    assert cell_config(base, 63).model.gbm_leaves == 63 and not hasattr(base.model, "gbm_leaves")


def _ic_frame(score, ret, moments=200, n=50):
    d = np.repeat(pd.bdate_range("2024-01-01", periods=moments // 2), 2 * n)
    b = np.tile(np.repeat([11, 23], n), moments // 2)
    return pd.DataFrame({"date": d, "bar_index": b, "symbol": np.tile(np.arange(n), moments), "prob": score, "ret": ret})


def test_paired_difference_is_zero_for_identical_scores_and_positive_for_a_better_one():
    rng = np.random.default_rng(1)
    ret = rng.normal(size=10_000)
    weak = _ic_frame(0.02 * ret + rng.normal(size=10_000), ret)
    strong = _ic_frame(0.3 * ret + rng.normal(size=10_000), ret)
    same = paired_ic_difference(weak, weak, n_boot=300)
    assert same["diff"] == 0 and same["lo"] == 0 and same["hi"] == 0
    better = paired_ic_difference(strong, weak, n_boot=300)
    assert better["lo"] > 0.1


def test_capacity_verdict_follows_the_rule():
    s = {"d504_l15": {"t": 0.5}, "d126_l15": {"t": 0.2}, "d504_l63": {"t": 3.5}}
    flat = {"d126_l15": {"lo": -0.01}, "d504_l63": {"lo": -0.002}}
    assert capacity_verdict({k: {"t": 0.1} for k in s}, flat, "d504_l15")["level"] == "signal-limited"
    assert capacity_verdict(s, flat, "d504_l15")["level"] == "mixed"
    up = {"d126_l15": {"lo": -0.01}, "d504_l63": {"lo": 0.001}}
    v = capacity_verdict(s, up, "d504_l15")
    assert v["level"] == "capacity or data helps" and v["cells"] == ["d504_l63"]


def test_planted_signal_uses_only_the_inputs_and_is_standardized_per_moment():
    data = _data(n=30, sessions=60)
    for shape in ("momentum", "interaction"):
        s = planted_signal(data, shape, CHANNELS)
        moved = dict(data, ret=data["ret"] + 0.05)
        assert np.array_equal(s, planted_signal(moved, shape, CHANNELS))
        means = pd.Series(s).groupby([data["date"], data["bar_index"]]).mean()
        assert np.abs(means).max() < 1e-9 and np.isfinite(s).all()
    inter = planted_signal(data, "interaction", CHANNELS)
    gated = data["X"][:, CHANNELS.index("volume_z"), -1] > 0.5
    assert 0.1 < gated.mean() < 0.9                                     # the gate really splits the rows
    assert np.corrcoef(inter, planted_signal(data, "momentum", CHANNELS))[0, 1] < 0.9
    with pytest.raises(ValueError, match="unknown shape"):
        planted_signal(data, "astrology", CHANNELS)


def test_calibration_hits_the_target_ic_and_planting_moves_only_returns():
    data = _data(n=60, sessions=80)
    s = planted_signal(data, "momentum", CHANNELS)
    assert calibrate_edge(data, s, 0.0) == 0.0
    for target in (0.03, 0.1):
        lam = calibrate_edge(data, s, target, min_peers=20)
        planted = plant(data, s, lam)
        got = mean_rank_ic(s, planted["ret"], data["date"], data["bar_index"], data["symbol"], 20)
        assert got == pytest.approx(target, rel=0.1)
        assert planted["X"] is data["X"] and not np.array_equal(planted["ret"], data["ret"])


def test_planted_verdict_needs_t_and_half_of_the_oracle_at_break_even():
    rows = [{"shape": "momentum", "target": 0.0075, "t": 2.0, "ic": 0.004, "oracle_ic": 0.0075},
            {"shape": "momentum", "target": 0.014, "t": 4.0, "ic": 0.009, "oracle_ic": 0.014},
            {"shape": "interaction", "target": 0.014, "t": 4.0, "ic": 0.005, "oracle_ic": 0.014},
            {"shape": "interaction", "target": 0.02, "t": 6.0, "ic": 0.015, "oracle_ic": 0.02}]
    v = planted_verdict(rows, 0.014)
    assert v["momentum"] == {"adequate": True, "smallest_recovered": 0.014}
    assert v["interaction"] == {"adequate": False, "smallest_recovered": 0.02}


def test_the_pipeline_finds_a_planted_edge_and_not_one_that_is_absent():
    """End to end on synthetic bars: plant an edge, apply the relative target, run the walk-forward trees."""
    data = _data(n=40, sessions=220, seed=3)
    everyone = set(data["symbol"])
    s = planted_signal(data, "momentum", CHANNELS)
    results = {}
    for target in (0.0, 0.2):
        d = plant(data, s, calibrate_edge(data, s, target, min_peers=20))
        keep = apply_target(d, "relative", everyone, everyone, min_peers=20)
        folds = _folds(d, everyone, everyone, keep)
        with tempfile.TemporaryDirectory() as tmp:
            run_arch(d, folds, "gbm", 0, _cfg(), Path(tmp), log=lambda _: None)
            results[target] = cross_sectional_ic(load_predictions(Path(tmp)), min_peers=20, n_boot=300)
    assert results[0.2]["t"] > 3 and results[0.2]["ic"] > 0.1
    assert abs(results[0.0]["t"]) < 3
