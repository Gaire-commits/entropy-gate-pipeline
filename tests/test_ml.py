"""Gradient-boosted trees: learn what is there, stop early on noise, and use entropy only through gbm_ent."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["EGP_DEVICE"] = "cpu"

pytest.importorskip("sklearn")

from src.dataset import build_dataset, walk_forward_splits
from src.entropy_features import ENTROPY_COLUMNS
from src.experiment import load_predictions, run_arch
from src.ml import predict_gbm, train_gbm, tree_features, window_summary
from src.synthetic import ar_returns, bars_from_returns


def test_window_summary_is_last_mean_spread_min_max_per_channel():
    X = np.array([[[1.0, 2.0, 3.0], [0.0, 0.0, 6.0]]])
    np.testing.assert_allclose(window_summary(X)[0], [3, 6, 2, 2, np.std([1, 2, 3]), np.std([0, 0, 6]), 1, 0, 3, 6])


def _data(n=4000, seed=0, rule="window"):
    """Synthetic samples. 'window': the label follows the window's last value. 'entropy': the
    label follows the window only when an entropy feature is high and is reversed when it is
    low, so a model without that feature sees a coin flip."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2, 12)).astype(np.float32)
    data = {"X": X, "bar_index": rng.integers(0, 78, n), "y": None}
    for c in ENTROPY_COLUMNS:
        data[c] = rng.normal(size=n)
    signal = X[:, 0, -1] + 0.7 * rng.normal(size=n)
    if rule == "entropy":
        signal = np.where(data["ent_trend3_hour"] > 0, signal, -signal)
    elif rule == "noise":
        signal = rng.normal(size=n)
    data["y"] = (signal > 0).astype(np.int64)
    return data


def _fit(data, arch):
    tr, va, te = np.arange(0, 2400), np.arange(2400, 3000), np.arange(3000, len(data["y"]))
    model, info = train_gbm(tree_features(data, tr, arch), data["y"][tr], tree_features(data, va, arch), data["y"][va])
    prob = predict_gbm(model, tree_features(data, te, arch), info)
    return ((prob > 0.5) == (data["y"][te] == 1)).mean(), info


def test_gbm_learns_a_window_signal():
    acc, info = _fit(_data(rule="window"), "gbm")
    assert acc > 0.75 and info["best_epoch"] > 10


def test_gbm_stops_early_on_noise():
    acc, info = _fit(_data(rule="noise"), "gbm")
    assert abs(acc - 0.5) < 0.06 and info["best_epoch"] < 60


def test_only_gbm_ent_can_use_a_signal_that_depends_on_entropy():
    data = _data(rule="entropy")
    plain, _ = _fit(data, "gbm")
    with_entropy, _ = _fit(data, "gbm_ent")
    assert abs(plain - 0.5) < 0.06
    assert with_entropy > 0.7


def test_tree_models_run_through_the_sweep_and_save_the_entropy_features(tmp_path):
    cfg = SimpleNamespace(
        features=SimpleNamespace(channels=["log_return", "session_return"], window=12, horizon=30, embargo=1,
                                 stride=12, allow_overnight=False, normalize="fold_standardize", label_deadzone=0.0),
        model=SimpleNamespace(epochs=30))
    data = build_dataset({"T": bars_from_returns("T", ar_returns(120, 0.2, seed=7))}, None, cfg)
    folds = walk_forward_splits(data["date"], SimpleNamespace(train_days=60, val_days=15, test_days=20, step_days=20))[:2]
    for arch in ("gbm", "gbm_ent"):
        assert run_arch(data, folds, arch, 0, cfg, tmp_path, log=lambda _: None) == len(folds)
    pred = load_predictions(tmp_path)
    assert set(pred["arch"]) == {"gbm", "gbm_ent"}
    assert set(ENTROPY_COLUMNS) <= set(pred.columns)
    assert pred["prob"].between(0, 1).all() and (pred["n_params"] > 0).all()
