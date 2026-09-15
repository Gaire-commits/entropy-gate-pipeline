"""Causality guarantees. These are the tests that matter most: a lookahead bug
does not crash, it just produces a backtest that cannot be reproduced live."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import gaf_encode, make_windows, mtf_encode, normalize_windows

BARS_PER_SESSION = 78


def _fixture(n_sessions=6, n_channels=3):
    n = BARS_PER_SESSION * n_sessions
    rng = np.random.default_rng(0)
    features = rng.normal(size=(n, n_channels))
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    session_id = np.repeat(np.arange(n_sessions), BARS_PER_SESSION)
    return features, close, session_id


def test_window_contents_never_include_a_bar_past_the_signal():
    features, close, sessions = _fixture()
    features[:] = np.arange(features.shape[0])[:, None]

    out = make_windows(features, close, window=20, horizon=5, embargo=1, session_id=sessions)
    for i, signal_bar in enumerate(out["signal_bar"]):
        assert out["X"][i].max() <= signal_bar


def test_label_is_read_strictly_after_the_window():
    features, close, sessions = _fixture()
    window, horizon, embargo = 20, 5, 1

    out = make_windows(features, close, window, horizon, embargo, session_id=sessions)
    for i, signal_bar in enumerate(out["signal_bar"]):
        entry, exit_ = signal_bar + embargo, signal_bar + embargo + horizon
        assert entry > signal_bar
        assert out["ret"][i] == pytest.approx(np.log(close[exit_] / close[entry]))


def test_intraday_samples_never_straddle_a_session():
    features, close, sessions = _fixture()
    window, horizon, embargo = 20, 5, 1

    out = make_windows(features, close, window, horizon, embargo, session_id=sessions)
    for signal_bar in out["signal_bar"]:
        start = signal_bar - window + 1
        exit_ = signal_bar + embargo + horizon
        assert sessions[start] == sessions[exit_]


def test_impossible_geometry_is_rejected_rather_than_silently_going_overnight():
    """window+embargo+horizon > session length has no intraday solution."""
    features, close, sessions = _fixture()
    with pytest.raises(ValueError, match="exceeds the longest session"):
        make_windows(features, close, window=64, horizon=12, embargo=12, session_id=sessions)


def test_overnight_mode_must_be_opted_into():
    features, close, sessions = _fixture()
    out = make_windows(
        features, close, window=64, horizon=12, embargo=12, session_id=sessions, allow_overnight=True
    )
    assert out["X"].shape[0] > 0


def test_deadzone_drops_samples_below_the_threshold():
    features, close, sessions = _fixture()
    kwargs = dict(window=20, horizon=5, embargo=1, session_id=sessions)

    loose = make_windows(features, close, deadzone=0.0, **kwargs)
    strict = make_windows(features, close, deadzone=0.002, **kwargs)
    assert strict["X"].shape[0] < loose["X"].shape[0]
    assert (np.abs(strict["ret"]) >= 0.002).all()


def test_normalization_uses_only_in_window_statistics():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 24, 3))
    normalized = normalize_windows(X, "window_zscore")

    np.testing.assert_allclose(normalized.mean(axis=1), 0, atol=1e-10)
    np.testing.assert_allclose(normalized.std(axis=1), 1, atol=1e-10)

    # Shifting one window must not move any other window.
    shifted = X.copy()
    shifted[0] += 50.0
    assert np.allclose(normalize_windows(shifted, "window_zscore")[1:], normalized[1:])


def test_gaf_output_is_bounded_and_square():
    x = np.random.default_rng(2).normal(size=(5, 16))
    img = gaf_encode(x)
    assert img.shape == (5, 16, 16)
    assert np.abs(img).max() <= 1.0 + 1e-9


def test_mtf_rows_are_transition_probabilities():
    x = np.random.default_rng(3).normal(size=(4, 32))
    img = mtf_encode(x, n_bins=4)
    assert img.shape == (4, 32, 32)
    assert (img >= 0).all() and (img <= 1).all()


def test_empty_result_keeps_downstream_shapes_valid():
    features, close, sessions = _fixture(n_sessions=1)
    out = make_windows(features[:30], close[:30], window=20, horizon=5, embargo=1, session_id=sessions[:30], deadzone=10.0)
    assert out["X"].shape == (0, 20, 3)
    assert out["y"].shape == (0,)
