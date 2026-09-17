"""Causality guarantees. These are the tests that matter most: a lookahead bug
does not crash, it just produces a backtest that cannot be reproduced live."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import (
    apply_standardizer, build_channels, fit_standardizer, gaf_encode, make_windows, mtf_encode,
    normalize_windows, session_return,
)
from src.synthetic import ar_returns, bars_from_returns

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
        assert sessions[signal_bar - window + 1] == sessions[signal_bar + embargo + horizon]


def test_impossible_geometry_is_rejected_rather_than_silently_going_overnight():
    features, close, sessions = _fixture()
    with pytest.raises(ValueError, match="exceeds the longest session"):
        make_windows(features, close, window=64, horizon=12, embargo=12, session_id=sessions)


def test_overnight_mode_must_be_opted_into():
    features, close, sessions = _fixture()
    out = make_windows(features, close, window=64, horizon=12, embargo=12, session_id=sessions, allow_overnight=True)
    assert out["X"].shape[0] > 0


def test_first_half_hour_geometry_gives_one_sample_per_day():
    """6 + 66 + 6 = 78: signal from 9:30-10:00, hold from the 15:30 close to the 16:00 close."""
    features, close, sessions = _fixture(n_sessions=5)
    out = make_windows(features, close, window=6, horizon=6, embargo=66, session_id=sessions)
    assert len(out["y"]) == 5
    offsets = out["signal_bar"] % BARS_PER_SESSION
    assert (offsets == 5).all()
    for i, signal_bar in enumerate(out["signal_bar"]):
        assert out["ret"][i] == pytest.approx(np.log(close[signal_bar + 72] / close[signal_bar + 66]))


def test_stride_makes_holds_back_to_back_within_a_day():
    features, close, sessions = _fixture()
    out = make_windows(features, close, window=48, horizon=12, embargo=1, session_id=sessions, stride=12)
    per_day = np.bincount(sessions[out["signal_bar"]])
    assert (per_day == 2).all()
    entries = out["signal_bar"][:2] + 1
    assert entries[1] - entries[0] == 12


def test_missing_bars_drop_the_affected_windows_instead_of_shifting_time():
    features, close, sessions = _fixture()
    features[100] = np.nan
    out = make_windows(features, close, window=20, horizon=5, embargo=1, session_id=sessions)
    for signal_bar in out["signal_bar"]:
        assert not (signal_bar - 19 <= 100 <= signal_bar)


def test_small_labels_are_not_dropped_here():
    """Dropping test samples by the size of a future move would be lookahead."""
    features, close, sessions = _fixture()
    close[:] = 100.0
    out = make_windows(features, close, window=20, horizon=5, embargo=1, session_id=sessions)
    assert len(out["y"]) > 0 and np.allclose(out["ret"], 0.0)


def test_window_return_is_the_cumulative_return_over_the_window():
    features, close, sessions = _fixture()
    out = make_windows(features, close, window=20, horizon=5, embargo=1, session_id=sessions)
    for i, signal_bar in enumerate(out["signal_bar"]):
        start = signal_bar - 19
        expected = np.log(close[signal_bar] / close[start - 1]) if start >= 1 else np.nan
        assert out["window_return"][i] == pytest.approx(expected, nan_ok=True)


def test_session_return_measures_from_the_previous_close_and_ignores_the_future():
    df = bars_from_returns("X", ar_returns(3, 0.0))
    values = session_return(df)
    close = df["close"].to_numpy()
    assert np.isnan(values[:78]).all()
    assert values[78] == pytest.approx(np.log(close[78] / close[77]))
    assert values[83] == pytest.approx(np.log(close[83] / close[77]))

    changed = df.copy()
    changed.iloc[100:, changed.columns.get_loc("close")] *= 1.5
    assert np.allclose(session_return(changed)[:100], values[:100], equal_nan=True)


def test_every_channel_is_causal():
    df = bars_from_returns("X", ar_returns(4, 0.0))
    channels = ["log_return", "session_return", "hl_range", "close_open", "volume_z", "signed_volume"]
    before = build_channels(df, channels)
    changed = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        changed.iloc[200:, changed.columns.get_loc(col)] *= 1.7
    after = build_channels(changed, channels)
    np.testing.assert_allclose(after[:200], before[:200], equal_nan=True)


def test_window_zscore_erases_the_trend_and_window_scale_keeps_it():
    X = np.ones((5, 24, 1)) * 0.001
    assert np.allclose(normalize_windows(X, "window_zscore").sum(axis=1), 0)
    trending = np.linspace(0, 0.002, 24)[None, :, None] + np.random.default_rng(0).normal(0, 1e-4, (5, 24, 1))
    assert (normalize_windows(trending, "window_scale").sum(axis=1) > 0).all()


def test_fold_standardizer_uses_training_statistics_only():
    rng = np.random.default_rng(1)
    train = rng.normal(5.0, 2.0, (200, 3, 10))
    test = rng.normal(50.0, 2.0, (20, 3, 10))
    stats = fit_standardizer(train)
    np.testing.assert_allclose(apply_standardizer(train, stats).mean(axis=(0, 2)), 0, atol=1e-5)
    assert (apply_standardizer(test, stats).mean(axis=(0, 2)) > 15).all()


def test_gaf_output_is_bounded_and_square():
    img = gaf_encode(np.random.default_rng(2).normal(size=(5, 16)))
    assert img.shape == (5, 16, 16)
    assert np.abs(img).max() <= 1.0 + 1e-9


def test_mtf_rows_are_transition_probabilities():
    img = mtf_encode(np.random.default_rng(3).normal(size=(4, 32)), n_bins=4)
    assert img.shape == (4, 32, 32)
    assert (img >= 0).all() and (img <= 1).all()


def test_empty_result_keeps_downstream_shapes_valid():
    features, close, sessions = _fixture(n_sessions=1)
    features[:] = np.nan
    out = make_windows(features, close, window=20, horizon=5, embargo=1, session_id=sessions)
    assert out["X"].shape == (0, 20, 3)
    assert out["y"].shape == (0,)
