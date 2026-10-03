"""Entropy as features: correct, causal, missing when it should be, and honest about its blind spot."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import build_dataset
from src.entropy import permutation_entropy
from src.entropy_features import (
    ENTROPY_COLUMNS, OVERNIGHT, apply_entropy_inputs, entropy_matrix, fit_entropy_inputs, intraday_entropy,
)
from src.screening import apply_gate, screen_universe
from src.synthetic import BARS, ar_returns, bars_from_returns, regime_returns


def test_intraday_features_match_a_direct_calculation_on_the_trailing_returns():
    r = np.random.default_rng(0).normal(size=1000)
    bars = np.array([400, 700, 999])
    out = intraday_entropy(r, bars)
    for k, b in enumerate(bars):
        day = r[b - 77 : b + 1][-60:]
        hourly = r[b - 383 : b + 1].reshape(32, 12).sum(axis=1)[-28:]
        assert out["ent_pe3_day"][k] == pytest.approx(permutation_entropy(day, m=3))
        assert out["ent_pe4_day"][k] == pytest.approx(permutation_entropy(day, m=4))
        assert out["ent_pe3_hour"][k] == pytest.approx(permutation_entropy(hourly, m=3))


def test_features_never_look_past_the_signal_bar():
    rng = np.random.default_rng(1)
    r = rng.normal(size=1000)
    changed = r.copy()
    changed[601:] = rng.normal(size=399)
    a, b = intraday_entropy(r, np.array([600])), intraday_entropy(changed, np.array([600]))
    for col in a:
        assert a[col][0] == b[col][0]


def test_too_little_history_is_missing_not_a_number():
    r = np.random.default_rng(2).normal(size=500)
    early = intraday_entropy(r, np.array([30]))
    assert all(np.isnan(v[0]) for v in early.values())
    gappy = r.copy()
    gappy[400:430] = np.nan                      # 30 of the last 78 bars missing
    out = intraday_entropy(gappy, np.array([450]))
    assert np.isnan(out["ent_pe3_day"][0]) and np.isnan(out["ent_trend3_day"][0])


def test_ordered_returns_read_lower_entropy_and_more_runs_than_noise():
    bars = np.arange(5 * BARS, 60 * BARS, BARS) - 1
    trend = intraday_entropy(ar_returns(60, +0.4, seed=3), bars)
    noise = intraday_entropy(ar_returns(60, 0.0, seed=3), bars)
    assert np.mean(trend["ent_trend3_day"]) > np.mean(noise["ent_trend3_day"]) + 0.03
    assert np.mean(trend["ent_pe3_day"]) < np.mean(noise["ent_pe3_day"])


def test_slow_drift_is_nearly_invisible_to_five_minute_patterns_but_shows_hourly():
    """The blind spot behind the hourly features: a drift that barely changes over three bars
    shifts their ranks together, so 5-minute ordinal patterns read it as noise."""
    r, regime = regime_returns(400, seed=3)
    bars = np.arange(5 * BARS, 400 * BARS + 1, BARS) - 1           # each session's last bar
    out = intraday_entropy(r, bars)
    on = regime[4:] == 1

    def separation(col):
        a, b = out[col][on], out[col][~on]
        return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)

    assert abs(separation("ent_trend3_day")) < 0.3
    assert separation("ent_trend3_hour") > 0.5


def _cfg():
    return SimpleNamespace(features=SimpleNamespace(
        channels=["log_return", "session_return"], window=12, horizon=30, embargo=1,
        stride=12, allow_overnight=False, normalize="none"))


def test_dataset_attaches_overnight_readings_from_the_previous_close():
    df = bars_from_returns("T", ar_returns(30, 0.1, seed=4))
    s = SimpleNamespace(embedding_dim=4, delay=1, window=390, series="log_return",
                        n_surrogates=19, min_bars_required=300, seed=0)
    screen = apply_gate(screen_universe({"T": df}, s))
    data = build_dataset({"T": df}, screen, _cfg())

    readings = screen.dropna(subset=["trade_date"]).set_index("trade_date")
    dated = data["has_reading"]
    assert dated.any() and not dated.all()
    for i in np.flatnonzero(dated)[:20]:
        row = readings.loc[data["date"][i]]
        assert data["ent_z_pe"][i] == pytest.approx(row["z_unweighted"])
        assert data["ent_log_cents"][i] == pytest.approx(np.log1p(row["ticks_per_bar"]))
    assert np.isnan(data["ent_z_pe"][~dated]).all()


def test_without_a_screen_the_overnight_features_are_missing_and_intraday_ones_are_not():
    data = build_dataset({"T": bars_from_returns("T", ar_returns(30, 0.1, seed=5))}, None, _cfg())
    assert set(ENTROPY_COLUMNS) <= set(data)
    assert np.isnan(data["ent_z_pe"]).all()
    assert np.isfinite(data["ent_pe3_day"]).mean() > 0.9


def test_neural_inputs_are_scaled_from_training_rows_and_flag_missing_readings():
    rng = np.random.default_rng(6)
    E = rng.normal(size=(200, len(ENTROPY_COLUMNS)))
    E[:30, : len(OVERNIGHT)] = np.nan                        # warm-up days without an overnight reading
    stats = fit_entropy_inputs(E[:150])
    train, test = apply_entropy_inputs(E[:150], stats), apply_entropy_inputs(E[150:] * 100, stats)
    assert train.shape[1] == len(ENTROPY_COLUMNS) + 1        # plus the missing-reading flag
    np.testing.assert_allclose(train[:, :-1].mean(axis=0), 0, atol=1e-6)
    assert np.abs(test[:, :-1]).max() > 10                   # test rows are not rescaled to themselves
    assert (train[:30, -1] == 0).all() and (train[30:, -1] == 1).all()
    assert np.isfinite(train).all()

    no_screen = E.copy()
    no_screen[:, : len(OVERNIGHT)] = np.nan
    stats = fit_entropy_inputs(no_screen[:150])
    assert apply_entropy_inputs(no_screen, stats).shape[1] == len(ENTROPY_COLUMNS) - len(OVERNIGHT)


def test_entropy_matrix_follows_the_column_order():
    data = {c: np.full(3, float(i)) for i, c in enumerate(ENTROPY_COLUMNS)}
    np.testing.assert_array_equal(entropy_matrix(data, np.array([0, 2]))[0], np.arange(len(ENTROPY_COLUMNS)))
