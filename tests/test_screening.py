"""The gate: shuffle-tested, absolute, lagged one session, and not fooled by zig-zags."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import build_dataset
from src.screening import apply_gate, screen_symbol, screen_universe
from src.synthetic import ar_returns, bars_from_returns

SESSIONS = 40


def _screen(returns, price=400.0, tick=None, symbol="TEST"):
    df = bars_from_returns(symbol, returns, price=price, tick=tick)
    table = screen_symbol(df, window=390, m=4, n_surrogates=99, min_bars_required=300, symbol=symbol)
    table.insert(1, "symbol", symbol)
    return df, table


def _features_cfg():
    return SimpleNamespace(
        features=SimpleNamespace(
            channels=["log_return", "session_return"], window=24, horizon=6, embargo=1,
            stride=1, allow_overnight=False, normalize="none",
        )
    )


def test_reading_applies_to_the_following_session():
    _, table = _screen(ar_returns(SESSIONS, 0.0))
    dated = table.dropna(subset=["trade_date"])
    assert (dated["trade_date"] > dated["session_date"]).all()
    days = list(table["session_date"])
    for _, row in dated.iterrows():
        assert row["trade_date"] == days[days.index(row["session_date"]) + 1]
    assert pd.isna(table.iloc[-1]["trade_date"])


def test_noise_passes_at_about_the_nominal_rate():
    rates = []
    for seed in range(4):
        _, table = _screen(ar_returns(60, 0.0, seed=seed))
        rates.append(apply_gate(table)["pass_trend"].mean())
    assert np.mean(rates) < 0.15


def test_trend_passes_and_zigzag_does_not():
    _, trend = _screen(ar_returns(SESSIONS, +0.3, seed=1))
    _, zigzag = _screen(ar_returns(SESSIONS, -0.3, seed=2))
    trend, zigzag = apply_gate(trend), apply_gate(zigzag)
    assert trend["pass_trend"].mean() > 0.7
    assert zigzag["pass_trend"].mean() == 0.0
    # plain entropy cannot tell them apart -- that is why it is not the default
    assert zigzag["pass_entropy"].mean() > 0.7


def test_tick_constrained_prices_are_refused_under_every_statistic():
    _, table = _screen(ar_returns(SESSIONS, +0.3, scale=0.0008, seed=3), price=12.0, tick=0.01)
    gated = apply_gate(table, min_ticks_per_bar=6)
    assert gated["tick_limited"].all()
    assert not gated[["pass_trend", "pass_entropy", "pass_entropy_weighted", "passed"]].any().any()


def test_gate_is_absolute_so_a_day_can_approve_nothing():
    bars = {s: bars_from_returns(s, ar_returns(SESSIONS, 0.0, seed=i)) for i, s in enumerate("ABCD")}
    cfg = SimpleNamespace(embedding_dim=4, delay=1, window=390, series="log_return",
                          n_surrogates=99, min_bars_required=300, seed=0)
    gated = apply_gate(screen_universe(bars, cfg))
    per_day = gated.groupby("session_date")["passed"].sum()
    assert (per_day == 0).mean() > 0.5


def test_readings_are_reproducible():
    df = bars_from_returns("X", ar_returns(SESSIONS, 0.1))
    a = screen_symbol(df, 390, symbol="X", seed=7)
    b = screen_symbol(df, 390, symbol="X", seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_unknown_gate_statistic_is_rejected():
    _, table = _screen(ar_returns(SESSIONS, 0.0))
    with pytest.raises(ValueError, match="unknown gate statistic"):
        apply_gate(table, statistic="vibes")


def test_a_reading_never_marks_samples_from_the_day_it_was_computed():
    """The leak this guards against: an 11am sample approved by statistics measured at 4pm."""
    df, table = _screen(ar_returns(SESSIONS, 0.0))
    gated = apply_gate(table)
    computed_on = gated["session_date"].iloc[2]
    for col in ("pass_trend", "pass_entropy", "pass_entropy_weighted"):
        gated[col] = gated["session_date"] == computed_on

    data = build_dataset({"TEST": df}, gated, _features_cfg())
    marked_days = set(pd.to_datetime(data["date"][data["pass_trend"]]))
    assert computed_on not in marked_days
    assert marked_days == {gated["trade_date"].iloc[2]}


def test_every_sample_is_kept_whether_or_not_the_gate_approved_it():
    df, table = _screen(ar_returns(SESSIONS, 0.0))
    gated = apply_gate(table)
    gated[["pass_trend", "pass_entropy", "pass_entropy_weighted"]] = False
    with_gate = build_dataset({"TEST": df}, gated, _features_cfg())
    without = build_dataset({"TEST": df}, None, _features_cfg())
    assert len(with_gate["y"]) == len(without["y"])
    assert not with_gate["pass_trend"].any()


def test_stale_screen_table_is_rejected():
    df, table = _screen(ar_returns(SESSIONS, 0.0))
    with pytest.raises(ValueError, match="older version"):
        build_dataset({"TEST": df}, table.drop(columns="trade_date"), _features_cfg())
