"""Synthetic bars with known structure, for tests and the smoke test.

Everything comes out in the same shape real cached data does after
`data.to_session_grid`, so the pipeline cannot tell the difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import to_session_grid

BARS = 78


def bars_from_returns(
    symbol: str,
    returns: np.ndarray,
    price: float = 400.0,
    start: str = "2016-01-04",
    seed: int = 0,
    tick: float | None = None,
) -> pd.DataFrame:
    """Turn a flat array of per-bar log returns (n_sessions * 78) into gridded OHLCV bars."""
    rng = np.random.default_rng(seed)
    n = returns.size
    if n % BARS:
        raise ValueError(f"returns length {n} is not a whole number of {BARS}-bar sessions")
    close = price * np.exp(np.cumsum(returns))
    if tick:
        close = np.round(close / tick) * tick
    open_ = np.r_[price, close[:-1]]
    spread = np.abs(rng.normal(0, 0.0004, n)) * close

    days = pd.bdate_range(start, periods=n // BARS)
    offsets = pd.timedelta_range("09:30:00", periods=BARS, freq="5min")
    ts = pd.DatetimeIndex((days.values[:, None] + offsets.values[None, :]).ravel()).tz_localize("America/New_York")
    raw = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.lognormal(11, 0.6, n),
        },
        index=pd.MultiIndex.from_arrays([np.full(n, symbol), ts], names=["symbol", "timestamp"]),
    )
    return to_session_grid(raw, 5)


def ar_returns(n_sessions: int, phi: float, scale: float = 0.001, seed: int = 0) -> np.ndarray:
    """AR(1) bar returns with fat-tailed shocks. phi > 0 trends, phi < 0 zig-zags."""
    rng = np.random.default_rng(seed)
    e = rng.standard_t(4, n_sessions * BARS) * scale / np.sqrt(2)
    r = np.zeros_like(e)
    for i in range(1, r.size):
        r[i] = phi * r[i - 1] + e[i]
    return r


def regime_returns(n_sessions: int, regime_sessions: int = 10, scale: float = 0.001, drift: float = 0.0006,
                   persistence: float = 0.98, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Bar returns that switch between a trending and a noise regime, in spells of about
    `regime_sessions` sessions.

    Trending: a slowly varying drift (an AR(1) with `persistence`, standard deviation
    `drift`) under the noise, so a window's direction tends to carry on. Noise: nothing
    to predict, so a model that follows the window just pays the cost. The noise is
    scaled so both regimes have the same volatility: a model cannot tell them apart
    by the size of the moves. A short window shows little of which regime it is in; a
    few sessions of ordinal statistics show it clearly. This is the case where the
    entropy features should tell a model when to trust its signal.
    Returns (bar returns, 1.0 for trending sessions and 0.0 for noise).
    """
    rng = np.random.default_rng(seed)
    regime = np.empty(n_sessions)
    state, left = 1.0, 0
    for s in range(n_sessions):
        if left == 0:
            state = float(rng.integers(0, 2))
            left = max(2, int(rng.poisson(regime_sessions)))
        regime[s], left = state, left - 1
    on = np.repeat(regime, BARS)
    mu = np.zeros(n_sessions * BARS)
    shocks = rng.normal(0, drift * np.sqrt(1 - persistence**2), mu.size)
    for i in range(1, mu.size):
        mu[i] = persistence * mu[i - 1] + shocks[i]
    noise_sd = np.where(on > 0, np.sqrt(scale**2 - drift**2), scale)
    return on * mu + noise_sd * rng.normal(size=mu.size), regime


def half_hour_momentum_returns(n_sessions: int, beta: float, scale: float = 0.001, seed: int = 0) -> np.ndarray:
    """Noise, except the last half-hour tends to follow the first half-hour's direction.

    `beta` is how much of the first half-hour's move carries into the last one.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(0, scale, (n_sessions, BARS))
    first = r[:, :6].sum(axis=1)
    r[:, 72:] += beta * first[:, None] / 6
    return r.ravel()
