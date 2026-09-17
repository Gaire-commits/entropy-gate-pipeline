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


def half_hour_momentum_returns(n_sessions: int, beta: float, scale: float = 0.001, seed: int = 0) -> np.ndarray:
    """Noise, except the last half-hour tends to follow the first half-hour's direction.

    `beta` is how much of the first half-hour's move carries into the last one.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(0, scale, (n_sessions, BARS))
    first = r[:, :6].sum(axis=1)
    r[:, 72:] += beta * first[:, None] / 6
    return r.ravel()
