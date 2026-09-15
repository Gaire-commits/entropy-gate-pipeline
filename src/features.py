"""Causal windowing and encoders that turn OHLCV bars into model input.

Every function here is strictly backward-looking. The one place lookahead can
enter a price model is the boundary between a feature window and its label, so
that boundary is made explicit: a signal read at the end of a window is acted
on `embargo` bars later and held for `horizon` bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CHANNEL_BUILDERS = {}


def _channel(name):
    def wrap(fn):
        CHANNEL_BUILDERS[name] = fn
        return fn

    return wrap


@_channel("log_return")
def _log_return(df: pd.DataFrame) -> np.ndarray:
    return np.log(df["close"] / df["close"].shift(1)).to_numpy()


@_channel("hl_range")
def _hl_range(df: pd.DataFrame) -> np.ndarray:
    return ((df["high"] - df["low"]) / df["close"]).to_numpy()


@_channel("close_open")
def _close_open(df: pd.DataFrame) -> np.ndarray:
    return ((df["close"] - df["open"]) / df["open"]).to_numpy()


@_channel("volume_z")
def _volume_z(df: pd.DataFrame, lookback: int = 78) -> np.ndarray:
    v = df["volume"].astype(float)
    mu = v.rolling(lookback, min_periods=lookback // 2).mean()
    sd = v.rolling(lookback, min_periods=lookback // 2).std()
    return ((v - mu) / sd.replace(0, np.nan)).to_numpy()


@_channel("signed_volume")
def _signed_volume(df: pd.DataFrame) -> np.ndarray:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    v = df["volume"].astype(float)
    return (direction * np.log1p(v)).to_numpy()


def build_channels(df: pd.DataFrame, channels: list[str]) -> np.ndarray:
    """Stack named channels into (n_bars, n_channels). Warm-up rows hold NaN."""
    missing = set(channels) - CHANNEL_BUILDERS.keys()
    if missing:
        raise ValueError(f"unknown channels: {sorted(missing)}")
    return np.column_stack([CHANNEL_BUILDERS[c](df) for c in channels])


def make_windows(
    features: np.ndarray,
    close: np.ndarray,
    window: int,
    horizon: int,
    embargo: int,
    session_id: np.ndarray | None = None,
    deadzone: float = 0.0,
    allow_overnight: bool = False,
) -> dict[str, np.ndarray]:
    """Cut causal (X, y) pairs.

    Window i spans bars [t, t+window-1]. Entry happens `embargo` bars after the
    last bar in the window, exit `horizon` bars after entry, so no label can be
    read from a bar the features already saw.

    With `session_id` and `allow_overnight=False`, the whole span from first
    feature bar to exit must sit inside one session. Checking the window and the
    trade separately is not enough: that admits samples whose features come from
    one day and whose fill comes from the next, which is an overnight strategy
    priced as if it were intraday.
    """
    n_bars, n_ch = features.shape
    X, y, ret, idx = [], [], [], []

    if session_id is not None and not allow_overnight:
        span = window + embargo + horizon
        _, counts = np.unique(session_id, return_counts=True)
        if span > counts.max():
            raise ValueError(
                f"window({window}) + embargo({embargo}) + horizon({horizon}) = {span} bars "
                f"exceeds the longest session ({counts.max()} bars); no intraday sample can fit. "
                "Shorten the window/horizon or set allow_overnight=True."
            )

    last_start = n_bars - window - embargo - horizon
    for t in range(max(0, last_start + 1)):
        w_end = t + window - 1
        entry = w_end + embargo
        exit_ = entry + horizon
        if exit_ >= n_bars:
            break

        block = features[t : w_end + 1]
        if not np.isfinite(block).all():
            continue
        if session_id is not None and not allow_overnight and session_id[t] != session_id[exit_]:
            continue
        if session_id is not None and allow_overnight and session_id[t] != session_id[w_end]:
            continue

        fwd = float(np.log(close[exit_] / close[entry]))
        if not np.isfinite(fwd) or abs(fwd) < deadzone:
            continue

        X.append(block)
        ret.append(fwd)
        y.append(1 if fwd > 0 else 0)
        idx.append(w_end)

    if not X:
        return {
            "X": np.empty((0, window, n_ch)),
            "y": np.empty(0, dtype=np.int64),
            "ret": np.empty(0),
            "signal_bar": np.empty(0, dtype=np.int64),
        }
    return {
        "X": np.stack(X),
        "y": np.asarray(y, dtype=np.int64),
        "ret": np.asarray(ret),
        "signal_bar": np.asarray(idx, dtype=np.int64),
    }


def normalize_windows(X: np.ndarray, mode: str = "window_zscore") -> np.ndarray:
    """Scale each window using only that window's own statistics.

    Global or training-set scaling would leak distributional information across
    the walk-forward boundary; per-window scaling cannot.
    """
    if mode == "none":
        return X
    if mode == "window_zscore":
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        return (X - mu) / np.where(sd > 0, sd, 1.0)
    if mode == "window_minmax":
        lo = X.min(axis=1, keepdims=True)
        hi = X.max(axis=1, keepdims=True)
        span = hi - lo
        return 2 * (X - lo) / np.where(span > 0, span, 1.0) - 1
    raise ValueError(f"unknown normalize mode: {mode}")


def gaf_encode(x: np.ndarray, kind: str = "summation") -> np.ndarray:
    """Gramian Angular Field: (n, L) series -> (n, L, L) image.

    Encodes temporal correlation as 2-D spatial structure, which is the actual
    justification for putting an image architecture like ResNet-2D on a series.
    """
    x = np.atleast_2d(x)
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    scaled = np.clip(2 * (x - lo) / span - 1, -1.0, 1.0)
    comp = np.sqrt(np.clip(1 - scaled**2, 0.0, None))

    if kind == "summation":
        return scaled[:, :, None] * scaled[:, None, :] - comp[:, :, None] * comp[:, None, :]
    if kind == "difference":
        return comp[:, :, None] * scaled[:, None, :] - scaled[:, :, None] * comp[:, None, :]
    raise ValueError(f"unknown GAF kind: {kind}")


def mtf_encode(x: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Markov Transition Field: (n, L) series -> (n, L, L) transition-probability image."""
    x = np.atleast_2d(x)
    n, length = x.shape
    out = np.zeros((n, length, length))

    for i in range(n):
        edges = np.quantile(x[i], np.linspace(0, 1, n_bins + 1)[1:-1])
        q = np.searchsorted(edges, x[i])
        trans = np.zeros((n_bins, n_bins))
        np.add.at(trans, (q[:-1], q[1:]), 1.0)
        row = trans.sum(axis=1, keepdims=True)
        trans /= np.where(row > 0, row, 1.0)
        out[i] = trans[np.ix_(q, q)]
    return out
