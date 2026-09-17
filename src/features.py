"""Causal windowing, normalization, and encoders that turn bars into model input.

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


@_channel("session_return")
def _session_return(df: pd.DataFrame) -> np.ndarray:
    return session_return(df)


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


def session_return(df: pd.DataFrame) -> np.ndarray:
    """Log return from the previous session's last close to each bar's close.

    Includes the overnight gap. At 10:00 this is the first-half-hour return
    that intraday momentum studies condition on.
    """
    sessions = df["session_id"]
    prev_close = df["close"].groupby(sessions).last().shift(1)
    return np.log(df["close"] / sessions.map(prev_close)).to_numpy()


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
    stride: int = 1,
    allow_overnight: bool = False,
) -> dict[str, np.ndarray]:
    """Cut causal (X, y) pairs.

    Window i spans bars [t, t+window-1]. Entry happens `embargo` bars after the
    last bar in the window, exit `horizon` bars after entry, so no label can be
    read from a bar the features already saw.

    With `session_id` and `allow_overnight=False`, the whole span from first
    feature bar to exit must sit inside one session. Checking the window and the
    trade separately is not enough: that admits samples whose features come from
    one day and whose fill comes from the next.

    `stride` counts from each session's first bar. Setting it to `horizon` makes
    consecutive holds on the same day back-to-back instead of overlapping, so
    each sample is a separate trade rather than a near-copy of its neighbour.

    No sample is dropped for having a small label. Filtering on the size of a
    future move is fine for choosing what to train on and lookahead for choosing
    what to test on, so that decision belongs to the caller.
    """
    n_bars, n_ch = features.shape
    X, y, ret, idx, win_ret = [], [], [], [], []

    if session_id is not None:
        starts = np.r_[0, np.flatnonzero(np.diff(session_id)) + 1]
        offset = np.arange(n_bars) - np.repeat(starts, np.diff(np.r_[starts, n_bars]))
        if not allow_overnight:
            span = window + embargo + horizon
            longest = np.diff(np.r_[starts, n_bars]).max()
            if span > longest:
                raise ValueError(
                    f"window({window}) + embargo({embargo}) + horizon({horizon}) = {span} bars "
                    f"exceeds the longest session ({longest} bars); no intraday sample can fit. "
                    "Shorten the window/horizon or set allow_overnight=True."
                )
    else:
        offset = np.arange(n_bars)

    last_start = n_bars - window - embargo - horizon
    for t in range(max(0, last_start + 1)):
        if offset[t] % stride:
            continue
        w_end = t + window - 1
        entry = w_end + embargo
        exit_ = entry + horizon

        if session_id is not None and not allow_overnight and session_id[t] != session_id[exit_]:
            continue
        if session_id is not None and allow_overnight and session_id[t] != session_id[w_end]:
            continue
        block = features[t : w_end + 1]
        if not np.isfinite(block).all():
            continue

        fwd = float(np.log(close[exit_] / close[entry]))
        if not np.isfinite(fwd):
            continue

        X.append(block)
        ret.append(fwd)
        y.append(1 if fwd > 0 else 0)
        idx.append(w_end)
        win_ret.append(float(np.log(close[w_end] / close[t - 1])) if t >= 1 else np.nan)

    if not X:
        return {
            "X": np.empty((0, window, n_ch)),
            "y": np.empty(0, dtype=np.int64),
            "ret": np.empty(0),
            "signal_bar": np.empty(0, dtype=np.int64),
            "window_return": np.empty(0),
        }
    return {
        "X": np.stack(X),
        "y": np.asarray(y, dtype=np.int64),
        "ret": np.asarray(ret),
        "signal_bar": np.asarray(idx, dtype=np.int64),
        "window_return": np.asarray(win_ret),
    }


def normalize_windows(X: np.ndarray, mode: str) -> np.ndarray:
    """Per-window scaling for (n, window, channels) arrays.

    `window_zscore` subtracts each window's mean, which zeroes the cumulative
    return of every sample -- the model can no longer see whether price went up
    or down over the window. It is kept only to reproduce earlier runs.
    `window_scale` divides by the window's spread and keeps the mean.
    """
    if mode in ("none", "fold_standardize"):
        return X
    if mode == "window_zscore":
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        return (X - mu) / np.where(sd > 0, sd, 1.0)
    if mode == "window_scale":
        sd = X.std(axis=1, keepdims=True)
        return X / np.where(sd > 0, sd, 1.0)
    raise ValueError(f"unknown normalize mode: {mode}")


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and spread from training samples, for (n, channels, length) arrays.

    Fit on the training fold only. The statistics then carry the training
    period's scale into validation and test, which is information the model
    would really have had at that point.
    """
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True)
    return mu, np.where(sd > 0, sd, 1.0)


def apply_standardizer(X: np.ndarray, stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mu, sd = stats
    return ((X - mu) / sd).astype(np.float32)


def gaf_encode(x: np.ndarray, kind: str = "summation") -> np.ndarray:
    """Gramian Angular Field: (n, L) series -> (n, L, L) image.

    Every row is min-max rescaled first, so absolute level is lost; feed it a
    cumulative path (session_return) rather than raw returns if the direction
    of travel should survive into the image.
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
