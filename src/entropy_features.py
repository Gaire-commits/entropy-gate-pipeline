"""Entropy as features: the gate's readings become model inputs instead of a pass/fail filter.

The hard gate passed about as often as pure noise and threw away everything about a
reading except one bit. Here the same ordinal statistics are kept as continuous
numbers, at two time scales, and the models and the decision engine decide how
much they matter.

overnight: the trailing 5-session reading from scripts/run_screen.py, taken at the
    previous close and attached to the next session (as the gate was). Entropy and
    weighted entropy as z-scores against shuffled copies of the same returns, the
    share of monotone patterns (steady runs) and its excess over the shuffles,
    statistical complexity, and the two data-quality checks: the share of tied
    values and the typical bar-to-bar move in cents.
intraday: the last 60 five-minute returns ending at the sample's signal bar, which
    early in a session reach back into the previous one: permutation entropy at
    m = 3 and m = 4 and the monotone share at m = 3. And the same two m = 3
    statistics on hourly returns (sums of 12 bars) over the trailing 5 sessions.
    Same bars the model's window ends on, so nothing from the future.

Why the hourly scale: ordinal patterns of 5-minute returns are nearly blind to a
slowly varying drift, the kind of trend multi-hour momentum lives on. A drift that
barely changes over three bars shifts all three returns alike and leaves their
ranks, and so the patterns, as noise would. For a Gaussian process the monotone
share at m = 3 depends on rho1 and rho2 only through (2 rho1 - 1 - rho2) / (2 (1 - rho1)),
so a drift with rho1 ~ rho2 reads like noise however strong it is. Summing returns
over an hour lets the drift outgrow the noise and become visible to the patterns.

Missing values stay NaN here: warm-up days before the first overnight reading, and
signal bars with fewer than 60 valid returns in the trailing 78 bars. Trees take
NaN as its own branch; neural inputs are filled with training-fold medians and get
an explicit missing flag, so a missing reading never looks like a low or high one.
"""

from __future__ import annotations

import numpy as np

from .entropy import _row_entropy, batched_pattern_distributions, monotone_codes

# screen column -> feature name
OVERNIGHT = {
    "z_unweighted": "ent_z_pe",
    "z_weighted": "ent_z_wpe",
    "monotone_share": "ent_trend_share",
    "monotone_excess": "ent_trend_excess",
    "complexity": "ent_complexity",
    "tie_fraction": "ent_ties",
    "ticks_per_bar": "ent_log_cents",
}
INTRADAY = ["ent_pe3_day", "ent_pe4_day", "ent_trend3_day", "ent_pe3_hour", "ent_trend3_hour"]
ENTROPY_COLUMNS = list(OVERNIGHT.values()) + INTRADAY

LOOKBACK = 78   # search one session's worth of bars back from the signal bar
LENGTH = 60     # and use the last 60 valid returns found there
HOUR, HOURS = 12, 32     # hourly returns over the trailing 5 sessions (32 x 12 = 384 bars)
MIN_HOURS = 28           # hours with at least 3 of their 4 quarter-hours present


def overnight_values(screen_column: str, values: np.ndarray) -> np.ndarray:
    """Screen statistic -> feature value. Cents per bar spans orders of magnitude, so it is logged."""
    values = np.asarray(values, dtype=float)
    return np.log1p(values) if screen_column == "ticks_per_bar" else values


def _trailing(r: np.ndarray, bars: np.ndarray, n: int) -> np.ndarray:
    """(len(bars), n) matrix of the n values ending at each bar, NaN before the series starts."""
    idx = bars[:, None] - np.arange(n - 1, -1, -1)[None, :]        # oldest ... signal bar
    return np.where(idx >= 0, r[np.clip(idx, 0, None)], np.nan)


def _last_valid(M: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows with at least `length` finite values, compacted to their last `length` (order kept)."""
    finite = np.isfinite(M)
    enough = finite.sum(axis=1) >= length
    order = np.argsort(finite, axis=1, kind="stable")
    return np.take_along_axis(M, order, axis=1)[enough, -length:], enough


def intraday_entropy(log_returns: np.ndarray, signal_bars: np.ndarray,
                     lookback: int = LOOKBACK, length: int = LENGTH) -> dict[str, np.ndarray]:
    """Ordinal statistics of the trailing returns at each signal bar, for one symbol's bar series."""
    r = np.asarray(log_returns, dtype=float)
    bars = np.asarray(signal_bars, dtype=np.int64)
    out = {name: np.full(len(bars), np.nan) for name in INTRADAY}
    if len(bars) == 0:
        return out
    up, down = monotone_codes(3)

    compact, enough = _last_valid(_trailing(r, bars, lookback), length)
    if enough.any():
        p3, _ = batched_pattern_distributions(compact, m=3)
        p4, _ = batched_pattern_distributions(compact, m=4)
        out["ent_pe3_day"][enough] = _row_entropy(p3, 3)
        out["ent_pe4_day"][enough] = _row_entropy(p4, 4)
        out["ent_trend3_day"][enough] = p3[:, up] + p3[:, down]

    # Hourly returns: blocks of 12 bars ending at the signal bar. A block with a missing
    # quarter-hour or more is dropped rather than summed short.
    blocks = _trailing(r, bars, HOUR * HOURS).reshape(len(bars), HOURS, HOUR)
    present = np.isfinite(blocks).sum(axis=2) >= HOUR - 3
    hourly = np.where(present, np.nansum(blocks, axis=2), np.nan)
    compact, enough = _last_valid(hourly, MIN_HOURS)
    if enough.any():
        p3, _ = batched_pattern_distributions(compact, m=3)
        out["ent_pe3_hour"][enough] = _row_entropy(p3, 3)
        out["ent_trend3_hour"][enough] = p3[:, up] + p3[:, down]
    return out


def entropy_matrix(data: dict, rows: np.ndarray | None = None) -> np.ndarray:
    """(n, len(ENTROPY_COLUMNS)) float matrix with NaN where a value is missing."""
    cols = [np.asarray(data[c], dtype=float) for c in ENTROPY_COLUMNS]
    M = np.column_stack(cols)
    return M if rows is None else M[rows]


def fit_entropy_inputs(E_train: np.ndarray) -> dict:
    """Fill and scale statistics for neural inputs, from the training fold only.

    Columns with no value at all in training (overnight features when no screen was
    run) are dropped. One flag column marks whether the overnight reading exists.
    """
    keep = np.isfinite(E_train).any(axis=0)
    E = E_train[:, keep]
    median = np.nanmedian(E, axis=0)
    filled = np.where(np.isfinite(E), E, median)
    sd = filled.std(axis=0)
    overnight = keep[: len(OVERNIGHT)].any()
    return {"keep": keep, "median": median, "mean": filled.mean(axis=0),
            "sd": np.where(sd > 0, sd, 1.0), "flag": bool(overnight)}


def apply_entropy_inputs(E: np.ndarray, stats: dict) -> np.ndarray:
    kept = E[:, stats["keep"]]
    filled = np.where(np.isfinite(kept), kept, stats["median"])
    scaled = (filled - stats["mean"]) / stats["sd"]
    if stats["flag"]:
        present = np.isfinite(E[:, : len(OVERNIGHT)]).all(axis=1).astype(float)
        scaled = np.column_stack([scaled, present])
    return scaled.astype(np.float32)
