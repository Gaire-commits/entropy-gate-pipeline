"""The predictability gate.

One reading per symbol per session, computed after that session's close on the
trailing window, deciding eligibility for the *next* session: the gate is a
nightly batch, and a live system cannot know today's closing statistics while
today is still trading.

Each reading tests the window against shuffled copies of itself (see
`entropy.surrogate_test`), so fat tails and a few giant moves cannot pass for
structure. The gate is absolute: on a day when nothing beats its shuffles,
nothing trades. It never looks at forward returns, so it cannot leak label
information into selection.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from .entropy import surrogate_test, tie_fraction, embed

SERIES_BUILDERS = {
    "log_return": lambda df: np.log(df["close"] / df["close"].shift(1)).to_numpy(),
    "signed_volume": lambda df: (
        np.sign(df["close"].diff()).fillna(0.0) * np.log1p(df["volume"].astype(float))
    ).to_numpy(),
}

GATE_STATISTICS = {
    "trend": "p_trend",
    "entropy": "p_unweighted",
    "entropy_weighted": "p_weighted",
}


def _symbol_rng(seed: int, symbol: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(symbol.encode())])


def screen_symbol(
    df: pd.DataFrame,
    window: int,
    m: int = 4,
    tau: int = 1,
    series: str = "log_return",
    n_surrogates: int = 99,
    tie_handling: str = "stable",
    min_bars_required: int = 300,
    seed: int = 0,
    symbol: str = "",
) -> pd.DataFrame:
    """Trailing-window ordinal statistics, one row per session.

    `session_date` is when the reading was computed (that session's close).
    `trade_date` is the following session, the first one allowed to act on it.
    The final session has no following session in the data, so its trade_date
    is NaT -- live, that reading applies to tomorrow.

    `ticks_per_bar` is the typical bar-to-bar price change measured in cents.
    Below a handful of cents, rounding to the tick makes returns zig-zag on its
    own, which is order in the numbers but not in the market.
    """
    if series not in SERIES_BUILDERS:
        raise ValueError(f"unknown series '{series}'; options: {sorted(SERIES_BUILDERS)}")
    if "session_id" not in df.columns:
        raise ValueError("frame needs a session_id column (see data.to_session_grid)")

    rng = _symbol_rng(seed, symbol)
    x = SERIES_BUILDERS[series](df)
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    sessions = df["session_id"].to_numpy()
    timestamps = df.index.get_level_values("timestamp")

    starts = np.r_[0, np.flatnonzero(np.diff(sessions)) + 1]
    ends = np.r_[starts[1:] - 1, len(sessions) - 1]
    days = timestamps[starts].normalize().tz_localize(None)

    rows = []
    for k in range(len(starts)):
        end = ends[k]
        start = end - window + 1
        if start < 1:
            continue
        seg = x[start : end + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size < min_bars_required:
            continue

        stats = surrogate_test(seg, m, tau, n_surrogates, tie_handling, rng)
        price_steps = np.diff(close[start - 1 : end + 1])
        price_steps = price_steps[np.isfinite(price_steps)]
        dollar_volume = np.nansum(close[start : end + 1] * volume[start : end + 1])

        rows.append(
            {
                "session_date": days[k],
                "trade_date": days[k + 1] if k + 1 < len(starts) else pd.NaT,
                **stats,
                "tie_fraction": tie_fraction(embed(seg, m, tau)),
                "ticks_per_bar": float(price_steps.std() / 0.01) if price_steps.size > 1 else 0.0,
                "dollar_volume": float(dollar_volume),
                "n_bars": int(seg.size),
            }
        )
    return pd.DataFrame(rows)


def screen_universe(bars: dict[str, pd.DataFrame], cfg, progress=None) -> pd.DataFrame:
    """Run the screen across every cached symbol."""
    frames = []
    for symbol, df in bars.items():
        table = screen_symbol(
            df,
            window=cfg.window,
            m=cfg.embedding_dim,
            tau=cfg.delay,
            series=cfg.series,
            n_surrogates=cfg.n_surrogates,
            min_bars_required=cfg.min_bars_required,
            seed=cfg.seed,
            symbol=symbol,
        )
        if progress:
            progress(symbol, table)
        if table.empty:
            continue
        table.insert(1, "symbol", symbol)
        frames.append(table)
    if not frames:
        return pd.DataFrame(columns=["session_date", "trade_date", "symbol"])
    return pd.concat(frames, ignore_index=True).sort_values(["session_date", "symbol"])


def apply_gate(
    table: pd.DataFrame,
    statistic: str = "trend",
    alpha: float = 0.05,
    max_tie_fraction: float = 0.5,
    min_ticks_per_bar: float = 6.0,
) -> pd.DataFrame:
    """Mark which readings pass, for every gate statistic.

    `pass_trend`, `pass_entropy` and `pass_entropy_weighted` are all recorded so
    the gates can be compared on the same predictions later; `passed` is the one
    named by `statistic`. Stale windows (mostly repeated prices) and
    tick-constrained windows are refused under every statistic.

    Under pure noise about `alpha` of readings still pass. That is the expected
    false-positive rate, and it is why the gate is judged by what happens on the
    days it approves, not by how many days it approves.
    """
    if statistic not in GATE_STATISTICS:
        raise ValueError(f"unknown gate statistic '{statistic}'; options: {sorted(GATE_STATISTICS)}")
    out = table.copy()
    out["stale"] = out["tie_fraction"] > max_tie_fraction
    out["tick_limited"] = out["ticks_per_bar"] < min_ticks_per_bar
    eligible = ~out["stale"] & ~out["tick_limited"]
    for name, column in GATE_STATISTICS.items():
        out[f"pass_{name}"] = eligible & (out[column] <= alpha)
    out["passed"] = out[f"pass_{statistic}"]
    return out


def capacity_diagnostic(gated: pd.DataFrame) -> pd.DataFrame:
    """Does the gate systematically select thinner names?

    If the mechanism that makes a name readable is the same one that makes it
    illiquid, the strategy's capacity ceiling is set by its own selection logic.
    This table is the evidence either way.
    """
    rows = []
    for date, day in gated.groupby("session_date"):
        passed = day[day["passed"]]
        if passed.empty:
            continue
        rows.append(
            {
                "session_date": date,
                "n_passed": len(passed),
                "n_universe": len(day),
                "median_dv_passed": passed["dollar_volume"].median(),
                "median_dv_universe": day["dollar_volume"].median(),
                "dv_ratio": passed["dollar_volume"].median() / day["dollar_volume"].median(),
            }
        )
    return pd.DataFrame(rows)
