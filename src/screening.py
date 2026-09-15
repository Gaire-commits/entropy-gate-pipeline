"""The predictability gate.

One entropy reading per symbol per session, computed after that session's close
on the trailing window, then a cross-sectional cut that keeps the most ordered
names. A reading decides eligibility for the *next* session: the gate is a
nightly batch, and a live system cannot know today's closing entropy while today
is still trading. It never looks at forward returns, so it cannot leak label
information into selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .entropy import permutation_entropy, statistical_complexity, tie_fraction, ordinal_patterns

SERIES_BUILDERS = {
    "log_return": lambda df: np.log(df["close"] / df["close"].shift(1)).to_numpy(),
    "price": lambda df: df["close"].to_numpy(),
    "signed_volume": lambda df: (
        np.sign(df["close"].diff()).fillna(0.0) * np.log1p(df["volume"].astype(float))
    ).to_numpy(),
}


def screen_symbol(
    df: pd.DataFrame,
    window: int,
    m: int = 4,
    tau: int = 1,
    weighted: bool = True,
    series: str = "log_return",
    tie_handling: str = "stable",
    min_bars_required: int = 300,
) -> pd.DataFrame:
    """Trailing-window entropy readings, one row per session.

    `session_date` is when the reading was computed (that session's close).
    `trade_date` is the following session, the first one allowed to act on it.
    The final session has no following session in the data, so its trade_date
    is NaT -- live, that reading applies to tomorrow.
    """
    if series not in SERIES_BUILDERS:
        raise ValueError(f"unknown series '{series}'; options: {sorted(SERIES_BUILDERS)}")
    if "session_id" not in df.columns:
        raise ValueError("frame needs a session_id column (see data.add_session_id)")

    x = SERIES_BUILDERS[series](df)
    sessions = df["session_id"].to_numpy()
    timestamps = df.index.get_level_values("timestamp")
    dollar_volume = (df["close"] * df["volume"]).to_numpy()

    unique_sessions = np.unique(sessions)
    first_bar = {s: int(np.flatnonzero(sessions == s)[0]) for s in unique_sessions}

    rows = []
    for k, session in enumerate(unique_sessions):
        end = int(np.flatnonzero(sessions == session)[-1])
        trade_date = (
            timestamps[first_bar[unique_sessions[k + 1]]].normalize().tz_localize(None)
            if k + 1 < len(unique_sessions)
            else pd.NaT
        )
        start = end - window + 1
        if start < 1 or end - start + 1 < min_bars_required:
            continue

        seg = x[start : end + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size < min_bars_required:
            continue

        _, vectors = ordinal_patterns(seg, m, tau, tie_handling)
        h_norm, complexity = statistical_complexity(seg, m, tau, weighted, tie_handling)
        rows.append(
            {
                "session_date": timestamps[end].normalize().tz_localize(None),
                "trade_date": trade_date,
                "pe":permutation_entropy(seg, m, tau, weighted, True, tie_handling),
                "pe_unweighted": permutation_entropy(seg, m, tau, False, True, tie_handling),
                "complexity": complexity,
                "tie_fraction": tie_fraction(vectors),
                "dollar_volume": float(np.nansum(dollar_volume[start : end + 1])),
                "n_bars": int(seg.size),
            }
        )
    return pd.DataFrame(rows)


def screen_universe(bars: dict[str, pd.DataFrame], cfg) -> pd.DataFrame:
    """Run the screen across every cached symbol."""
    frames = []
    for symbol, df in bars.items():
        table = screen_symbol(
            df,
            window=cfg.window,
            m=cfg.embedding_dim,
            tau=cfg.delay,
            weighted=cfg.weighted,
            series=cfg.series,
            tie_handling=cfg.tie_handling,
            min_bars_required=cfg.min_bars_required,
        )
        if table.empty:
            continue
        table.insert(1, "symbol", symbol)
        frames.append(table)
    if not frames:
        return pd.DataFrame(
            columns=["session_date", "symbol", "pe", "complexity", "tie_fraction", "dollar_volume"]
        )
    return pd.concat(frames, ignore_index=True).sort_values(["session_date", "pe"])


def apply_gate(table: pd.DataFrame, select_quantile: float = 0.30, max_tie_fraction: float = 0.5) -> pd.DataFrame:
    """Flag the low-entropy tail of each cross-section.

    A high tie fraction means the low reading came from repeated quotes rather
    than real order, so those names are refused regardless of where they rank.
    """
    out = table.copy()
    out["pe_rank"] = out.groupby("session_date")["pe"].rank(pct=True)
    out["stale"] = out["tie_fraction"] > max_tie_fraction
    out["passed"] = (out["pe_rank"] <= select_quantile) & ~out["stale"]
    return out


def capacity_diagnostic(gated: pd.DataFrame) -> pd.DataFrame:
    """Does the gate systematically select thinner names?

    If the mechanism that makes a name readable is the same one that makes it
    illiquid, the strategy's capacity ceiling is set by its own selection logic.
    This table is the evidence either way.
    """
    grouped = gated.groupby("session_date")
    rows = []
    for date, day in grouped:
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
                "mean_pe_passed": passed["pe"].mean(),
                "mean_pe_universe": day["pe"].mean(),
            }
        )
    return pd.DataFrame(rows)
