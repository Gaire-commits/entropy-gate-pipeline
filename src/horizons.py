"""E9: how does the reversal change with the horizon, from 30 minutes to 2 hours?

E5-E8 measured it at 30 minutes (the move over the last 6 bars, held for 6). The question here is
whether a longer lookback or hold makes it stronger. Two things have to be kept apart:

- **The amount in bps grows with the horizon on its own.** Moves scale with the square root of time,
  so the same skill per unit of risk pays about twice as much at 2 hours as at 30 minutes. A bigger
  number at a longer horizon is therefore no evidence that the reversal itself is stronger.
- **The rank IC does not.** It measures skill per unit of risk, so it is the quantity compared across
  horizons. The amount is reported next to it.

Horizons are cells of (lookback, hold) in 5-minute bars. The decisive comparison is between the
shortest and the longest symmetric cell, at the day level: moments at different horizons fall at
different times of day, so cells are compared through each day's average IC, not moment by moment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .crosssection import moment_ic

MOMENT = ["date", "bar_index"]


def day_ic(frame: pd.DataFrame, score: str = "prob", min_peers: int = 20) -> pd.Series:
    """Each day's average rank IC across its moments (index: date)."""
    ic, _ = moment_ic(frame, score, min_peers)
    return ic.groupby(level="date").mean() if len(ic) else pd.Series(dtype=float)


def paired_day_difference(a: pd.Series, b: pd.Series, n_boot: int = 2000, seed: int = 0, level: float = 0.99) -> dict:
    """Mean of (a - b) over the days both have, with a day-resampling interval."""
    both = a.index.intersection(b.index)
    if len(both) < 2:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "days": int(len(both))}
    d = (a.loc[both] - b.loc[both]).to_numpy()
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    tail = 100 * (1 - level) / 2
    return {"diff": float(d.mean()), "lo": float(np.percentile(boot, tail)), "hi": float(np.percentile(boot, 100 - tail)),
            "days": int(len(d))}


def verdict(cells: dict[int, dict], longer_minus_shorter: dict, early: dict, late: dict, t_bar: float = 3.0) -> dict:
    """The rule in configs/sip_e9_horizons.yaml, in order. `cells` maps horizon (bars) to an IC summary."""
    present = {h for h, c in cells.items() if c.get("t", np.nan) >= t_bar}
    if not present:
        return {"level": "no reversal at these horizons", "reason": f"no horizon reaches t >= {t_bar:g}"}
    lo, hi = longer_minus_shorter.get("lo", np.nan), longer_minus_shorter.get("hi", np.nan)
    if not (lo > 0 or hi < 0):
        return {"level": "the same strength at every horizon",
                "reason": "the longest horizon's IC is not distinguishable from the shortest's at 99%; the amount in "
                          "bps grows with the horizon only because moves are bigger"}
    sign = np.sign(longer_minus_shorter["diff"])
    if not (np.sign(early.get("diff", np.nan)) == np.sign(late.get("diff", np.nan)) == sign):
        return {"level": "not stable", "reason": "the difference does not have the same sign in both halves of the history"}
    return {"level": "stronger at longer horizons" if sign > 0 else "weaker at longer horizons",
            "reason": "the longest horizon's IC differs from the shortest's at 99%, with the same sign in both halves"}
