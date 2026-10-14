"""How long must the forward test run? (LIVE_PLAN.md, step 1)

A design calculation on the history, not a test: it fixes the forward test's length and geometry
before any forward day exists. Two answers, which should roughly agree:

- **Analytic.** The days needed to see the mean daily IC above zero (one-sided, `alpha`) with `power`,
  if the forward IC equals the historical one, and if it is only half as strong.
- **Historical windows.** Every stretch of `n` consecutive days in 2016–2026 treated as if it were the
  forward test: the share that would have passed (a) "IC above zero" and (b) the full registered rule,
  "IC above zero and at least half the historical IC". This keeps the history's quiet and noisy
  stretches, which the analytic answer averages away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def days_needed(ic: float, sd: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Trading days for a one-sided test of mean daily IC > 0 to reach `power` when the true mean is `ic`
    and days vary with standard deviation `sd`."""
    if not (ic > 0 and sd > 0):
        return np.inf
    return float(((norm.ppf(1 - alpha) + norm.ppf(power)) * sd / ic) ** 2)


def window_pass_rates(days: pd.Series, lengths: tuple[int, ...], alpha: float = 0.05, step: int = 5,
                      reference: float | None = None) -> pd.DataFrame:
    """For each length `n`, every `n`-day window of `days` (each day's IC, in date order), starting every
    `step` days: the share whose mean is above zero at one-sided `alpha` (`share_above`), and the share
    that also reaches half of `reference` (the full history's mean by default; `share_rule`)."""
    v = days.dropna().to_numpy(float)
    ref = float(v.mean()) if reference is None else reference
    z = norm.ppf(1 - alpha)
    rows = []
    for n in lengths:
        if n < 2 or n > len(v):
            rows.append({"days": n, "windows": 0, "share_above": np.nan, "share_rule": np.nan})
            continue
        starts = np.arange(0, len(v) - n + 1, step)
        w = np.stack([v[s: s + n] for s in starts])
        mean = w.mean(axis=1)
        t = mean / (w.std(axis=1, ddof=1) / np.sqrt(n))
        above = t >= z
        rows.append({"days": n, "windows": len(starts), "share_above": float(above.mean()),
                     "share_rule": float((above & (mean >= 0.5 * ref)).mean())})
    return pd.DataFrame(rows)


def combine(a: pd.Series, b: pd.Series) -> pd.Series:
    """The average of two universes' daily ICs over the days both have: each universe is ranked within
    itself, so a large cap is never ranked against a mid cap."""
    both = a.index.intersection(b.index)
    return (a.loc[both] + b.loc[both]) / 2
