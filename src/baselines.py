"""Rules that need no training. Any model worth its parameters has to beat these.

- always_up: go long every sample. Measures the market's drift over the
  holding period, which a classifier can match just by predicting "up".
- momentum_day: follow the sign of the return since the previous close. In the
  first-half-hour setup this is the classic intraday momentum rule.
- momentum_window: follow the sign of the feature window's cumulative return.

Rules output a probability of "up" so they flow through the same evaluation as
the models. A signal of exactly zero, or a missing one, gives 0.5 and no trade.
"""

from __future__ import annotations

import numpy as np

RULES = ("always_up", "momentum_day", "momentum_window")


def is_rule(arch: str) -> bool:
    return arch in RULES


def rule_probability(name: str, data: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    if name == "always_up":
        return np.ones(len(idx))
    if name == "momentum_day":
        signal = data["day_return"][idx]
    elif name == "momentum_window":
        signal = data["window_return"][idx]
    else:
        raise ValueError(f"unknown rule '{name}'; options: {RULES}")
    return np.where(signal > 0, 1.0, np.where(signal < 0, 0.0, 0.5))
