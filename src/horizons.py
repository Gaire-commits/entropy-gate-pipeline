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


# ------------------------------------------------------------------ E10: the same signals, held for longer

from .online import panel_days  # noqa: E402

OFFSETS = (6, 12, 18, 24)      # signal bars: 10:00, 10:30, 11:00, 11:30; all leave room for a 4-hour hold


def collect_overnight(bars: dict[str, pd.DataFrame], lookback: int = 6, embargo: int = 1,
                      offsets: tuple[int, ...] = OFFSETS, min_peers: int = 20) -> pd.DataFrame:
    """The same signal as E5 (own move relative to the market over `lookback` bars, reversed), held for a
    full trading day: entry `embargo` bar after the signal, exit at the same bar of the next trading day.

    Rows: date (of the signal), bar_index (the signal bar), symbol, ret (log, entry to exit), own (the
    signal). A stock must have a trade at every bar of its lookback window, at the entry bar, and at the
    exit bar; the next trading day is the next day the stock itself has bars. The signal uses nothing
    later than its own bar, and the exit is known only on the next day.
    """
    symbols = np.array(list(bars))
    pending: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.datetime64]] = {}
    parts = []
    for date, fields, slots in panel_days(bars):
        with np.errstate(divide="ignore", invalid="ignore"):
            logc = np.log(fields["close"])
        for o, (idx, signal, entry, signal_date) in list(pending.items()):
            exit_ = o + embargo
            if exit_ >= logc.shape[0]:
                continue
            here = np.isfinite(logc[exit_, idx]) & (exit_ < slots[idx])
            if here.any():
                parts.append(pd.DataFrame({"date": np.datetime64(signal_date, "ns"), "bar_index": np.int16(o),
                                           "symbol": symbols[idx[here]], "own": signal[here].astype(np.float32),
                                           "ret": (logc[exit_, idx[here]] - entry[here]).astype(np.float32)}))
        pending = {}
        finite = np.isfinite(logc)
        for o in offsets:
            entry_bar = o + embargo
            if entry_bar >= logc.shape[0]:
                continue
            valid = finite[o - lookback: o + 1].all(axis=0) & finite[entry_bar] & (entry_bar < slots)
            idx = np.flatnonzero(valid)
            if len(idx) < min_peers:
                continue
            move = logc[o, idx] - logc[o - lookback, idx]
            pending[o] = (idx, -(move - move.mean()), logc[entry_bar, idx], date)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def growth_verdict(base: dict, longest: dict, paired: dict, early: dict, late: dict, t_bar: float = 3.0) -> dict:
    """The rule in configs/sip_e10_holds.yaml for one comparison: the IC at a long hold against the
    shortest, on the same moments (`paired`: 99%; `early`, `late`: 95%)."""
    if not base.get("t", np.nan) >= t_bar:
        return {"level": "no reversal at these signal times", "reason": f"the shortest hold does not reach t >= {t_bar:g}"}
    lo, hi = paired.get("lo", np.nan), paired.get("hi", np.nan)
    if not (lo > 0 or hi < 0):
        return {"level": "does not keep growing",
                "reason": "the longer hold's IC is not distinguishable from the shortest hold's at 99%, on the same moments"}
    sign = np.sign(paired["diff"])
    if not (np.sign(early.get("diff", np.nan)) == np.sign(late.get("diff", np.nan)) == sign):
        return {"level": "not stable", "reason": "the difference does not have the same sign in both halves of the history"}
    return {"level": "keeps growing with the hold" if sign > 0 else "fades with the hold",
            "reason": "the longer hold's IC differs from the shortest hold's at 99%, with the same sign in both halves"}
