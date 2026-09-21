"""Scoring predictions, with uncertainty measured in days rather than trades.

Samples from the same day are not independent: they share the market's move,
and windows on the same stock overlap. So confidence intervals come from
resampling whole days with replacement (a day-block bootstrap). Treating
thousands of same-day trades as separate evidence would make every interval
several times too narrow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trades(pred: pd.DataFrame, threshold: float = 0.5, demean_by: str | None = None) -> pd.DataFrame:
    """Rows that become trades, with direction and gross P&L in bps.

    A probability of exactly 0.5 is no view and no trade. `threshold` above 0.5
    trades only the more confident predictions.

    `demean_by` subtracts each group's own average P&L, usually per symbol. The
    gate refuses whole ETFs whose prices move only a few cents a bar, so
    comparing approved days against the rest otherwise compares expensive ETFs
    against cheap ones. After demeaning, the comparison is within each ETF.
    """
    confidence = (pred["prob"] - 0.5).abs()
    taken = pred[(confidence > 0) & (confidence >= threshold - 0.5)].copy()
    taken["direction"] = np.where(taken["prob"] > 0.5, 1.0, -1.0)
    taken["gross_bps"] = taken["direction"] * taken["ret"] * 1e4
    taken["correct"] = (taken["direction"] > 0) == (taken["y"] == 1)
    if demean_by:
        taken["gross_bps"] -= taken.groupby(demean_by)["gross_bps"].transform("mean")
    return taken


def coverage_eligible(pred: pd.DataFrame, min_history: int = 200) -> np.ndarray:
    """Rows in folds that have enough earlier predictions to set a cutoff from."""
    folds = pred["fold"].to_numpy()
    eligible = np.zeros(len(pred), dtype=bool)
    for fold in np.unique(folds):
        if (folds < fold).sum() >= min_history:
            eligible |= folds == fold
    return eligible


def coverage_mask(pred: pd.DataFrame, coverage: float, min_history: int = 200) -> np.ndarray:
    """Keep the most confident `coverage` share of predictions.

    The cutoff for each fold comes from *earlier* folds only, the way it would
    have to be chosen live. Taking the cutoff from the whole test period would
    quietly use the future to decide which trades to keep.

    Early folds without enough history to set a cutoff are dropped at every
    coverage level, so the levels stay comparable; `coverage_eligible` reports
    which rows those are.

    A cut can only land where confidences differ. Rules whose probabilities are
    always 0 or 1 carry no ranking at all and keep every trade, and any model
    with tied confidences keeps more than the target share. The realized share
    is reported next to the target for that reason.
    """
    confidence = (pred["prob"] - 0.5).abs().to_numpy()
    folds = pred["fold"].to_numpy()
    mask = np.zeros(len(pred), dtype=bool)
    for fold in np.unique(folds):
        earlier = confidence[folds < fold]
        if earlier.size < min_history:
            continue
        cutoff = np.quantile(earlier, 1 - coverage)
        mask |= (folds == fold) & (confidence >= cutoff)
    return mask


def _day_totals(t: pd.DataFrame, days: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = t.groupby("date").agg(pnl=("gross_bps", "sum"), n=("gross_bps", "size"), hits=("correct", "sum"))
    grouped = grouped.reindex(days, fill_value=0)
    return grouped["pnl"].to_numpy(float), grouped["n"].to_numpy(float), grouped["hits"].to_numpy(float)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def summarize(
    pred: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 0,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Accuracy and gross bps per trade, each with a 95% day-bootstrap interval.

    Net return at a cost of c bps per round trip is gross minus c, so costs are
    applied when reporting rather than resampled.
    """
    t = trades(pred, threshold)
    days = np.unique(pred["date"])
    if t.empty:
        return {"n_trades": 0, "n_days": len(days), "coverage": 0.0}

    pnl, n, hits = _day_totals(t, days)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(days), (n_boot, len(days)))
    boot_n = n[idx].sum(axis=1)
    boot_gross = _ratio(pnl[idx].sum(axis=1), boot_n)
    boot_acc = _ratio(hits[idx].sum(axis=1), boot_n)

    return {
        "n_trades": int(len(t)),
        "n_days": int(len(days)),
        "coverage": float(len(t) / len(pred)),
        "accuracy": float(hits.sum() / n.sum()),
        "accuracy_lo": float(np.nanpercentile(boot_acc, 2.5)),
        "accuracy_hi": float(np.nanpercentile(boot_acc, 97.5)),
        "gross_bps": float(pnl.sum() / n.sum()),
        "gross_bps_lo": float(np.nanpercentile(boot_gross, 2.5)),
        "gross_bps_hi": float(np.nanpercentile(boot_gross, 97.5)),
        "share_days_positive": float((pnl[n > 0] > 0).mean()),
    }


def compare_subsets(
    pred: pd.DataFrame,
    mask: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    threshold: float = 0.5,
    demean_by: str | None = None,
) -> dict[str, float]:
    """Gross bps per trade inside `mask` vs outside it, and the difference, resampling days jointly.

    Used for Q1: the same predictions, split by whether the gate approved the
    day. Resampling the same days for both sides keeps the comparison paired.
    """
    pred = pred.reset_index(drop=True)
    mask = np.asarray(mask, dtype=bool)
    days = np.unique(pred["date"])
    everything = trades(pred, threshold, demean_by)
    inside = np.isin(everything.index.to_numpy(), np.flatnonzero(mask))
    t_in, t_out = everything[inside], everything[~inside]
    pnl_in, n_in, _ = _day_totals(t_in, days)
    pnl_out, n_out, _ = _day_totals(t_out, days)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(days), (n_boot, len(days)))
    boot_in = _ratio(pnl_in[idx].sum(axis=1), n_in[idx].sum(axis=1))
    boot_out = _ratio(pnl_out[idx].sum(axis=1), n_out[idx].sum(axis=1))
    diff = boot_in - boot_out

    def point(pnl, n):
        return float(pnl.sum() / n.sum()) if n.sum() else float("nan")

    return {
        "share_in": float(np.mean(mask)),
        "trades_in": int(n_in.sum()),
        "trades_out": int(n_out.sum()),
        "gross_bps_in": point(pnl_in, n_in),
        "gross_bps_out": point(pnl_out, n_out),
        "diff_bps": point(pnl_in, n_in) - point(pnl_out, n_out),
        "diff_bps_lo": float(np.nanpercentile(diff, 2.5)) if np.isfinite(diff).any() else float("nan"),
        "diff_bps_hi": float(np.nanpercentile(diff, 97.5)) if np.isfinite(diff).any() else float("nan"),
    }
