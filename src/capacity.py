"""E2: is the bottleneck the model or the signal?

Two experiments on the replication's setup (configs/ndx_replication.yaml: train on the 418 S&P 500 stocks
outside the Nasdaq-100, relative target, every stock scored out of time), both judged by rules fixed in
configs/e2_capacity.yaml before running.

**Learning curve and capacity ladder.** The same tree model trained on the most recent 126, 252 or 504
days of each fold's training window, and with trees of 4, 15 or 63 leaves. Every cell is scored on exactly
the same test rows, so cells can be compared moment by moment (`paired_ic_difference`). If more data or a
bigger model found more, some cell would beat the default (504 days, 15 leaves) clearly; if the inputs
hold no signal, every cell sits near zero.

**Planted edges.** The real returns, plus an edge of known size that is a function of the model's own
inputs: `momentum` (the window's return, which a linear model can find) or `interaction` (the window's
return only when volume is high, which needs a tree). Its size is set so that the edge itself would
score a chosen cross-sectional IC (the "oracle" IC), at and around the IC a book needs to pay. Then the
same pipeline is run as on the real data. If it recovers an edge of break-even size, the models and the
measurement are good enough, and a null on the real data is a missing signal, not a weak model. It
cannot speak for signals in information the model never sees.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .crosssection import MOMENT, moment_ic

SHAPES = ("momentum", "interaction")


# ------------------------------------------------------------------ cells

def cell_name(train_days: int, leaves: int) -> str:
    return f"d{train_days}_l{leaves}"


def trim_train_days(folds: list[dict], dates: np.ndarray, days: int) -> list[dict]:
    """Each fold's training rows restricted to its most recent `days` trading days. Validation and test
    rows are untouched, so every cell is scored on the same rows."""
    out = []
    for fold in folds:
        f = dict(fold)
        tr = fold["train"]
        kept = np.unique(dates[tr])[-days:]
        f["train"] = tr[np.isin(dates[tr], kept)]
        out.append(f)
    return out


def cell_config(cfg, leaves: int):
    """A copy of the config whose trees have `leaves` leaves."""
    c = copy.deepcopy(cfg)
    c.model.gbm_leaves = int(leaves)
    return c


def paired_ic_difference(a: pd.DataFrame, b: pd.DataFrame, min_peers: int = 20, n_boot: int = 2000,
                         seed: int = 0, level: float = 0.99) -> dict:
    """Mean IC of `a` minus that of `b` over the moments both scored, with a day-block interval.

    Both frames hold the same rows scored by two models, so the market's noise is shared and cancels:
    much tighter than comparing two separate intervals.
    """
    ia, _ = moment_ic(a, "prob", min_peers)
    ib, _ = moment_ic(b, "prob", min_peers)
    both = ia.index.intersection(ib.index)
    if not len(both):
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "days": 0}
    d = (ia.loc[both] - ib.loc[both]).groupby(level="date").mean().to_numpy()
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    tail = 100 * (1 - level) / 2
    return {"diff": float(d.mean()), "lo": float(np.percentile(boot, tail)),
            "hi": float(np.percentile(boot, 100 - tail)), "days": int(len(d))}


def capacity_verdict(summaries: dict[str, dict], paired: dict[str, dict], default: str, t_bar: float = 3.0) -> dict:
    """The E2a rule (configs/e2_capacity.yaml)."""
    better = [c for c, p in paired.items() if p["lo"] > 0 and summaries[c].get("t", np.nan) >= t_bar]
    any_t = [c for c, s in summaries.items() if s.get("t", np.nan) >= t_bar]
    if better:
        return {"level": "capacity or data helps", "cells": better,
                "reason": "a cell beats the default on the same rows (99% interval above zero) and reaches t >= 3"}
    if not any_t:
        return {"level": "signal-limited", "cells": [],
                "reason": "no amount of data or model size tried reaches t >= 3, and none beats the default"}
    return {"level": "mixed", "cells": any_t,
            "reason": "a cell reaches t >= 3 but none beats the default clearly on the same rows"}


# ------------------------------------------------------------------ planted edges

def zscore_moments(values: np.ndarray, dates: np.ndarray, bar_index: np.ndarray, clip: float = 4.0) -> np.ndarray:
    """Standardize across the stocks at each (date, bar), clipped and re-centred; constant moments give 0."""
    f = pd.DataFrame({"d": dates, "b": bar_index, "v": np.asarray(values, float)})
    g = f.groupby(["d", "b"])["v"]
    sd = g.transform("std").to_numpy()
    z = (f["v"].to_numpy() - g.transform("mean").to_numpy()) / np.where(sd > 0, sd, np.inf)
    z = pd.Series(np.clip(np.nan_to_num(z, nan=0.0), -clip, clip))
    return (z - z.groupby([f["d"], f["b"]]).transform("mean")).to_numpy()     # clipping can shift the mean


def planted_signal(data: dict, shape: str, channels: list[str]) -> np.ndarray:
    """A standardized edge built only from what the model sees at the signal bar (never the return).

    momentum: the window's own return. interaction: the window's return where the last bar's volume
    z-score is above 0.5, zero elsewhere -- a tree can find it, a linear model only partly.
    """
    if shape not in SHAPES:
        raise ValueError(f"unknown shape '{shape}'; options: {SHAPES}")
    d, b = data["date"], data["bar_index"]
    w = np.nan_to_num(data["window_return"].astype(float), nan=0.0)   # NaN only for a symbol's very first window
    momentum = zscore_moments(w, d, b)
    if shape == "momentum":
        return momentum
    if "volume_z" not in channels:
        raise ValueError("the interaction edge needs the volume_z channel")
    vz = data["X"][:, channels.index("volume_z"), -1].astype(float)
    return zscore_moments(momentum * (vz > 0.5), d, b)


def mean_rank_ic(s: np.ndarray, ret: np.ndarray, dates, bar_index, symbols, min_peers: int = 20) -> float:
    frame = pd.DataFrame({"date": dates, "bar_index": bar_index, "symbol": symbols, "prob": s, "ret": ret})
    ic, _ = moment_ic(frame, "prob", min_peers)
    return float(ic.mean()) if len(ic) else np.nan


def calibrate_edge(data: dict, s: np.ndarray, target_ic: float, rows: np.ndarray | None = None,
                   min_peers: int = 20, tol: float = 0.03, max_moments: int = 2000, seed: int = 0) -> float:
    """The edge size in bps per standard deviation of `s` at which `s` scores `target_ic` against the
    returns plus the edge. Bisection on a sample of moments; returns lambda in bps."""
    if target_ic <= 0:
        return 0.0
    idx = np.arange(len(s)) if rows is None else np.asarray(rows)
    moments = pd.DataFrame({"d": data["date"][idx], "b": data["bar_index"][idx]})
    keys = moments.drop_duplicates()
    if len(keys) > max_moments:
        keys = keys.sample(max_moments, random_state=seed)
    pick = idx[moments.merge(keys.assign(_k=1), on=["d", "b"], how="left")["_k"].notna().to_numpy()]
    args = (data["date"][pick], data["bar_index"][pick], data["symbol"][pick])
    base, sp = data["ret"][pick].astype(float), s[pick]

    def ic_at(lam):
        return mean_rank_ic(sp, base + lam * 1e-4 * sp, *args, min_peers=min_peers)

    lo, hi = 0.0, 1.0
    while ic_at(hi) < target_ic and hi < 1e4:
        lo, hi = hi, hi * 2
    for _ in range(40):
        mid = (lo + hi) / 2
        value = ic_at(mid)
        if abs(value - target_ic) <= tol * target_ic:
            return mid
        lo, hi = (mid, hi) if value < target_ic else (lo, mid)
    return (lo + hi) / 2


def plant(data: dict, s: np.ndarray, lam_bps: float) -> dict:
    """A shallow copy of `data` whose forward return carries the edge; features are untouched."""
    out = dict(data)
    out["ret"] = data["ret"] + lam_bps * 1e-4 * s
    out["y"] = (out["ret"] > 0).astype(np.int64)
    return out


def planted_verdict(rows: list[dict], break_even: float, t_bar: float = 3.0, share: float = 0.5) -> dict:
    """The E2b rule: per shape, is an edge of break-even size recovered (t >= 3 and at least half of the
    oracle IC)? Also the smallest planted size recovered."""
    out = {}
    for shape in sorted({r["shape"] for r in rows}):
        rs = sorted((r for r in rows if r["shape"] == shape), key=lambda r: r["target"])
        found = [r for r in rs if r.get("t", np.nan) >= t_bar and r.get("ic", np.nan) >= share * r.get("oracle_ic", np.inf)]
        at = [r for r in rs if abs(r["target"] - break_even) < 1e-9]
        out[shape] = {"adequate": bool(at and at[0] in found),
                      "smallest_recovered": found[0]["target"] if found else None}
    return out
