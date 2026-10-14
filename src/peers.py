"""E12: does a stock follow the stocks it actually moves with?

E5 asked whether a stock catches up with its GICS sector and found that the sector adds nothing: the
signal worked no better with real sectors than with shuffled ones. A sector is a coarse, fixed label,
though. This tries the idea with peers taken from the data: each month, each stock's `k` stocks whose
5-minute returns (relative to the market) moved most closely with its own over the previous
`peer_days` trading days.

- **Peers are causal.** Picked on the first trading day of each month from the previous `peer_days`
  days only, and kept for that month.
- **Signal.** The peers' average move relative to the market over the last `lookback` bars.
- **The own reversal is in the way.** A stock moves with its peers, so their move carries part of its
  own move, and a stock's own move reverses (E5). Two defences:
  1. at each moment the peers' move is residualized on the stock's own move (ranks, least squares across
     stocks), so only the part of the peers' move the stock has not shared is scored; and
  2. the same is done with `k` random peers per stock, redrawn each month, on the same moments.
  Only real peers beating random ones, on the same moments, is following.
- **Stale prices.** As in E5, every bar from the start of the lookback to the signal bar, and the entry
  and exit bars, must have a trade.
- **Reported:** 10 peers instead of 5, a 1-hour hold, the sector average (E5's peers) through the same
  residual, and how often a correlated peer is in the stock's own sector.

`collect` runs a universe a day at a time. The rule is in configs/sip_e12_peers.yaml.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from .online import panel_days

MOMENT = ["date", "bar_index"]


def relative_returns(close: np.ndarray) -> np.ndarray:
    """Per-bar log returns minus the average across the stocks that traded in both bars (NaN where a
    stock lacks either bar). `close` is (bars, stocks)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(close), axis=0)
    finite = np.isfinite(r)
    count = finite.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        market = np.where(finite, r, 0.0).sum(axis=1, keepdims=True) / count
    return np.where(finite, r - market, np.nan)


def pick_peers(history: np.ndarray, k: int, min_share: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each stock's `k` most correlated other stocks over `history` (time, stocks), best first.

    Returns (peers, their correlations, eligible): peers is (stocks, k) with -1 where a stock is not
    eligible (finite in under `min_share` of the bars) or has too few eligible others."""
    n = history.shape[1]
    finite = np.isfinite(history)
    ok = finite.mean(axis=0) >= min_share
    x = np.where(finite, history, 0.0)
    x = x - x.mean(axis=0)
    sd = np.sqrt((x ** 2).sum(axis=0))
    z = np.divide(x, sd, out=np.zeros_like(x), where=sd > 0)
    corr = z.T @ z
    corr[:, ~ok] = -np.inf
    np.fill_diagonal(corr, -np.inf)
    peers = np.full((n, k), -1)
    rho = np.full((n, k), np.nan)
    if ok.sum() > k:
        idx = np.flatnonzero(ok)
        order = np.argsort(-corr[idx], axis=1)[:, :k]
        peers[idx] = order
        rho[idx] = np.take_along_axis(corr[idx], order, axis=1)
    return peers, rho, ok


def random_peers(ok: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """`k` distinct random other eligible stocks for each eligible stock (-1 elsewhere): the control."""
    n = len(ok)
    out = np.full((n, k), -1)
    pool = np.flatnonzero(ok)
    if len(pool) <= k:
        return out
    for i in pool:
        others = pool[pool != i]
        out[i] = rng.choice(others, size=k, replace=False)
    return out


def peer_mean(own: np.ndarray, valid: np.ndarray, peers: np.ndarray, min_valid: int) -> np.ndarray:
    """Each stock's average `own` over its valid peers; NaN with fewer than `min_valid` of them."""
    has = peers >= 0
    safe = np.where(has, peers, 0)
    use = has & valid[safe]
    count = use.sum(axis=1)
    total = np.where(use, own[safe], 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(count >= min_valid, total / count, np.nan)


def _sector_mean(own: np.ndarray, sector: np.ndarray, valid: np.ndarray, min_sector: int) -> np.ndarray:
    n_groups = int(sector.max()) + 1
    x = np.where(valid, own, 0.0)
    total = np.bincount(sector, weights=x, minlength=n_groups)
    count = np.bincount(sector, weights=valid.astype(float), minlength=n_groups)
    others = count[sector] - valid
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (total[sector] - x) / others
    return np.where(valid & (others >= min_sector), mean, np.nan)


def day_signals(close: np.ndarray, slots: np.ndarray, peers: np.ndarray, randoms: np.ndarray, sector: np.ndarray,
                ks: tuple[int, ...] = (5, 10), holds: tuple[int, ...] = (6, 12), lookback: int = 6, embargo: int = 1,
                stride: int = 6, min_valid: int = 3, min_sector: int = 5) -> dict[str, np.ndarray]:
    """Signals and forward returns for one day. `peers`/`randoms` are (stocks, max(ks)), best first.

    Columns: bar_index, stock, own (the stock's move relative to the market, reversed, as in E5),
    peers_k and random_k (the peers' move relative to the market) for each k, sector (the sector's
    average move relative to the market, itself excluded), and ret_h for each hold (NaN where the exit
    bar is missing or past the day's end). A moment needs only its lookback and entry bars."""
    cols = ["bar_index", "stock", "own", "sector"] + [f"{p}_{k}" for p in ("peers", "random") for k in ks] + \
        [f"ret_{h}" for h in holds]
    out = {c: [] for c in cols}
    finite = np.isfinite(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(close)
    n_bars = close.shape[0]
    for o in range(lookback, n_bars - embargo - min(holds), stride):
        entry = o + embargo
        valid = finite[o - lookback: o + 1].all(axis=0) & finite[entry] & (entry < slots)
        if valid.sum() < 2:
            continue
        move = np.where(valid, logc[o] - logc[o - lookback], 0.0)
        market = move[valid].mean()
        signals = {f"{p}_{k}": peer_mean(move, valid, table[:, :k], min(min_valid, k)) - market
                   for p, table in (("peers", peers), ("random", randoms)) for k in ks}
        idx = np.flatnonzero(valid & np.all([np.isfinite(v) for v in signals.values()], axis=0))
        if not len(idx):
            continue
        out["bar_index"].append(np.full(len(idx), o))
        out["stock"].append(idx)
        out["own"].append((market - move)[idx])
        out["sector"].append((_sector_mean(move, sector, valid, min_sector) - market)[idx])
        for name, v in signals.items():
            out[name].append(v[idx])
        for h in holds:
            exit_ = entry + h
            if exit_ >= n_bars:
                out[f"ret_{h}"].append(np.full(len(idx), np.nan))
                continue
            ok = finite[exit_] & (exit_ < slots)
            out[f"ret_{h}"].append(np.where(ok, logc[exit_] - logc[entry], np.nan)[idx])
    return {c: (np.concatenate(v) if v else np.empty(0)) for c, v in out.items()}


def collect(bars: dict[str, pd.DataFrame], sectors: dict[str, str], ks: tuple[int, ...] = (5, 10),
            holds: tuple[int, ...] = (6, 12), lookback: int = 6, embargo: int = 1, stride: int = 6,
            peer_days: int = 60, min_share: float = 0.8, min_valid: int = 3, min_sector: int = 5,
            seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every stock-moment of the universe, and one row per monthly peer refresh.

    Moments start once `peer_days` days of history exist. The refresh table records how many stocks
    were eligible, the average correlation of the chosen peers and the share of them in the stock's
    own sector (is this just the sector again?)."""
    symbols = np.array(list(bars))
    names = sorted({sectors.get(s, "unknown") for s in symbols})
    sector = np.array([names.index(sectors.get(s, "unknown")) for s in symbols])
    rng = np.random.default_rng(seed)
    k_max = max(ks)
    history: deque = deque(maxlen=peer_days)
    peers = randoms = None
    month = None
    parts, refreshes = [], []
    for date, fields, slots in panel_days(bars):
        this_month = np.datetime64(date, "M")
        if this_month != month and len(history) == peer_days:
            peers, rho, ok = pick_peers(np.concatenate(list(history)), k_max, min_share)
            randoms = random_peers(ok, k_max, rng)
            chosen = peers[:, :min(ks)]
            has = chosen >= 0
            same = (sector[np.where(has, chosen, 0)] == sector[:, None]) & has
            refreshes.append({"date": pd.Timestamp(date), "eligible": int(ok.sum()),
                              "peer_corr": float(np.nanmean(rho[:, :min(ks)])) if has.any() else np.nan,
                              "same_sector": float(same.sum() / has.sum()) if has.any() else np.nan})
            month = this_month
        if peers is not None:
            day = day_signals(fields["close"], slots, peers, randoms, sector, ks, holds, lookback, embargo, stride,
                              min_valid, min_sector)
            if len(day["stock"]):
                parts.append(pd.DataFrame({"date": np.datetime64(date, "ns"), "bar_index": day["bar_index"].astype(np.int16),
                                           "symbol": symbols[day["stock"].astype(int)],
                                           **{c: day[c].astype(np.float32) for c in day if c not in ("bar_index", "stock")}}))
        history.append(relative_returns(fields["close"]))
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return frame, pd.DataFrame(refreshes)


def residualize(frame: pd.DataFrame, col: str, on: str = "own") -> pd.Series:
    """At each moment, the rank of `col` with its least-squares fit on the rank of `on` removed.

    What is left is the part of the peers' move that is not a restatement of the stock's own move, so
    a stock's own reversal cannot show up through it (beyond what ranks miss)."""
    keys = [frame[k] for k in MOMENT]
    g = frame.groupby(keys)
    x, z = g[col].rank(), g[on].rank()
    xc = x - x.groupby(keys).transform("mean")
    zc = z - z.groupby(keys).transform("mean")
    num = (xc * zc).groupby(keys).transform("sum")
    den = (zc * zc).groupby(keys).transform("sum")
    beta = (num / den.where(den > 0)).fillna(0.0)
    return (xc - beta * zc).astype(np.float32)


def verdict(paired: dict, early: dict, late: dict) -> dict:
    """The rule in configs/sip_e12_peers.yaml: real peers' residual IC minus random peers', on the same
    moments (`paired`, 99%), then the same sign in both halves (`early`, `late`, 95%)."""
    lo, hi = paired.get("lo", np.nan), paired.get("hi", np.nan)
    if not (lo > 0 or hi < 0):
        return {"level": "correlated peers add nothing", "live": False,
                "reason": "real peers predict no better than random peers at 99%, on the same moments"}
    sign = np.sign(paired["diff"])
    if not (np.sign(early.get("diff", np.nan)) == np.sign(late.get("diff", np.nan)) == sign):
        return {"level": "not stable", "live": False,
                "reason": "real minus random peers does not have the same sign in both halves of the history"}
    if sign > 0:
        return {"level": "stocks follow their correlated peers", "live": True,
                "reason": "real peers beat random peers at 99%, with the same sign in both halves"}
    return {"level": "stocks move against their correlated peers", "live": False,
            "reason": "real peers predict worse than random ones at 99%, with the same sign in both halves: not following"}
