"""E5: when a stock's sector has moved and the stock has not, does it catch up?

The lead-lag idea: news reaches some stocks first and their sector peers follow a little later (Lo and
MacKinlay 1990; Hou 2007 within industries). Every model before this saw only a stock's own window, so
this is information they never had.

- **Signal (the gap).** Over the last `lookback` bars: the average return of the stock's sector peers
  (itself excluded) minus the stock's own return. Positive means the peers rose and it lagged.
- **Trade.** Enter one bar after the signal bar, hold `hold` bars. Scored across stocks at each moment
  (rank IC, then a long-short quintile book), so the market's move cancels.
- **Stale prices.** A stock that has not traded shows an old price and seems to "follow" later. Every bar
  from the start of the lookback to the signal bar, and the entry and exit bars, must have a trade.
- **The control that matters: shuffled sectors.** The gap contains the stock's own return with a minus
  sign, and short-term reversal of a stock's own return (bid-ask bounce) would make any such signal
  "work". The same gap computed with sector labels shuffled between stocks keeps that reversal and
  removes the sector. Only the true gap beating the shuffled one, on the same moments, is lead-lag.
- **Leaders (reported).** The gap measured against the sector's three most traded stocks (by dollar
  volume over the previous 20 days) instead of all peers.
- No pair search: sectors come from the universe file, leaders from past volume.

`day_signals` works on one day's bars; `collect` runs it over a universe a day at a time.
The rule is in configs/sip_e5_leadlag.yaml.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from .online import panel_days

SIGNALS = ["gap", "gap_shuffled", "gap_leaders", "peers", "own"]


def _peer_mean(own: np.ndarray, groups: np.ndarray, valid: np.ndarray, n_groups: int, min_group: int,
               weight: np.ndarray | None = None) -> np.ndarray:
    """Each stock's group average of `own` excluding itself, over valid stocks (`weight` marks who counts
    as a peer: all valid stocks by default, or only the leaders). NaN where fewer than `min_group` others."""
    w = valid if weight is None else (valid & weight)
    x = np.where(w, own, 0.0)
    total = np.bincount(groups, weights=x, minlength=n_groups)
    count = np.bincount(groups, weights=w.astype(float), minlength=n_groups)
    others = count[groups] - w
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (total[groups] - x) / others
    return np.where(valid & (others >= min_group), mean, np.nan)


def day_signals(close: np.ndarray, slots: np.ndarray, sector: np.ndarray, shuffled: np.ndarray,
                leaders: np.ndarray, lookback: int = 6, hold: int = 6, embargo: int = 1, stride: int = 6,
                min_sector: int = 5, min_leaders: int = 2) -> dict[str, np.ndarray]:
    """Signals and forward returns for one day. `close` is (bars, stocks) with NaN where no trade;
    `sector` and `shuffled` are integer labels; `leaders` marks each sector's most traded stocks."""
    n_groups = int(max(sector.max(), shuffled.max())) + 1
    out = {k: [] for k in ["bar_index", "stock", "ret"] + SIGNALS}
    finite = np.isfinite(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(close)
    for o in range(lookback, close.shape[0] - embargo - hold, stride):
        entry, exit_ = o + embargo, o + embargo + hold
        valid = finite[o - lookback: o + 1].all(axis=0) & finite[entry] & finite[exit_] & (exit_ < slots)
        if valid.sum() < 2:
            continue
        own = np.where(valid, logc[o] - logc[o - lookback], 0.0)
        market = own[valid].mean()
        peers = _peer_mean(own, sector, valid, n_groups, min_sector)
        idx = np.flatnonzero(valid & np.isfinite(peers))
        if not len(idx):
            continue
        out["bar_index"].append(np.full(len(idx), o))
        out["stock"].append(idx)
        out["ret"].append((logc[exit_] - logc[entry])[idx])
        out["gap"].append((peers - own)[idx])
        out["gap_shuffled"].append((_peer_mean(own, shuffled, valid, n_groups, min_sector) - own)[idx])
        out["gap_leaders"].append((_peer_mean(own, sector, valid, n_groups, min_leaders, leaders) - own)[idx])
        out["peers"].append((peers - market)[idx])
        out["own"].append((market - own)[idx])
    return {k: (np.concatenate(v) if v else np.empty(0)) for k, v in out.items()}


def shuffle_labels(sector: np.ndarray, seed: int = 0) -> np.ndarray:
    """The same labels dealt to different stocks: group sizes kept, membership broken."""
    return np.random.default_rng(seed).permutation(sector)


def pick_leaders(dollar: np.ndarray, sector: np.ndarray, n: int = 3) -> np.ndarray:
    """Mark each sector's `n` stocks with the most traded dollars (NaN = no history: never a leader)."""
    mark = np.zeros(len(sector), dtype=bool)
    for g in np.unique(sector):
        members = np.flatnonzero((sector == g) & np.isfinite(dollar))
        mark[members[np.argsort(-dollar[members])[:n]]] = True
    return mark


def collect(bars: dict[str, pd.DataFrame], sectors: dict[str, str], lookback: int = 6, hold: int = 6,
            embargo: int = 1, stride: int = 6, min_sector: int = 5, n_leaders: int = 3,
            leader_days: int = 20, seed: int = 0) -> pd.DataFrame:
    """Every stock-moment of the universe: date, bar_index, symbol, ret and the signals (float32).

    Leaders are chosen each day from dollar volume over the previous `leader_days` days only."""
    symbols = np.array(list(bars))
    names = sorted({sectors.get(s, "unknown") for s in symbols})
    sector = np.array([names.index(sectors.get(s, "unknown")) for s in symbols])
    shuffled = shuffle_labels(sector, seed)
    history: deque = deque(maxlen=leader_days)
    parts = []
    for date, fields, slots in panel_days(bars):
        past = np.nansum(np.stack(history), axis=0) if len(history) == leader_days else np.full(len(symbols), np.nan)
        leaders = pick_leaders(np.where(past > 0, past, np.nan), sector, n_leaders)
        day = day_signals(fields["close"], slots, sector, shuffled, leaders, lookback, hold, embargo, stride, min_sector)
        if len(day["stock"]):
            frame = pd.DataFrame({"date": np.datetime64(date, "ns"), "bar_index": day["bar_index"].astype(np.int16),
                                  "symbol": symbols[day["stock"].astype(int)],
                                  **{k: day[k].astype(np.float32) for k in ["ret"] + SIGNALS}})
            parts.append(frame)
        history.append(np.nansum(fields["close"] * fields["volume"], axis=0))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def verdict(gap: dict, paired: dict, early: dict, late: dict, book: dict, cost_bps: float, t_bar: float = 3.0) -> dict:
    """The rule in configs/sip_e5_leadlag.yaml, in order."""
    if not gap.get("t", np.nan) >= t_bar:
        return {"level": "no catch-up effect", "reason": f"the gap's IC does not reach t >= {t_bar:g}"}
    if not paired.get("lo", np.nan) > 0:
        return {"level": "own reversal, not lead-lag",
                "reason": "the gap predicts no better with real sectors than with shuffled ones, so it is the "
                          "stock's own return reversing, not its sector leading"}
    if not (early.get("ic", np.nan) > 0 and late.get("ic", np.nan) > 0):
        return {"level": "not stable", "reason": "the gap's IC is not positive in both halves of the history"}
    if book.get("gross_lo", np.nan) - cost_bps > 0:
        return {"level": "catches up and pays", "reason": "the long-short book's net interval is above zero"}
    return {"level": "catches up, does not pay", "reason": "stocks do follow their sector, by less than the cost of trading it"}
