"""E6: does permutation entropy tell when the short-term reversal is stronger?

E5 found the project's first real pattern: a stock that lagged the market over the last 30 minutes
recovers a little over the next 30 (IC +0.0064, t 7.5, ten years of SIP bars). The entropy gate of
this thesis was built to say *when* a pattern can be trusted, and until now had no pattern to say it
about. This test gives it one.

- **Signal.** The stock's own move over the last `lookback` bars, relative to the market, reversed.
  Entry one bar later, `hold` bars; exactly E5's "own move reversed".
- **State, known at the signal bar.** From the stock's last 60 five-minute returns (reaching into the
  previous session, as the pipeline's intraday entropy features do): permutation entropy at m = 3
  (the registered state), the share of monotone patterns, the lag-1 autocorrelation, and the
  volatility. Also the size of the move itself.
- **Question.** At each moment the stocks are split into thirds by the state. Is the reversal's IC
  different in the top third than in the bottom third? The comparison is within a moment, so the
  time of day and the market's mood cancel.
- **The control that matters: volatility.** Entropy moves with volatility, and volatility alone might
  set how strong a reversal is. The same comparison is repeated inside each third of volatility. If
  entropy only mattered through volatility, the matched difference vanishes.
- Costs play no part in the verdict (decided 2026-10-13); the question is whether structure exists.

The rule is in configs/sip_e6_reversal.yaml.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .crosssection import MOMENT, moment_ic
from .entropy import _row_entropy, batched_pattern_distributions, monotone_codes
from .entropy_features import _last_valid
from .online import panel_days

STATES = ["pe3", "trend3", "ac1", "vol", "size"]
TAIL = 78          # bars of the previous session kept, so a window can reach back across the open


def day_states(logc_ext: np.ndarray, slots: np.ndarray, lookback: int = 6, hold: int = 6, embargo: int = 1,
               stride: int = 6, window: int = 60) -> dict[str, np.ndarray]:
    """One day's signals, returns and states. `logc_ext` is (TAIL + bars, stocks): the previous
    session's last TAIL log closes (NaN if none) followed by today's."""
    logc = logc_ext[TAIL:]
    finite = np.isfinite(logc)
    r_ext = np.diff(logc_ext, axis=0)                               # row k: bar k+1 minus bar k (extended index)
    up, down = monotone_codes(3)
    out = {k: [] for k in ["bar_index", "stock", "ret", "signal"] + STATES}
    for o in range(lookback, logc.shape[0] - embargo - hold, stride):
        entry, exit_ = o + embargo, o + embargo + hold
        valid = finite[o - lookback: o + 1].all(axis=0) & finite[entry] & finite[exit_] & (exit_ < slots)
        idx = np.flatnonzero(valid)
        if len(idx) < 2:
            continue
        own = logc[o, idx] - logc[o - lookback, idx]
        rel = own - own.mean()
        trailing = r_ext[o: o + TAIL, idx].T                        # the 78 one-bar returns ending at bar o
        compact, enough = _last_valid(trailing, window)
        states = {k: np.full(len(idx), np.nan) for k in STATES}
        states["size"] = np.abs(rel)
        if enough.any():
            p3, _ = batched_pattern_distributions(compact, m=3)
            z = compact - compact.mean(axis=1, keepdims=True)
            var = (z ** 2).sum(axis=1)
            states["pe3"][enough] = _row_entropy(p3, 3)
            states["trend3"][enough] = p3[:, up] + p3[:, down]
            states["ac1"][enough] = np.divide((z[:, 1:] * z[:, :-1]).sum(axis=1), var, out=np.full(len(var), np.nan), where=var > 0)
            states["vol"][enough] = compact.std(axis=1)
        out["bar_index"].append(np.full(len(idx), o))
        out["stock"].append(idx)
        out["ret"].append(logc[exit_, idx] - logc[entry, idx])
        out["signal"].append(-rel)
        for k in STATES:
            out[k].append(states[k])
    return {k: (np.concatenate(v) if v else np.empty(0)) for k, v in out.items()}


def collect_states(bars: dict[str, pd.DataFrame], lookback: int = 6, hold: int = 6, embargo: int = 1,
                   stride: int = 6, window: int = 60) -> pd.DataFrame:
    """Every stock-moment: date, bar_index, symbol, ret, signal and the states (float32)."""
    symbols = np.array(list(bars))
    tail = np.full((TAIL, len(symbols)), np.nan)
    parts = []
    for date, fields, slots in panel_days(bars):
        with np.errstate(divide="ignore", invalid="ignore"):
            logc = np.log(fields["close"])
        day = day_states(np.vstack([tail, logc]), slots, lookback, hold, embargo, stride, window)
        if len(day["stock"]):
            parts.append(pd.DataFrame({"date": np.datetime64(date, "ns"), "bar_index": day["bar_index"].astype(np.int16),
                                       "symbol": symbols[day["stock"].astype(int)],
                                       **{k: day[k].astype(np.float32) for k in ["ret", "signal"] + STATES}}))
        had = slots > 0                                             # a stock absent today keeps its older tail
        padded = np.vstack([tail, logc])[-TAIL:]
        tail = np.where(had[None, :], padded, tail)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def conditional_ic(frame: pd.DataFrame, state: str, control: str | None = None, groups: int = 3,
                   min_group: int = 30, n_boot: int = 2000, seed: int = 0, level: float = 0.99) -> dict:
    """The reversal's IC in the bottom and top third of `state`, and their difference.

    Thirds are cut within each moment (inside each third of `control` when given), so the
    comparison never mixes moments. ICs are averaged within days; the difference (top minus
    bottom) gets a day-block interval at `level`.
    """
    need = [state] + ([control] if control else [])
    f = frame.dropna(subset=need + ["signal", "ret"])
    if f.empty:
        return {"days": 0}
    f = f[MOMENT + ["symbol", "signal", "ret"] + need].copy()
    keys = list(MOMENT)
    if control:
        n = f.groupby(keys)[control].transform("size")
        f["_c"] = np.minimum((groups * (f.groupby(keys)[control].rank(method="first") - 0.5) / n).astype(int), groups - 1)
        keys = keys + ["_c"]
    else:
        f["_c"] = 0
    n = f.groupby(keys)[state].transform("size")
    f["_g"] = np.minimum((groups * (f.groupby(keys)[state].rank(method="first") - 0.5) / n).astype(int), groups - 1)
    f["_bar"] = f["bar_index"].astype(np.int64)
    f["bar_index"] = f["_bar"] * 100 + f["_c"] * 10 + f["_g"]       # one "moment" per (moment, control third, state third)
    ic, _ = moment_ic(f.rename(columns={"signal": "prob"}), "prob", min_peers=min_group)
    if ic.empty:
        return {"days": 0}
    table = ic.rename("ic").reset_index()
    table["g"] = table["bar_index"] % 10
    by_day = table.groupby(["date", "g"])["ic"].mean().unstack("g")
    both = by_day[[0, groups - 1]].dropna()
    d = (both[groups - 1] - both[0]).to_numpy()
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    tail = 100 * (1 - level) / 2
    sd = d.std(ddof=1) if len(d) > 1 else np.nan
    return {"low": float(both[0].mean()), "high": float(both[groups - 1].mean()),
            "mid": float(by_day[1].mean()) if groups == 3 and 1 in by_day else np.nan,
            "diff": float(d.mean()), "lo": float(np.percentile(boot, tail)), "hi": float(np.percentile(boot, 100 - tail)),
            "t": float(d.mean() / (sd / np.sqrt(len(d)))) if sd and sd > 0 else np.nan, "days": int(len(d))}


def verdict(single: dict, matched: dict, early: dict, late: dict) -> dict:
    """The rule in configs/sip_e6_reversal.yaml, in order."""
    def excludes_zero(r):
        return r.get("lo", np.nan) > 0 or r.get("hi", np.nan) < 0

    if not excludes_zero(single):
        return {"level": "entropy does not condition the reversal",
                "reason": "the reversal is as strong among the most ordered stocks as among the least"}
    if not excludes_zero(matched) or np.sign(matched["diff"]) != np.sign(single["diff"]):
        return {"level": "volatility, not entropy",
                "reason": "the difference disappears among stocks of similar volatility"}
    if not (np.sign(early.get("diff", np.nan)) == np.sign(late.get("diff", np.nan)) == np.sign(single["diff"])):
        return {"level": "not stable", "reason": "the difference does not have the same sign in both halves of the history"}
    where = "high" if single["diff"] > 0 else "low"
    return {"level": f"entropy conditions the reversal (stronger when entropy is {where})",
            "reason": "the difference holds at 99%, survives matching on volatility, and has the same sign in both halves"}


# ------------------------------------------------------------------ E7: the reversal as a slope

def slope_by_group(frame: pd.DataFrame, by: str | None = "size", groups: int = 3, min_peers: int = 20,
                   n_boot: int = 2000, seed: int = 0, level: float = 0.99) -> dict:
    """The share of a stock's lag that it recovers, in each group of `by` (thirds by default).

    E6 measured the reversal as a correlation, and a correlation is larger wherever the signal
    varies more against the same noise: a perfectly uniform reversal shows a higher IC among large
    moves. A slope does not have that problem. Here the slope is through the origin, of the forward
    return relative to the market on the signal (the reversed relative move): bps recovered per bps
    lagged. Sums are pooled over a day and days are resampled together for every group, so the
    difference between the top and the bottom group is paired.

    Returns the slope per group with its interval, and `diff` (top minus bottom) with its interval,
    standard error and `detectable` (the difference this data would see at this level, 80% of the time).
    """
    cols = MOMENT + ["signal", "ret"] + ([by] if by else [])
    f = frame.dropna(subset=cols)[cols].copy()
    g = f.groupby(MOMENT)["ret"]
    f = f[g.transform("size") >= min_peers]
    if f.empty:
        return {"days": 0}
    f["rel"] = f["ret"] - f.groupby(MOMENT)["ret"].transform("mean")
    if by:
        n = f.groupby(MOMENT)[by].transform("size")
        f["_g"] = np.minimum((groups * (f.groupby(MOMENT)[by].rank(method="first") - 0.5) / n).astype(int), groups - 1)
    else:
        groups, f["_g"] = 1, 0
    f["sxy"], f["sxx"] = f["signal"].astype(float) * f["rel"], f["signal"].astype(float) ** 2
    day = f.groupby(["date", "_g"])[["sxy", "sxx"]].sum().unstack("_g").fillna(0.0)
    sxy, sxx = day["sxy"].to_numpy(), day["sxx"].to_numpy()                   # (days, groups)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(day), (n_boot, len(day)))
    boot = sxy[idx].sum(axis=1) / sxx[idx].sum(axis=1)                        # (n_boot, groups)
    tail = 100 * (1 - level) / 2
    slopes = sxy.sum(axis=0) / sxx.sum(axis=0)
    out = {"slope": slopes.tolist(), "lo": np.percentile(boot, tail, axis=0).tolist(),
           "hi": np.percentile(boot, 100 - tail, axis=0).tolist(), "days": int(len(day)), "rows": int(len(f)),
           "typical_move_bps": (1e4 * np.sqrt(sxx.sum(axis=0) / f.groupby("_g").size().to_numpy())).tolist()}
    if groups > 1:
        d = boot[:, -1] - boot[:, 0]
        se = float(d.std())
        z = {0.99: 2.576, 0.95: 1.96}.get(level, 2.576)
        out.update(diff=float(slopes[-1] - slopes[0]), diff_lo=float(np.percentile(d, tail)),
                   diff_hi=float(np.percentile(d, 100 - tail)), diff_se=se, detectable=(z + 0.84) * se)
    return out


def slope_verdict(overall: dict, by_size: dict, early: dict, late: dict) -> dict:
    """The rule in configs/sip_e7_slope.yaml, in order."""
    if not overall.get("lo", [np.nan])[0] > 0:
        return {"level": "no reversal as a slope", "reason": "the share recovered is not above zero at 99%"}
    if not (by_size.get("diff_lo", np.nan) > 0 or by_size.get("diff_hi", np.nan) < 0):
        return {"level": "a uniform reversal",
                "reason": "stocks recover the same share of a large lag as of a small one, as far as this data can tell"}
    if not (np.sign(early.get("diff", np.nan)) == np.sign(late.get("diff", np.nan)) == np.sign(by_size["diff"])):
        return {"level": "not stable", "reason": "the difference does not have the same sign in both halves of the history"}
    which = "larger" if by_size["diff"] > 0 else "smaller"
    return {"level": f"not uniform: a {which} share is recovered after large moves",
            "reason": "the share recovered differs between the largest and smallest moves at 99%, with the same sign in both halves"}
