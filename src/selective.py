"""The selective strategy: trade only a model's strongest signals, across a wide universe so they
still come often. The project's main case from 2026-10-03.

For each model and each kept share of signals (the cutoff for a quarter comes from earlier
quarters only), it reports how often it trades, how good each trade is, what it earns per day,
whether the result holds quarter after quarter, and whether it is skill or volatility.

The volatility question matters because the one lead so far (S&P 500 logistic regression, +6.15
bps gross on its most confident 10%) came with accuracy that barely moved (0.503 to 0.512). If a
model is most confident when the market is most volatile, its confident trades earn more bps
simply because every move is bigger. Two checks separate that from skill:

- risk-scaled return: each trade's return divided by the trailing volatility of its own symbol
  (known before the trade) times the square root of the holding period, so a "big" move means big
  for that moment. Skill raises it as the filter tightens; volatility does not.
- a placebo: keep the same share of signals ranked by trailing volatility instead of confidence,
  and trade the model's direction. Compare the two on risk-scaled return, not bps: a model with a
  little skill everywhere makes the most bps where moves are biggest, so in bps the placebo can win
  even when confidence is real skill. Confidence is skill when its trades earn more per unit of
  their own risk than the placebo's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import coverage_eligible, summarize, trades

SHARES = (1.0, 0.2, 0.1, 0.05, 0.02)


def top_share_mask(score: np.ndarray, folds: np.ndarray, share: float, min_history: int = 200) -> np.ndarray:
    """Rows whose score clears the cutoff that keeps the top `share`, set from earlier folds only.

    Missing scores are never selected. Folds without `min_history` earlier scores keep nothing.
    """
    score = np.asarray(score, dtype=float)
    mask = np.zeros(len(score), dtype=bool)
    for fold in np.unique(folds):
        earlier = score[folds < fold]
        earlier = earlier[np.isfinite(earlier)]
        if earlier.size < min_history:
            continue
        this = folds == fold
        if share >= 1.0:
            mask |= this
        else:
            mask |= this & (score >= np.quantile(earlier, 1 - share))
    return mask


def trailing_volatility(bars: dict[str, pd.DataFrame], keys: pd.DataFrame,
                        lookback: int = 78, min_periods: int = 60) -> pd.Series:
    """Standard deviation of the last `lookback` five-minute log returns at each (symbol, date,
    bar_index) in `keys`, the signal bar included and nothing after it. Aligned to `keys`' rows.

    The series runs across sessions, so early in a day it includes the overnight move.
    """
    parts = []
    for symbol, wanted in keys.groupby("symbol", sort=False):
        if symbol not in bars:
            continue
        df = bars[symbol]
        r = np.log(df["close"] / df["close"].shift(1))
        ts = df.index.get_level_values("timestamp")
        table = pd.DataFrame({
            "date": ts.normalize().tz_localize(None),
            "bar_index": df["bar_index"].to_numpy(),
            "vol": r.rolling(lookback, min_periods=min_periods).std().to_numpy(),
        }).drop_duplicates(["date", "bar_index"])
        found = wanted.reset_index().merge(table, on=["date", "bar_index"], how="left")
        parts.append(pd.Series(found["vol"].to_numpy(), index=found["index"].to_numpy()))
    if not parts:
        return pd.Series(np.nan, index=keys.index)
    return pd.concat(parts).reindex(keys.index)


def _score(p: pd.DataFrame, mask: np.ndarray, days: np.ndarray, cost: float, horizon: int,
           n_boot: int, seed: int) -> dict:
    chosen = p.assign(prob=np.where(mask, p["prob"], 0.5))
    s = summarize(chosen, n_boot=n_boot, seed=seed)
    t = trades(chosen)
    out = {"trades": s["n_trades"], "trades_per_day": s["n_trades"] / len(days)}
    if t.empty:
        return out
    net = t["gross_bps"] - cost
    by_fold = net.groupby(t["fold"]).mean()
    out.update(
        days_traded=t["date"].nunique() / len(days),
        long_share=float((t["direction"] > 0).mean()),
        hit_rate=s["accuracy"],
        gross=s["gross_bps"], gross_lo=s["gross_bps_lo"], gross_hi=s["gross_bps_hi"],
        net=s["gross_bps"] - cost,
        net_per_day=float(net.groupby(t["date"]).sum().sum() / len(days)),
        folds_positive=int((by_fold > 0).sum()), folds=int(p["fold"].nunique()),
    )
    if "vol" in t.columns and t["vol"].notna().any():
        # Each trade's return in units of its own expected move (trailing volatility x sqrt of the
        # hold), x100. Scored through the same day-block bootstrap by rescaling `ret`, so trades
        # without a volatility reading drop out instead of counting as zero.
        scale = p["vol"].to_numpy(float) * np.sqrt(horizon)
        usable = mask & np.isfinite(scale) & (scale > 0)
        scaled = chosen.assign(prob=np.where(usable, chosen["prob"], 0.5),
                               ret=np.where(usable, p["ret"] / np.where(usable, scale, 1.0), 0.0) / 100)
        r = summarize(scaled, n_boot=n_boot, seed=seed)
        out.update(risk_scaled=r.get("gross_bps", np.nan), risk_lo=r.get("gross_bps_lo", np.nan),
                   risk_hi=r.get("gross_bps_hi", np.nan), vol_ratio=float(t["vol"].mean() / p["vol"].mean()))
    return out


def selective_table(p: pd.DataFrame, cost: float, horizon: int, shares=SHARES, min_history: int = 200,
                    n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """One model's seed-averaged predictions -> one row per (selector, kept share).

    `p` needs fold, date, prob, ret and y, plus `vol` for the risk and placebo checks. Folds
    without enough history to set a cutoff are dropped at every level, so levels compare.
    """
    p = p.reset_index(drop=True)
    # Cutoffs are set on every row, so a quarter's cutoff can use the quarters before it,
    # including those too early to be scored themselves; scoring then keeps eligible rows only.
    eligible = coverage_eligible(p, min_history)
    if not eligible.any():
        return pd.DataFrame()
    folds = p["fold"].to_numpy()
    confidence = (p["prob"] - 0.5).abs().to_numpy()
    has_vol = "vol" in p.columns and p["vol"].notna().any()
    scored = p[eligible].reset_index(drop=True)
    days = np.unique(scored["date"])
    rows = []
    for share in shares:
        selectors = [("confidence", confidence)]
        if has_vol and share < 1.0:
            selectors.append(("volatility (placebo)", p["vol"].to_numpy(float)))
        for name, score in selectors:
            mask = (top_share_mask(score, folds, share, min_history) & (confidence > 0))[eligible]
            rows.append({"selector": name, "share": share, "kept": float(mask.mean()),
                         **_score(scored, mask, days, cost, horizon, n_boot, seed)})
    return pd.DataFrame(rows)
