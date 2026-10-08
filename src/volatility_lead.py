"""E1: is the gain at the most volatile moments a real effect, or one look among many?

The lead (RESULTS.md, selective trading): on the S&P 500, trading the models' mostly long direction at the
2% most volatile stock-moments earned +15.21 bps gross [+1.57, +29.18] over the test quarters. It was the
best of many cells (models x kept shares x selectors), so on its own it proves little. This module tests
it once, under a rule fixed before running (configs/e1_volatility.yaml):

- **No model.** The rule is: go long every stock-moment whose trailing volatility is in the top 2%. If
  the lead was the volatility selection and not the models, this keeps it; if it needs the models, the
  rule loses it.
- **Untouched data.** The lead was found in the test quarters (from 2022-10-27). A rule without a model
  needs no training, so it can be scored on 2020-2022 too, which nothing in this project has scored.
  That earlier period is the replication; the test quarters are only the reproduction.
- **Past-only cutoffs.** Each month's top-2% cutoff and median volatility come from rows dated before
  that month.
- **Uncertainty in blocks of 5 trading days.** Volatile moments cluster over consecutive days, so
  resampling single days would make the intervals too narrow.
- **Not one episode.** The test quarters are rescored without the largest volatile episode and without
  April 2025.
- **Costs that rise with volatility.** Spreads widen in stress, so besides the flat 2 bps a stress cost
  scales with the stock's volatility against the past median (2 bps x that ratio, between 1 and 4).
- **Market or stock.** The market-neutral version subtracts the average return of all stocks at the same
  moment: stock-level selection survives that, a bet that the market rebounds does not.

`analyse` takes one row per stock-moment (date, bar_index, symbol, ret_bps, vol) and returns every number
the report and the verdict need.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MOMENT = ["date", "bar_index"]


@dataclass
class LeadParams:
    share: float = 0.02
    burn_in_days: int = 60
    block_days: int = 5
    base_cost_bps: float = 2.0
    cost_cap: float = 4.0
    hedge_cost_bps: float = 0.5
    discovery_start: str = "2022-10-27"
    claim_bps: float = 15.21
    n_boot: int = 2000
    seed: int = 0
    min_peers: int = 20
    hot_quantile: float = 0.75
    top_days: int = 10
    merge_gap_days: int = 5
    excluded: tuple = ("2025-04-01", "2025-04-30")
    stress_share: float = 0.10
    other_shares: tuple = (0.05, 0.10)
    extra: dict = field(default_factory=dict)


def _days(dates) -> np.ndarray:
    return pd.DatetimeIndex(dates).normalize().values.astype("datetime64[D]")


def past_statistic(values: np.ndarray, dates, fn, burn_in_days: int) -> np.ndarray:
    """For each row, `fn` of the finite values of every row dated before the first day of its month.

    NaN for rows whose month starts before `burn_in_days` trading days of earlier data exist.
    """
    values = np.asarray(values, float)
    d = _days(dates)
    order = np.argsort(d, kind="stable")
    sv, sd = values[order], d[order]
    calendar = np.unique(sd)
    out = np.full(len(values), np.nan)
    months = sd.astype("datetime64[M]")
    for m in np.unique(months):
        start = m.astype("datetime64[D]")
        end = (m + 1).astype("datetime64[D]")
        lo, hi = np.searchsorted(sd, start, "left"), np.searchsorted(sd, end, "left")
        if np.searchsorted(calendar, start) < burn_in_days:
            continue
        past = sv[:lo]
        past = past[np.isfinite(past)]
        if past.size:
            out[order[lo:hi]] = fn(past)
    return out


def top_share(score: np.ndarray, dates, share: float, burn_in_days: int) -> np.ndarray:
    """Rows whose score clears the top-`share` cutoff of all earlier months' rows."""
    score = np.asarray(score, float)
    cut = past_statistic(score, dates, lambda v: np.quantile(v, 1 - share), burn_in_days)
    with np.errstate(invalid="ignore"):
        return np.isfinite(cut) & np.isfinite(score) & (score >= cut)


def stress_cost(vol: np.ndarray, dates, base_bps: float, cap: float, burn_in_days: int) -> np.ndarray:
    """base x (volatility / median volatility of earlier months), clipped to [1, cap]."""
    med = past_statistic(vol, dates, np.median, burn_in_days)
    with np.errstate(invalid="ignore", divide="ignore"):
        return base_bps * np.clip(np.asarray(vol, float) / med, 1.0, cap)


def block_mean(dates, values: np.ndarray, calendar: np.ndarray, block_days: int = 5, n_boot: int = 2000,
               seed: int = 0) -> dict:
    """Mean over rows, with a 95% interval from resampling blocks of `block_days` consecutive trading days.

    `calendar` lists every trading day of the period, including days without rows, so blocks follow the
    calendar and empty days count. Circular moving blocks; the ratio of resampled sums is the estimate.
    `mde` is the effect this much data would detect 80% of the time (2.8 standard errors).
    """
    cal = np.asarray(calendar, "datetime64[D]")
    v = np.asarray(values, float)
    d = _days(dates)
    pos = np.searchsorted(cal, d)
    ok = (pos < len(cal)) & np.isfinite(v)
    ok[ok] &= cal[pos[ok]] == d[ok]
    pos, v = pos[ok], v[ok]
    out = {"rows": int(len(v)), "days": int(len(np.unique(pos))), "calendar_days": int(len(cal))}
    if not len(v) or not len(cal):
        return {**out, "mean": np.nan, "lo": np.nan, "hi": np.nan, "se": np.nan, "mde": np.nan}
    D = len(cal)
    s = np.bincount(pos, weights=v, minlength=D)
    n = np.bincount(pos, minlength=D).astype(float)
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(D / block_days))
    idx = ((rng.integers(0, D, (n_boot, blocks))[:, :, None] + np.arange(block_days)) % D).reshape(n_boot, -1)
    den = n[idx].sum(axis=1)
    boot = np.divide(s[idx].sum(axis=1), den, out=np.full(n_boot, np.nan), where=den > 0)
    se = float(np.nanstd(boot))
    return {**out, "mean": float(s.sum() / n.sum()), "lo": float(np.nanpercentile(boot, 2.5)),
            "hi": float(np.nanpercentile(boot, 97.5)), "se": se, "mde": 2.8 * se}


def find_episodes(rows_per_day: np.ndarray, hot_quantile: float = 0.75, merge_gap: int = 5) -> list[tuple[int, int]]:
    """Volatile episodes as (first, last) positions in the calendar.

    A day is hot when its count of selected rows is in the top (1 - hot_quantile) of days with any; hot
    days at most `merge_gap` trading days apart belong to one episode, which spans every day between them.
    """
    counts = np.asarray(rows_per_day, float)
    positive = counts[counts > 0]
    if not positive.size:
        return []
    hot = np.flatnonzero(counts >= np.quantile(positive, hot_quantile))
    episodes, start, prev = [], hot[0], hot[0]
    for h in hot[1:]:
        if h - prev > merge_gap:
            episodes.append((int(start), int(prev)))
            start = h
        prev = h
    episodes.append((int(start), int(prev)))
    return episodes


def _calendar(dates, start, end) -> np.ndarray:
    days = np.unique(_days(dates))
    return days[(days >= np.datetime64(start, "D")) & (days < np.datetime64(end, "D"))]


def analyse(rows: pd.DataFrame, p: LeadParams) -> dict:
    """Every number of the E1 report. `rows`: date, bar_index, symbol, ret_bps, vol (one row per stock-moment)."""
    rows = rows.reset_index(drop=True)
    dates = rows["date"].to_numpy()
    day = _days(dates)
    ret = rows["ret_bps"].to_numpy(float)
    vol = rows["vol"].to_numpy(float)
    chosen = top_share(vol, dates, p.share, p.burn_in_days)

    peers = rows.groupby(MOMENT)["ret_bps"].transform("size").to_numpy()
    bench = rows.groupby(MOMENT)["ret_bps"].transform("mean").to_numpy()
    bench = np.where(peers >= p.min_peers, bench, np.nan)
    cost = stress_cost(vol, dates, p.base_cost_bps, p.cost_cap, p.burn_in_days)
    series = {
        "gross": ret,
        "net_flat": ret - p.base_cost_bps,
        "net_stress": ret - cost,
        "neutral_gross": ret - bench,
        "neutral_net_stress": ret - bench - cost - p.hedge_cost_bps,
    }

    others = {sh: top_share(vol, dates, sh, p.burn_in_days) for sh in p.other_shares}
    mvol = rows.groupby(MOMENT)["vol"].transform("mean").to_numpy()
    stressed = top_share(mvol, dates, p.stress_share, p.burn_in_days)
    scored = np.isfinite(past_statistic(vol, dates, np.median, p.burn_in_days))
    first = day[scored].min() if scored.any() else day.min()
    split = np.datetime64(p.discovery_start, "D")
    periods = {"pre": (first, split), "disc": (split, day.max() + 1)}

    def stats(mask, values, cal):
        return block_mean(dates[mask], values[mask], cal, p.block_days, p.n_boot, p.seed)

    res = {"periods": {k: (str(a), str(b - 1)) for k, (a, b) in periods.items()}, "selected_share": float(chosen.mean())}
    for name, (a, b) in periods.items():
        cal = _calendar(dates, a, b)
        inside = chosen & (day >= a) & (day < b)
        r = {k: stats(inside, v, cal) for k, v in series.items()}
        r["stocks"] = int(rows.loc[inside, "symbol"].nunique())
        r["vol_ratio"] = float(np.nanmean(vol[inside]) / np.nanmean(vol[(day >= a) & (day < b)])) if inside.any() else np.nan
        r["mean_cost_stress"] = float(np.nanmean(cost[inside])) if inside.any() else np.nan
        r["all_rows_gross"] = stats((day >= a) & (day < b), ret, cal)
        r["by_bar"] = {int(k): stats(inside & (rows["bar_index"].to_numpy() == k), ret, cal)
                       for k in np.unique(rows.loc[inside, "bar_index"])}
        r["by_year"] = {int(y): stats(inside & (day.astype("datetime64[Y]").astype(int) + 1970 == y), ret, cal)
                        for y in np.unique(day[inside].astype("datetime64[Y]").astype(int) + 1970)}
        r["other_shares"] = {sh: stats(m & (day >= a) & (day < b), ret, cal) for sh, m in others.items()}

        counts = np.bincount(np.searchsorted(cal, day[inside]), minlength=len(cal))
        eps = find_episodes(counts, p.hot_quantile, p.merge_gap_days)
        pnl_by_day = np.bincount(np.searchsorted(cal, day[inside]), weights=ret[inside], minlength=len(cal))
        totals = [float(pnl_by_day[s:e + 1].sum()) for s, e in eps]
        r["episodes"] = [{"start": str(cal[s]), "end": str(cal[e]), "rows": int(counts[s:e + 1].sum()),
                          "pnl_share": t / pnl_by_day.sum() if pnl_by_day.sum() else np.nan}
                         for (s, e), t in zip(eps, totals)]
        if eps:
            s, e = eps[int(np.argmax(np.abs(totals)))]
            out = (day >= cal[s]) & (day <= cal[e])
            r["largest_episode"] = {"start": str(cal[s]), "end": str(cal[e]),
                                    "without": stats(inside & ~out, ret, cal[(cal < cal[s]) | (cal > cal[e])])}
        top = np.argsort(-np.abs(pnl_by_day))[: p.top_days]
        keep_days = np.setdiff1d(np.arange(len(cal)), top)
        r["top_days"] = {"days": [str(cal[i]) for i in np.sort(top)],
                         "pnl_share": float(pnl_by_day[top].sum() / pnl_by_day.sum()) if pnl_by_day.sum() else np.nan,
                         "without": stats(inside & ~np.isin(day, cal[top]), ret, cal[keep_days])}
        x0, x1 = (np.datetime64(x, "D") for x in p.excluded)
        r["without_excluded"] = stats(inside & ~((day >= x0) & (day <= x1)), ret, cal[(cal < x0) | (cal > x1)])

        r["market_stress"] = stats(inside & stressed, ret, cal)
        r["market_calm"] = stats(inside & ~stressed, ret, cal)
        r["market_stress_rows_share"] = float(stressed[inside].mean()) if inside.any() else np.nan
        res[name] = r
    res["verdict"] = verdict(res, p)
    return res


def verdict(res: dict, p: LeadParams) -> dict:
    """The rule in configs/e1_volatility.yaml, applied in order."""
    pre, disc = res["pre"], res["disc"]
    gp, gd = pre["gross"], disc["gross"]
    out = {"stock_level": bool(pre["neutral_gross"]["lo"] > 0 and disc["neutral_gross"]["lo"] > 0)}
    if not gd["lo"] > 0:
        return {**out, "level": "not replicated",
                "reason": "the rule without a model shows no gain whose interval is above zero even in the test quarters, "
                          "where the lead was found"}
    if not gp["lo"] > 0:
        if not gp["mde"] <= p.claim_bps:
            return {**out, "level": "inconclusive",
                    "reason": f"2020-2022 cannot tell (it would detect {gp['mde']:.1f} bps, the claim is {p.claim_bps} bps)"}
        return {**out, "level": "not replicated",
                "reason": "no gain whose interval is above zero in 2020-2022, the period nothing had scored before"}
    if not disc.get("largest_episode", {}).get("without", {}).get("mean", np.nan) > 0:
        return {**out, "level": "not replicated", "reason": "in the test quarters it rests on one volatile episode"}
    if not disc["top_days"]["without"]["mean"] > 0:
        return {**out, "level": "not replicated",
                "reason": f"in the test quarters it rests on its {p.top_days} biggest days"}
    if not disc["without_excluded"]["mean"] > 0:
        return {**out, "level": "not replicated", "reason": "in the test quarters it rests on April 2025"}
    if pre["net_stress"]["lo"] > 0 and disc["net_stress"]["lo"] > 0:
        return {**out, "level": "replicated, pays under stress costs", "reason": "every condition holds"}
    if pre["net_flat"]["lo"] > 0 and disc["net_flat"]["lo"] > 0:
        return {**out, "level": "replicated, pays at a flat 2 bps only",
                "reason": "the gain is real in both periods but does not survive costs that rise with volatility"}
    return {**out, "level": "replicated, does not pay", "reason": "the gain is real in both periods but smaller than costs"}
