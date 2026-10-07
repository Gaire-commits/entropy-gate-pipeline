"""Cross-sectional tests: what the breadth of a wide universe is worth, and what it is not.

Every stock sampled at the same date and bar shares that moment's market move, and stocks in a sector
share its move. Scoring each trade against its own direction therefore counts one bet on the market
hundreds of times: on the S&P 500, 673,694 trades give an interval no narrower than 14,005 ETF
trades, because trades placed together move together, and a model that is nearly always long makes
one market bet however many stocks it holds. The fundamental law of active management (Grinold and
Kahn) says why this matters: information ratio ~ IC x sqrt(breadth), where breadth counts
*independent* bets, not trades.

Three tools, all on saved walk-forward predictions, none needing retraining:

- effective_bets: how many independent bets a day's trades amount to, from how strongly the trades of
  a day move together. Around 1 means one bet on the market; the trade count means true diversification.
- cross_sectional_ic: at each moment, the rank correlation between the model's scores and the realized
  returns across stocks, averaged over days with a day-block interval. Market and (optionally) sector
  moves cancel in a cross-section, and every stock contributes, so it is the most powerful test of
  stock-level skill the data allows.
- quantile_book: long the top q and short the bottom q by score at each moment (optionally within each
  sector): market-neutral by construction, with the cost charged per position.

breakeven_ic converts the cost into the IC a signal needs before it pays, so a result near zero can be
read against what profitability would require.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

MOMENT = ["date", "bar_index"]


def with_sector(frame: pd.DataFrame, sectors: dict[str, str] | None) -> pd.DataFrame:
    """Add a `sector` column: from the symbol -> sector map, 'all' without one, 'unknown' for unmapped symbols."""
    out = frame.copy()
    out["sector"] = out["symbol"].map(sectors).fillna("unknown") if sectors else "all"
    return out


def wide_moments(frame: pd.DataFrame, min_peers: int) -> pd.DataFrame:
    """Keep only moments with at least `min_peers` symbols: a cross-section of three stocks is not a market."""
    n = frame.groupby(MOMENT)["symbol"].transform("size")
    return frame[n >= min_peers]


def _day_bootstrap(values: np.ndarray, weights: np.ndarray | None = None, n_boot: int = 2000, seed: int = 0):
    """(estimate, lo, hi, se) of a weighted mean over days, resampling whole days."""
    w = np.ones_like(values) if weights is None else np.asarray(weights, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (n_boot, len(values)))
    boot = (values[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
    return (float((values * w).sum() / w.sum()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)), float(boot.std()))


def moment_ic(frame: pd.DataFrame, score: str = "prob", min_peers: int = 20, sector_neutral: bool = False,
              ret: str = "ret") -> tuple[pd.Series, pd.Series]:
    """(rank IC per (date, bar) moment, symbols per moment) for moments with at least `min_peers` symbols.

    The Pearson correlation of the ranks of `score` and of `ret` across symbols; with `sector_neutral`
    each rank has its sector's average rank at that moment subtracted first.
    """
    need = list(dict.fromkeys(MOMENT + ["symbol", score, ret] + (["sector"] if sector_neutral else [])))
    f = wide_moments(frame, min_peers)[need].copy()
    if f.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    g = f.groupby(MOMENT)
    f["x"], f["y"] = g[score].rank(method="average"), g[ret].rank(method="average")
    if sector_neutral:
        sector = f.groupby(MOMENT + ["sector"])
        f["x"] -= sector["x"].transform("mean")
        f["y"] -= sector["y"].transform("mean")
    f["xy"], f["xx"], f["yy"] = f["x"] * f["y"], f["x"] ** 2, f["y"] ** 2
    s = f.groupby(MOMENT)[["x", "y", "xy", "xx", "yy"]].sum()
    n = f.groupby(MOMENT).size()
    cov = s["xy"] - s["x"] * s["y"] / n
    var = (s["xx"] - s["x"] ** 2 / n) * (s["yy"] - s["y"] ** 2 / n)
    ic = (cov / np.sqrt(var.where(var > 1e-12))).dropna()
    return ic, n.loc[ic.index]


def summarize_ic(ic: pd.Series, peers: pd.Series | None = None, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Average a per-moment IC within days, then over days, with a day-block interval and a t-statistic."""
    if ic.empty:
        return {"days": 0}
    by_day = ic.groupby(level="date").mean()
    mean, lo, hi, se = _day_bootstrap(by_day.to_numpy(), n_boot=n_boot, seed=seed)
    sd = by_day.std()
    return {"ic": mean, "ic_lo": lo, "ic_hi": hi, "ic_se": se, "ic_min_detectable": 2.8 * se,
            "t": float(by_day.mean() / (sd / np.sqrt(len(by_day)))) if sd > 0 else float("nan"),
            "share_days_positive": float((by_day > 0).mean()), "days": int(len(by_day)),
            "moments": int(len(ic)), "peers": float(peers.mean()) if peers is not None and len(peers) else float("nan")}


def cross_sectional_ic(frame: pd.DataFrame, score: str = "prob", min_peers: int = 20, sector_neutral: bool = False,
                       n_boot: int = 2000, seed: int = 0, ret: str = "ret") -> dict[str, float]:
    """Mean cross-sectional rank IC with a day-block interval.

    At each (date, bar) moment with enough symbols: the Pearson correlation of the ranks of `score` and of
    the realized return across symbols. With `sector_neutral` each rank has its sector's average rank at
    that moment subtracted first, so a model that only knows which sector will move scores zero. Moments
    are averaged within each day, and days are resampled for the interval, because a day's moments share
    news. `ic_min_detectable` is the IC this much data would detect 80% of the time.
    """
    ic, peers = moment_ic(frame, score, min_peers, sector_neutral, ret)
    return summarize_ic(ic, peers, n_boot, seed)


def cross_sectional_sigma(frame: pd.DataFrame, min_peers: int = 20, sector_neutral: bool = False) -> float:
    """Typical spread (std, in bps) of returns across stocks within a moment, after removing the market
    (and, if asked, the sector) average: the scale a cross-sectional IC converts into bps."""
    f = wide_moments(frame, min_peers)
    keys = MOMENT + (["sector"] if sector_neutral else [])
    resid = f["ret"] - f.groupby(keys)["ret"].transform("mean")
    return float(1e4 * np.sqrt((resid ** 2).groupby([f[k] for k in MOMENT]).mean()).mean())


def breakeven_ic(cost_bps: float, sigma_bps: float, q: float = 0.1) -> float:
    """The IC at which a long-top-q, short-bottom-q book earns exactly its cost.

    With a standardized score correlated IC with the return, the picks' average return is IC x sigma x m,
    where m = phi(z_q) / q is the average |z| of the top q of a normal score (1.755 for q = 10%).
    """
    m = norm.pdf(norm.ppf(1 - q)) / q
    return float(cost_bps / (sigma_bps * m))


def quantile_book(frame: pd.DataFrame, score: str = "prob", q: float = 0.1, min_peers: int = 20,
                  sector_neutral: bool = False, min_sector: int = 5, seed: int = 0,
                  n_per_side: int | None = None) -> pd.DataFrame:
    """Positions of a long-short book: +1 for the top and -1 for the bottom of `score` at each moment.

    The top and bottom `q` of the cross-section, or with `n_per_side` a fixed number of names on each side
    (a book of 2 x n_per_side stocks). Ranked within each sector when `sector_neutral` (a sector-balanced
    book: with `n_per_side` each sector gets picks in proportion to its size, at least one a side),
    otherwise across the whole moment. Ties, which coarse scores have a lot of, are broken at random so
    that symbol order cannot decide who is picked. Returns one row per position with `direction` and
    `gross_bps`.
    """
    need = MOMENT + ["symbol", score, "ret"] + (["sector"] if "sector" in frame else [])
    f = wide_moments(frame, min_peers)[need].copy()
    f["_s"] = f[score] + 1e-9 * np.random.default_rng(seed).random(len(f))
    keys = MOMENT + (["sector"] if sector_neutral else [])
    g = f.groupby(keys)["_s"]
    n = g.transform("size")
    floor = min_sector if sector_neutral else min_peers
    if n_per_side is None:
        pct = (g.rank(method="first") - 0.5) / n
        ok = n >= floor
        long_, short = pct > 1 - q, pct < q
    else:
        if sector_neutral:
            share = n / f.groupby(MOMENT)["_s"].transform("size")
            k = np.maximum(1, np.round(n_per_side * share)).astype(int)
        else:
            k = pd.Series(n_per_side, index=f.index)
        ok = (n >= floor) & (n >= 2 * k)
        long_ = g.rank(method="first", ascending=False) <= k
        short = g.rank(method="first", ascending=True) <= k
    f["direction"] = np.where(ok & long_, 1.0, np.where(ok & short, -1.0, 0.0))
    book = f[f["direction"] != 0].drop(columns="_s")
    book["gross_bps"] = book["direction"] * book["ret"] * 1e4
    return book


def effective_bets(book: pd.DataFrame, value: str = "gross_bps") -> dict[str, float]:
    """How many independent bets a day's positions amount to.

    Positions of one day have a common component: if each pair of positions has correlation rho, the
    average of n of them has variance sigma^2 [(1 - rho) / n + rho], which no number of extra positions
    can push below sigma^2 rho. So rho is read off the data, as the variance of day averages against the
    variance of single positions, and the effective number of bets is n / (1 + (n - 1) rho): near 1 for
    one bet on the market, near n for fully independent positions. Unit is the day, which is also the
    unit the confidence intervals resample.
    """
    v = book[value].to_numpy(float)
    if len(v) < 10:
        return {"rho": float("nan"), "bets_per_day": float("nan"), "positions_per_day": float("nan")}
    days = book.groupby("date")[value].agg(["mean", "size"])
    mu, sigma2 = v.mean(), v.var()
    a = float(((days["mean"] - mu) ** 2).mean())
    b = float((1.0 / days["size"]).mean())
    rho = float(np.clip((a / sigma2 - b) / (1 - b), 0.0, 1.0)) if sigma2 > 0 and b < 1 else float("nan")
    n_bar = float(days["size"].mean())
    return {"rho": rho, "positions_per_day": n_bar,
            "bets_per_day": float(n_bar / (1 + (n_bar - 1) * rho)) if np.isfinite(rho) else float("nan")}


def score_book(book: pd.DataFrame, cost_bps: float, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Gross and net bps per position with a day-block interval, exposure, and effective bets."""
    if book.empty:
        return {"positions": 0}
    d = book.groupby("date").agg(pnl=("gross_bps", "sum"), n=("gross_bps", "size"), exposure=("direction", "mean"))
    mean, lo, hi, _ = _day_bootstrap(d["pnl"].to_numpy() / d["n"].to_numpy(), d["n"].to_numpy(), n_boot, seed)
    legs = book.groupby(book["direction"] > 0)["gross_bps"].mean()
    return {
        "positions": int(len(book)), "positions_per_day": float(d["n"].mean()), "days": int(len(d)),
        "gross": mean, "gross_lo": lo, "gross_hi": hi, "net": mean - cost_bps,
        "long_leg": float(legs.get(True, np.nan)), "short_leg": float(legs.get(False, np.nan)),
        "hit_rate": float((book["gross_bps"] > 0).mean()),
        "net_exposure": float(d["exposure"].mean()), **effective_bets(book),
    }


def directional_book(frame: pd.DataFrame, score: str = "prob", mask: np.ndarray | None = None) -> pd.DataFrame:
    """The positions our earlier tests scored: each kept sample long or short by its own signal. A score of
    exactly 0.5 is no view and no position."""
    f = frame if mask is None else frame[np.asarray(mask, bool)]
    f = f[f[score] != 0.5]
    book = f[["date", "symbol", "bar_index", "ret"] + (["sector"] if "sector" in f else [])].copy()
    book["direction"] = np.where(f[score] > 0.5, 1.0, -1.0)
    book["gross_bps"] = book["direction"] * book["ret"] * 1e4
    return book


def zscore_within_moments(frame: pd.DataFrame, col: str) -> pd.Series:
    """Each score as standard deviations from its moment's mean; a moment with no spread scores 0."""
    g = frame.groupby(MOMENT)[col]
    sd = g.transform("std")
    return ((frame[col] - g.transform("mean")) / sd.where(sd > 1e-12)).fillna(0.0)


def ensemble_scores(frames: dict[str, pd.DataFrame], min_peers: int = 20) -> tuple[pd.DataFrame, dict[str, float]]:
    """Average several models' within-moment z-scores into one score, `ens`, on the rows all of them cover.

    Returns the merged frame (with `z_<model>` columns) and a summary of how alike the models are: the
    mean pairwise correlation of their z-scores and the number of independent models it amounts to,
    M / (1 + (M - 1) rho). Models trained on the same inputs and labels are close to one model.
    """
    key = ["fold", "date", "symbol", "bar_index"]
    base = None
    for name, f in frames.items():
        part = f[key + ["ret", "prob"]].rename(columns={"prob": f"p_{name}"})
        base = part if base is None else base.merge(part.drop(columns="ret"), on=key, how="inner")
    base = wide_moments(base, min_peers).reset_index(drop=True)
    zs = []
    for name in frames:
        base[f"z_{name}"] = zscore_within_moments(base, f"p_{name}")
        zs.append(f"z_{name}")
    base["ens"] = base[zs].mean(axis=1)
    m = len(zs)
    corr = np.corrcoef(base[zs].to_numpy().T) if m > 1 else np.ones((1, 1))
    rho = float(corr[~np.eye(m, dtype=bool)].mean()) if m > 1 else float("nan")
    return base, {"models": m, "mean_corr": rho, "independent_models": float(m / (1 + (m - 1) * rho)) if m > 1 else 1.0}


# ---------------------------------------------------------------- is the skill real? (scripts/skill_checks.py)

def past_symbol_mean(frame: pd.DataFrame, col: str) -> np.ndarray:
    """For each row, the mean of `col` over the same symbol's rows in earlier folds only (NaN before any)."""
    g = frame.groupby(["symbol", "fold"])[col].agg(["sum", "count"]).sort_index()
    earlier = g.groupby(level="symbol").cumsum() - g
    mean = earlier["sum"] / earlier["count"].where(earlier["count"] > 0)
    return mean.reindex(pd.MultiIndex.from_arrays([frame["symbol"], frame["fold"]])).to_numpy()


def skill_decomposition(frame: pd.DataFrame, score: str = "prob", min_peers: int = 20, n_boot: int = 2000,
                        seed: int = 0) -> dict[str, dict]:
    """Is a cross-sectional IC a fixed preference for some stocks, or timing?

    model: the plain IC.
    model, returns net of stock averages: the same score against each return minus that stock's average
        return over the whole period, which removes any stock's persistent out- or under-performance,
        including what a list of today's index members carries (survivorship). It uses the future, which is
        fine for a diagnosis; it is not a strategy.
    static: each stock scored by its average score in earlier quarters only -- one fixed ranking of stocks.
    timing: the score minus that earlier average, against returns net of stock averages.

    Rows from the first quarter have no earlier average and are left out of all four, so they compare.
    A static IC that carries the model's IC, with timing near zero, is the signature of a fixed tilt
    (survivorship can produce one); timing near the model IC is skill at telling when.
    """
    f = frame.copy()
    f["_static"] = past_symbol_mean(f, score)
    f = f[np.isfinite(f["_static"])].copy()
    f["_timing"] = f[score] - f["_static"]
    f["_ret_net"] = f["ret"] - f.groupby("symbol")["ret"].transform("mean")
    kw = {"min_peers": min_peers, "n_boot": n_boot, "seed": seed}
    return {
        "model": cross_sectional_ic(f, score, **kw),
        "model, returns net of stock averages": cross_sectional_ic(f, score, ret="_ret_net", **kw),
        "static": cross_sectional_ic(f, "_static", **kw),
        "timing": cross_sectional_ic(f, "_timing", ret="_ret_net", **kw),
    }


def ic_by_period(frame: pd.DataFrame, by: str = "fold", score: str = "prob", min_peers: int = 20) -> pd.DataFrame:
    """Mean IC, t and share of positive days for each value of `by` ("fold" for quarters, "bar_index" for the
    time of day), from the same per-moment ICs as the overall figure."""
    ic, _ = moment_ic(frame, score, min_peers)
    if ic.empty:
        return pd.DataFrame()
    moments = ic.to_frame("ic")
    if by in MOMENT:                                   # part of the moment's own key (e.g. the time of day)
        moments["_by"] = moments.index.get_level_values(by)
    else:
        moments = moments.join(frame[MOMENT + [by]].drop_duplicates(MOMENT).set_index(MOMENT)[by].rename("_by"))
    rows = []
    for value, part in moments.groupby("_by"):
        by_day = part["ic"].groupby(level="date").mean()
        sd = by_day.std()
        rows.append({by: value, "start": by_day.index.min(), "end": by_day.index.max(), "days": len(by_day),
                     "ic": float(by_day.mean()), "share_days_positive": float((by_day > 0).mean()),
                     "t": float(by_day.mean() / (sd / np.sqrt(len(by_day)))) if sd > 0 and len(by_day) > 1 else float("nan")})
    return pd.DataFrame(rows)


def consistency(per_period: pd.DataFrame) -> dict[str, float]:
    """How many periods had a positive IC, the one-sided sign-test p-value of that count, and the largest
    single period's share of the summed positive IC (near 1 = one period carries everything)."""
    from scipy.stats import binomtest

    ic = per_period["ic"].to_numpy()
    k, n = int((ic > 0).sum()), len(ic)
    positive = ic[ic > 0]
    return {"positive": k, "periods": n,
            "sign_p": float(binomtest(k, n, 0.5, alternative="greater").pvalue) if n else float("nan"),
            "largest_share": float(positive.max() / positive.sum()) if positive.size else float("nan")}
