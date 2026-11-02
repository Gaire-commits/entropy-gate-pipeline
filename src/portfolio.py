"""A portfolio tool on daily stock returns: forecast, rebalance, and test what permutation entropy adds.

Everything is arranged so a decision at the close of session t can only use what was known then:

  * features and eligibility use data up to and including t;
  * an order decided at t executes at session t+1 (its close by default, the official close per the data audit);
  * the forecast target runs from that execution price to the one `horizon` sessions later;
  * a model refit at t trains only on rows whose target was fully realized by t.

Three nested ridge models share the same rows, folds and portfolio rules: `base` (time-series and cross-sectional
features), `base_pe` (plus the cross-sectional rank of permutation entropy) and `base_pe_x` (plus interactions fixed in
advance). notebooks/portfolio_tool.ipynb runs them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import stats as sps

from src import volpanel


@dataclass(frozen=True)
class Config:
    # the portfolio
    rebalance_every: int = 5        # sessions between decisions
    horizon: int = 5                # sessions the forecast target spans
    top_frac: float = 0.10          # buy the best tenth of eligible stocks, equally weighted
    hold_frac: float = 0.20         # keep a holding while it stays in the best fifth (cuts turnover)
    sector_cap: float = 0.30        # no sector above this share of the names held
    execution: str = "next_close"   # or "next_open": the vendor's open is the first trade, not the official open
    # costs
    cost_bps: float = 5.0           # one-way: half the spread, fees, on every dollar traded
    impact: float = 1.0             # square-root impact: impact x daily vol x sqrt(trade / daily dollar volume)
    aum: float = 10e6               # starting capital in dollars, for the impact term
    # eligibility (known at the close of t)
    min_raw_close: float = 5.0
    min_dollar_volume: float = 10e6   # trailing 63-session median, earlier sessions only
    min_coverage: float = 0.9         # real bars over the last 252 sessions
    # the model
    ridge: float = 0.01             # penalty per row, on standardized features
    min_train_sessions: int = 504
    refit_every: int = 63
    train_sample_every: int = 5     # training dates, sampled so overlapping targets do not multiply rows
    # permutation entropy of daily returns, fixed before any result
    pe_m: int = 3
    pe_tau: int = 1
    pe_window: int = 63
    # the windows: decision dates. The last is not scored unless asked for
    validation_start: str = "2022-01-04"
    holdout_start: str = "2024-01-03"
    holdout_end: str = "2026-10-12"
    seed: int = 0


TS_FEATURES = ["ts_ret_1", "ts_ret_5", "ts_ret_21", "ts_mom_12_1", "ts_vol_ratio"]
XS_FEATURES = ["xs_ret_5", "xs_ret_21", "xs_mom_6_1", "xs_mom_12_1", "xs_vol_63"]
SECTOR_FEATURES = ["sec_rel_ret_21", "sec_rel_mom_12_1", "sector_ret_21"]
BASE = TS_FEATURES + XS_FEATURES + SECTOR_FEATURES
PE_FEATURE = ["xs_pe"]
# Fixed in advance. Ordinal disorder should matter for how far past moves carry on (momentum) or snap back (the 5-day
# move), and entropy moves with volatility, so its interaction with volatility is the control.
INTERACTIONS = ["pe_x_mom_12_1", "pe_x_ret_5", "pe_x_vol_63"]
MODELS = {"base": BASE, "base_pe": BASE + PE_FEATURE, "base_pe_x": BASE + PE_FEATURE + INTERACTIONS}


# ---------------------------------------------------------------- the market panel

@dataclass
class Market:
    """Wide tables, sessions by symbols. Prices are `all`-adjusted except raw_close; `real` marks a bar that traded."""
    close: pd.DataFrame
    open: pd.DataFrame
    raw_close: pd.DataFrame
    dollar_volume: pd.DataFrame
    real: pd.DataFrame
    event: pd.DataFrame            # sessions whose return is mechanical or not trustworthy: return set to zero
    sector: pd.Series
    truncated: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))


def lock_date(cfg: Config, score_holdout: bool = False) -> pd.Timestamp:
    """The last calendar date any price may come from: the hold-out's last day if it is being scored, else the day before it."""
    return pd.Timestamp(cfg.holdout_end) if score_holdout else pd.Timestamp(cfg.holdout_start) - pd.Timedelta(days=1)


def build_market(raw: pd.DataFrame, split: pd.DataFrame, adj: pd.DataFrame, actions: pd.DataFrame, registry: pd.DataFrame,
                 universe: pd.DataFrame, until: pd.Timestamp | None = None) -> Market:
    """The panel from the vendor's daily bars, with the rules the data audit settled on: placeholder bars do not count,
    the vendor's truncated days hold no usable price, and corporate-action sessions are marked (src/volpanel.py). Nothing
    after `until` is loaded, so a locked period is not merely unscored but absent."""
    def dated(f):
        f = f[f["symbol"].isin(universe["symbol"])].copy()
        f["date"] = pd.to_datetime(f["date"]).dt.normalize()
        return f if until is None else f[f["date"] <= until]
    raw, split, adj = dated(raw), dated(split), dated(adj)
    v = volpanel.vendor_frame(raw, split, adj)
    rules = volpanel.Rules()
    gaps = volpanel.market_gap_days(v, rules.market_gap_ratio, rules.market_gap_min_symbols)
    truncated = pd.DatetimeIndex(gaps[gaps].index)
    cal = pd.DatetimeIndex(sorted(v["date"].unique()))
    events = volpanel.event_sessions(actions, v, registry, cal, rules)
    events = events[events["symbol"].isin(universe["symbol"])]

    def wide(col, frame=v):
        return frame.pivot(index="date", columns="symbol", values=col).reindex(index=cal)
    real = wide("real").eq(True)
    real.loc[real.index.isin(truncated)] = False
    close = wide("close").where(real)
    dv = raw.assign(dv=raw["close"] * raw["volume"]).pivot(index="date", columns="symbol", values="dv").reindex(index=cal, columns=close.columns)
    event = pd.DataFrame(False, index=cal, columns=close.columns)
    for s, d in zip(events["symbol"], events["date"]):
        if s in event.columns and d in event.index:
            event.loc[d, s] = True
    sector = universe.set_index("symbol")["sector"].reindex(close.columns).fillna("Unknown")
    return Market(close=close, open=wide("open").where(real), raw_close=wide("raw_close").where(real),
                  dollar_volume=dv.where(real), real=real, event=event, sector=sector, truncated=truncated)


def clean_returns(m: Market) -> pd.DataFrame:
    """Daily log returns between consecutive sessions with a real bar; empty across a gap and on an event session."""
    r = np.log(m.close / m.close.shift(1))
    return r.where(~m.event)


def asset_returns(m: Market, execution: str = "next_close") -> pd.DataFrame:
    """What a holding earns between one session's execution price and the next: a gap in the data earns zero and then
    the whole move on the next real bar; an event session earns zero, because its return is not a price move."""
    price = {"next_close": m.close, "next_open": m.open}[execution]
    r = price.ffill() / price.ffill().shift(1) - 1.0
    return r.where(~m.event, 0.0).fillna(0.0)


def adv63(m: Market) -> pd.DataFrame:
    """Trailing median daily dollar volume, sessions before t only."""
    return m.dollar_volume.shift(1).rolling(63, min_periods=40).median()


# ---------------------------------------------------------------- permutation entropy

def rolling_pe(returns: pd.DataFrame, m: int = 3, tau: int = 1, window: int = 63, min_share: float = 0.9) -> pd.DataFrame:
    """Normalized permutation entropy of each stock's daily returns over the last `window` patterns, ending at t. A pattern
    holding a missing return or a tie is left out rather than resolved by position; too few valid patterns give no value."""
    from math import factorial, log
    x = returns.to_numpy(dtype=float)
    T, N = x.shape
    lag = (m - 1) * tau
    states = factorial(m)
    codes = np.full((T, N), -1, dtype=np.int64)
    if T > lag:
        v = np.stack([x[lag - k * tau: T - k * tau or None] for k in range(m - 1, -1, -1)], axis=2)   # oldest first
        order = np.argsort(v, axis=2, kind="stable")
        weights = m ** np.arange(m)
        raw = (order * weights).sum(axis=2)
        sorted_v = np.take_along_axis(v, order, axis=2)
        ok = np.isfinite(v).all(axis=2) & (np.diff(sorted_v, axis=2) != 0).all(axis=2)
        uniq, inv = np.unique(raw, return_inverse=True)
        c = inv.reshape(raw.shape)
        codes[lag:][ok] = c[ok]
        n_states = len(uniq)
    else:
        n_states = states
    counts = np.zeros((T + 1, N, n_states), dtype=np.int32)
    valid = codes >= 0
    rows, cols = np.nonzero(valid)
    np.add.at(counts, (rows + 1, cols, codes[rows, cols]), 1)
    counts = counts.cumsum(axis=0)
    out = np.full((T, N), np.nan)
    if T >= window:
        w = counts[window:] - counts[:-window]              # the window ending at each t
        total = w.sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = w / total[:, :, None]
            h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=2) / log(states)
        h[total < min_share * window] = np.nan
        out[window - 1:] = h
    return pd.DataFrame(out, index=returns.index, columns=returns.columns)


# ---------------------------------------------------------------- features, eligibility, target

def xs_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Each date's cross-sectional rank, centered: -0.5 to +0.5. Missing stays missing."""
    return (frame.rank(axis=1) - 0.5).div(frame.notna().sum(axis=1), axis=0) - 0.5


def _within_sector_rank(frame: pd.DataFrame, sector: pd.Series) -> pd.DataFrame:
    parts = [xs_rank(frame[names]) for _, names in sector.groupby(sector).groups.items()]
    return pd.concat(parts, axis=1).reindex(columns=frame.columns)


def _sector_level(frame: pd.DataFrame, sector: pd.Series) -> pd.DataFrame:
    """Rank of each stock's sector by the sector's mean, across sectors, mapped back to its stocks."""
    means = frame.T.groupby(sector).mean().T
    ranks = xs_rank(means)
    return pd.DataFrame({s: ranks[sector[s]] for s in frame.columns}, index=frame.index).where(frame.notna())


def trailing(returns: pd.DataFrame, n: int, skip: int = 0, min_share: float = 0.8) -> pd.DataFrame:
    """Sum of log returns over the n sessions ending `skip` sessions before t."""
    return returns.rolling(n, min_periods=int(np.ceil(min_share * n))).sum().shift(skip)


def build_features(m: Market, cfg: Config) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Every feature at the close of t, and which stocks are eligible at t. Ranks are taken among eligible stocks only,
    and a stock is eligible only if all the raw inputs exist, so the three models see exactly the same rows."""
    r = clean_returns(m)
    ret1, ret5, ret21 = trailing(r, 1, min_share=1.0), trailing(r, 5), trailing(r, 21)
    mom61, mom121 = trailing(r, 105, 21), trailing(r, 231, 21)
    vol21 = r.rolling(21, min_periods=17).std()
    vol63 = r.rolling(63, min_periods=50).std()
    vol252 = r.rolling(252, min_periods=200).std()
    pe = rolling_pe(r, cfg.pe_m, cfg.pe_tau, cfg.pe_window)

    inputs = [ret1.fillna(0.0).where(m.real), ret5, ret21, mom61, mom121, vol21, vol63, vol252, pe]
    have = np.logical_and.reduce([x.notna().to_numpy() for x in inputs])
    coverage = m.real.rolling(252, min_periods=252).mean()
    eligible = pd.DataFrame(have, index=m.close.index, columns=m.close.columns) & m.real \
        & (m.raw_close >= cfg.min_raw_close) & (adv63(m) >= cfg.min_dollar_volume) & (coverage >= cfg.min_coverage)

    clip = lambda f: f.clip(-4, 4).where(eligible)  # noqa: E731
    feats = {
        # each stock against its own history: the move in units of its own volatility
        "ts_ret_1": clip(ret1.fillna(0.0) / vol63),
        "ts_ret_5": clip(ret5 / (vol63 * np.sqrt(5))),
        "ts_ret_21": clip(ret21 / (vol63 * np.sqrt(21))),
        "ts_mom_12_1": clip(mom121 / (vol252 * np.sqrt(231))),
        "ts_vol_ratio": clip(np.log(vol21 / vol63)),
        # each stock against the others at t
        "xs_ret_5": xs_rank(ret5.where(eligible)),
        "xs_ret_21": xs_rank(ret21.where(eligible)),
        "xs_mom_6_1": xs_rank(mom61.where(eligible)),
        "xs_mom_12_1": xs_rank(mom121.where(eligible)),
        "xs_vol_63": xs_rank(vol63.where(eligible)),
        # each stock against its sector, and the sector against the others
        "sec_rel_ret_21": _within_sector_rank(ret21.where(eligible), m.sector),
        "sec_rel_mom_12_1": _within_sector_rank(mom121.where(eligible), m.sector),
        "sector_ret_21": _sector_level(ret21.where(eligible), m.sector),
        "xs_pe": xs_rank(pe.where(eligible)),
    }
    feats["pe_x_mom_12_1"] = feats["xs_pe"] * feats["xs_mom_12_1"]
    feats["pe_x_ret_5"] = feats["xs_pe"] * feats["xs_ret_5"]
    feats["pe_x_vol_63"] = feats["xs_pe"] * feats["xs_vol_63"]
    return feats, eligible


def forward_returns(m: Market, cfg: Config) -> pd.DataFrame:
    """The return from the execution price at t+1 to the one at t+1+horizon: what a decision at t can actually earn. It is
    realized at session index t+1+horizon, which is what purge_end() guards. Empty where either end has no real bar, since
    no fill can happen there and the book would not earn it. reconcile_target() checks this against the book itself."""
    growth = (1.0 + asset_returns(m, cfg.execution)).cumprod()
    fillable = m.real.shift(-1, fill_value=False) & m.real.shift(-(1 + cfg.horizon), fill_value=False)
    return (growth.shift(-(1 + cfg.horizon)) / growth.shift(-1) - 1.0).where(fillable)


def purge_end(refit_index: int, cfg: Config) -> int:
    """The last training date index for a model refit at the close of `refit_index`: its target must be realized by then."""
    return refit_index - 1 - cfg.horizon


# ---------------------------------------------------------------- the model

def fit_ridge(X: np.ndarray, y: np.ndarray, ridge: float) -> dict:
    """Ridge on standardized features, penalty `ridge` per row. Coefficients read as: target rank per one standard
    deviation of the feature."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    Z = (X - mean) / std
    A = Z.T @ Z + ridge * len(y) * np.eye(Z.shape[1])
    coef = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return {"mean": mean, "std": std, "coef": coef, "intercept": float(y.mean())}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    return model["intercept"] + ((X - model["mean"]) / model["std"]) @ model["coef"]


def _stack(feats: dict, names: list[str], day: pd.Timestamp, mask: pd.Series) -> tuple[np.ndarray, pd.Index]:
    idx = mask.index[mask.to_numpy()]
    return np.column_stack([feats[n].loc[day, idx].to_numpy(dtype=float) for n in names]), idx


def decision_dates(index: pd.DatetimeIndex, cfg: Config, score_holdout: bool = False) -> pd.DatetimeIndex:
    """Dates with a decision: every `rebalance_every` sessions once `min_train_sessions` of usable history exist, and only
    where the decision's whole target is realized by the lock."""
    at = np.arange(252 + cfg.min_train_sessions, len(index) - 1 - cfg.horizon, cfg.rebalance_every)
    return index[at[index[at + 1 + cfg.horizon] <= lock_date(cfg, score_holdout)]]


def scored_end(index: pd.DatetimeIndex, cfg: Config, score_holdout: bool = False) -> pd.Timestamp:
    """The last session whose returns may be scored."""
    return index[index <= lock_date(cfg, score_holdout)][-1]


def walk_forward(feats: dict, eligible: pd.DataFrame, target: pd.DataFrame, names: list[str], dates: pd.DatetimeIndex,
                 cfg: Config, fit_until: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample forecasts on `dates`, refitting every `refit_every` sessions on all earlier realized rows, none of them
    after `fit_until` if given. Returns the forecasts (dates by symbols) and each refit's coefficients."""
    index = eligible.index
    y = xs_rank(target.where(eligible))
    forecasts = pd.DataFrame(np.nan, index=dates, columns=eligible.columns)
    coefs, model, next_refit = [], None, -1
    train_days = index[252::cfg.train_sample_every]
    for day in dates:
        i = index.get_loc(day)
        if model is None or i >= next_refit:
            last = purge_end(i, cfg)
            Xs, ys = [], []
            limit = index[last] if fit_until is None else min(index[last], fit_until)
            for d in train_days[train_days <= limit]:
                mask = eligible.loc[d] & y.loc[d].notna()
                if mask.sum() < 50:
                    continue
                X, idx = _stack(feats, names, d, mask)
                ok = np.isfinite(X).all(axis=1)
                Xs.append(X[ok])
                ys.append(y.loc[d, idx].to_numpy(dtype=float)[ok])
            if not Xs:
                continue
            model = fit_ridge(np.vstack(Xs), np.concatenate(ys), cfg.ridge)
            coefs.append(pd.Series(model["coef"], index=names, name=day))
            next_refit = i + cfg.refit_every
        mask = eligible.loc[day]
        if mask.sum() < 50:
            continue
        X, idx = _stack(feats, names, day, mask)
        ok = np.isfinite(X).all(axis=1)
        forecasts.loc[day, idx[ok]] = predict(model, X[ok])
    return forecasts, pd.DataFrame(coefs)


def coefficient_table(coefs: pd.DataFrame) -> pd.DataFrame:
    """Mean coefficient over refits, how much it varies, and how often it keeps its sign: what the model actually leans on."""
    t = pd.DataFrame({"mean": coefs.mean(), "std": coefs.std(),
                      "same_sign": (np.sign(coefs).eq(np.sign(coefs.mean()))).mean()})
    return t.reindex(t["mean"].abs().sort_values(ascending=False).index)


def explain(model_coefs: pd.Series, feats: dict, day: pd.Timestamp, symbol: str, top: int = 2) -> str:
    """The features pushing one stock's forecast most against the other stocks that day, on the latest refit's coefficients."""
    def z(n):
        col = feats[n].loc[day].to_numpy(dtype=float)
        sd = np.nanstd(col)
        return (feats[n].loc[day, symbol] - np.nanmean(col)) / (sd if sd > 0 else 1.0)
    contrib = (model_coefs * pd.Series({n: z(n) for n in model_coefs.index}, dtype=float)).dropna()
    return ", ".join(f"{k} {v:+.3f}" for k, v in contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top).items())


# ---------------------------------------------------------------- forecast quality

def information_coefficient(forecasts: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """Spearman rank correlation of forecast and realized return across stocks, one value per decision date."""
    out = {}
    for day, f in forecasts.iterrows():
        pair = pd.DataFrame({"f": f, "y": target.loc[day]}).dropna()
        if len(pair) >= 20:
            out[day] = sps.spearmanr(pair["f"], pair["y"]).statistic
    return pd.Series(out, name="ic", dtype=float)


def newey_west_t(x: pd.Series | np.ndarray, lags: int = 4) -> float:
    """t-statistic of the mean with Newey-West errors."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    s = (e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else float("nan")


def clark_west(small: pd.DataFrame, large: pd.DataFrame, target: pd.DataFrame, eligible: pd.DataFrame) -> pd.Series:
    """Clark-West (2007) adjusted loss difference per decision date, on the target's cross-sectional rank. Positive means the
    larger, nesting model forecasts better after allowing for the noise its extra parameters add under the null."""
    y = xs_rank(target.where(eligible))
    out = {}
    for day in small.index:
        pair = pd.DataFrame({"a": small.loc[day], "b": large.loc[day], "y": y.loc[day]}).dropna()
        if len(pair) >= 20:
            out[day] = float((((pair["y"] - pair["a"]) ** 2) - ((pair["y"] - pair["b"]) ** 2) + ((pair["a"] - pair["b"]) ** 2)).mean())
    return pd.Series(out, name="clark_west", dtype=float)


# ---------------------------------------------------------------- the portfolio

def choose_names(forecast: pd.Series, held: set, sector: pd.Series, cfg: Config) -> list[str]:
    """The names to hold: current holdings still in the best `hold_frac`, topped up from the best forecasts to `top_frac`
    of the eligible stocks, with no sector above `sector_cap` of the names. Ties break by symbol, so reruns agree."""
    ranked = forecast.dropna().sort_index().sort_values(ascending=False, kind="stable")
    n = int(round(cfg.top_frac * len(ranked)))
    if n == 0:
        return []
    keep_zone = set(ranked.index[:int(round(cfg.hold_frac * len(ranked)))])
    limit = max(1, int(np.ceil(cfg.sector_cap * n)))
    chosen, count = [], {}
    candidates = [s for s in ranked.index if s in held and s in keep_zone] + [s for s in ranked.index if s not in held or s not in keep_zone]
    for s in candidates:
        sec = sector.get(s, "Unknown")
        if len(chosen) < n and count.get(sec, 0) < limit:
            chosen.append(s)
            count[sec] = count.get(sec, 0) + 1
    return chosen


def simulate(targets: dict[pd.Timestamp, dict[str, float]], m: Market, cfg: Config, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Run target weights through the market, session by session. A target decided at the close of t trades at t+1's
    execution price; before trading, the holdings have drifted with prices. A stock with no real bar that session cannot
    be traded and stays at its drifted weight. Cost on each stock: (cost_bps + square-root impact) on the amount traded.
    Nothing after `end` is earned. Returns per session: gross and net return, turnover (half the absolute trades), cost,
    holdings count, the largest trade as a share of a stock's daily dollar volume, and the stock that moved the book most."""
    r = asset_returns(m, cfg.execution)
    index, columns = r.index, r.columns
    pos = {s: k for k, s in enumerate(columns)}
    R, tradable = r.to_numpy(), m.real.reindex(index=index, columns=columns).to_numpy()
    adv = adv63(m).reindex(index=index, columns=columns).to_numpy()
    sig = clean_returns(m).rolling(63, min_periods=50).std().reindex(index=index, columns=columns).to_numpy()
    by_exec = {}
    for day, weights in targets.items():
        i = index.get_loc(day)
        if i + 1 < len(index):
            by_exec[i + 1] = (i, weights)
    if not by_exec:
        return pd.DataFrame(columns=["gross", "net", "turnover", "cost", "names", "max_participation", "top_symbol", "top_contribution"], dtype=float)
    first, last = min(by_exec), len(index) if end is None else index.searchsorted(end, side="right")
    w, nav, rows = np.zeros(len(columns)), 1.0, []
    for d in range(first, last):
        moves = w * R[d]
        top = int(np.argmax(np.abs(moves)))
        gross = float(np.nansum(moves))
        if gross > -1:
            w = w * (1.0 + R[d]) / (1.0 + gross)
        turnover = cost = participation = 0.0
        if d in by_exec:
            i, weights = by_exec[d]
            target = np.zeros(len(columns))
            for s, x in weights.items():
                target[pos[s]] = x
            can = tradable[d]
            stuck = np.where(~can, w, 0.0)
            free = np.where(can, target, 0.0)
            room = 1.0 - stuck.sum()
            if free.sum() > 0:
                free = free / free.sum() * room
            new = stuck + free
            trade = np.abs(new - w)
            dollars = trade * cfg.aum * nav
            with np.errstate(divide="ignore", invalid="ignore"):
                part = np.where(trade > 0, dollars / adv[i], 0.0)
            part = np.nan_to_num(part, nan=0.0, posinf=0.0)
            rate = cfg.cost_bps * 1e-4 + cfg.impact * np.nan_to_num(sig[i], nan=0.02) * np.sqrt(part)
            cost = float((trade * rate).sum())
            turnover = 0.5 * float(trade.sum())
            participation = float(part.max()) if len(part) else 0.0
            w = new
        net = (1.0 + gross) * (1.0 - cost) - 1.0
        nav *= 1.0 + net
        rows.append((index[d], gross, net, turnover, cost, int((w > 0).sum()), participation, columns[top], float(moves[top])))
    return pd.DataFrame(rows, columns=["date", "gross", "net", "turnover", "cost", "names", "max_participation", "top_symbol",
                                       "top_contribution"]).set_index("date")


def reconcile_target(targets: dict, m: Market, cfg: Config, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Does the forecast target equal what the book earns? For each decision: the book's gross return from its fill to the next
    fill, run through simulate(), against the same weights on forward_returns(), and against the target an execution one
    session earlier would have needed (`unlagged`). `target` is empty where a name could not fill."""
    free = replace(cfg, cost_bps=0.0, impact=0.0)
    daily = simulate(targets, m, free, end)
    fwd = forward_returns(m, cfg)
    growth = (1.0 + asset_returns(m, cfg.execution)).cumprod()
    index, rows = m.close.index, []
    for day in sorted(targets):
        i = index.get_loc(day)
        a, b = i + 1, i + 1 + cfg.horizon
        if b >= len(index) or index[b] not in daily.index:
            continue
        names = list(targets[day])
        w = np.array([targets[day][s] for s in names])
        book = float((1.0 + daily.loc[index[a + 1]:index[b], "gross"]).prod() - 1.0)
        rows.append({"decision": day, "fill": index[a], "next_fill": index[b], "book": book, "target": float(w @ fwd.loc[day, names].to_numpy(dtype=float)),
                     "unlagged": float(w @ (growth.iloc[b - 1][names].to_numpy() / growth.iloc[a - 1][names].to_numpy() - 1.0))})
    out = pd.DataFrame(rows)
    out["gap"] = out["book"] - out["target"]
    return out


def portfolio_targets(forecasts: pd.DataFrame, m: Market, cfg: Config) -> dict:
    """Equal weights on the chosen names at each decision, given what was held before it."""
    out, held = {}, set()
    for day, f in forecasts.iterrows():
        names = choose_names(f, held, m.sector, cfg)
        if names:
            out[day] = {s: 1.0 / len(names) for s in names}
            held = set(names)
    return out


def equal_weight_targets(eligible: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    """The equal-weight benchmark: every eligible stock, rebalanced on the same schedule and charged the same costs."""
    out = {}
    for day in dates:
        names = eligible.columns[eligible.loc[day].to_numpy()]
        if len(names):
            out[day] = {s: 1.0 / len(names) for s in names}
    return out


# ---------------------------------------------------------------- the scorecard

def performance(returns: pd.Series, rf: pd.Series | None = None, turnover: pd.Series | None = None) -> dict:
    """Annualized figures from daily net returns. Sharpe and Sortino are on returns over `rf` (zero if absent); Sortino's
    denominator is the downside deviation below it over all days."""
    x = returns.dropna()
    if x.empty:
        return {}
    rf = (rf.reindex(x.index).fillna(0.0) if rf is not None else pd.Series(0.0, index=x.index))
    ex = x - rf
    nav = (1.0 + x).cumprod()
    down = np.sqrt((np.minimum(ex, 0.0) ** 2).mean())
    out = {"return_ann": nav.iloc[-1] ** (252 / len(x)) - 1, "vol_ann": x.std() * np.sqrt(252),
           "sharpe": ex.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan,
           "sortino": ex.mean() / down * np.sqrt(252) if down > 0 else np.nan,
           "max_drawdown": float((nav / nav.cummax().clip(lower=1.0) - 1).min())}
    if turnover is not None:
        out["turnover_ann"] = turnover.reindex(x.index).fillna(0.0).sum() * 252 / len(x)
    return out


def window_of(dates: pd.DatetimeIndex, cfg: Config) -> pd.Series:
    v, h = pd.Timestamp(cfg.validation_start), pd.Timestamp(cfg.holdout_start)
    return pd.Series(np.select([dates >= h, dates >= v], ["holdout", "validation"], "development"), index=dates)


def block_bootstrap(x: np.ndarray, stat, block: int = 21, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of stat(x) resampling circular blocks of whole sessions, so serial dependence and paired rows stay."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, int(np.ceil(n / block))))
    offsets = (starts[..., None] + np.arange(block)) % n
    idx = offsets.reshape(n_boot, -1)[:, :n]
    values = np.array([stat(x[i]) for i in idx])
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def _ann_mean(d: np.ndarray) -> float:
    return float(np.nanmean(d) * 252)


def _sharpe_gap(pair: np.ndarray) -> float:
    a, b = pair[:, 0], pair[:, 1]
    return float((b.mean() / b.std() - a.mean() / a.std()) * np.sqrt(252))


def paired_comparison(small: dict, large: dict, target: pd.DataFrame, eligible: pd.DataFrame, rf: pd.Series | None = None,
                      start=None, end=None, seed: int = 0) -> dict:
    """The larger model against the smaller on the same dates in [start, end]: forecast (IC gap, Clark-West) and net of costs."""
    joined = pd.concat([small["daily"]["net"].rename("a"), large["daily"]["net"].rename("b")], axis=1).dropna().loc[start:end]
    if rf is not None:
        joined = joined.sub(rf.reindex(joined.index).fillna(0.0), axis=0)
    ics = pd.concat([small["ic"].rename("a"), large["ic"].rename("b")], axis=1).dropna().loc[start:end]
    d_ic = (ics["b"] - ics["a"]).to_numpy()
    cw = clark_west(small["forecasts"], large["forecasts"], target, eligible).loc[start:end]
    ic_lo, ic_hi = block_bootstrap(d_ic, np.nanmean, block=8, seed=seed)
    ret_lo, ret_hi = block_bootstrap((joined["b"] - joined["a"]).to_numpy(), _ann_mean, seed=seed)
    sh_lo, sh_hi = block_bootstrap(joined[["a", "b"]].to_numpy(), _sharpe_gap, seed=seed)
    return {"ic_small": ics["a"].mean(), "ic_large": ics["b"].mean(), "ic_gap": float(np.nanmean(d_ic)), "ic_gap_lo": ic_lo, "ic_gap_hi": ic_hi,
            "clark_west_t": newey_west_t(cw), "net_return_gap_ann": _ann_mean((joined["b"] - joined["a"]).to_numpy()),
            "net_gap_lo": ret_lo, "net_gap_hi": ret_hi, "sharpe_gap": _sharpe_gap(joined[["a", "b"]].to_numpy()),
            "sharpe_gap_lo": sh_lo, "sharpe_gap_hi": sh_hi, "dates": len(ics)}


def verdict(row: dict) -> str:
    """The rule, fixed before any result. Forecast: Clark-West t at least 1.645 (one-sided 5%) and the IC gap's interval above
    zero. Portfolio: the net-return gap's interval above zero. They are separate claims and can disagree."""
    forecast = row["clark_west_t"] >= 1.645 and row["ic_gap_lo"] > 0
    net = row["net_gap_lo"] > 0
    return f"forecast {'improves' if forecast else 'no improvement shown'}; net of costs {'improves' if net else 'no improvement shown'}"


def run_model(name: str, feats: dict, eligible: pd.DataFrame, target: pd.DataFrame, m: Market, dates: pd.DatetimeIndex, cfg: Config,
              end: pd.Timestamp | None = None) -> dict:
    """One model start to finish: forecasts, the IC, its portfolio through the market up to `end`."""
    forecasts, coefs = walk_forward(feats, eligible, target, MODELS[name], dates, cfg)
    return {"name": name, "forecasts": forecasts, "coefs": coefs, "ic": information_coefficient(forecasts, target),
            "end": end, "daily": simulate(portfolio_targets(forecasts, m, cfg), m, cfg, end)}


def resimulate(result: dict, m: Market, cfg: Config) -> pd.DataFrame:
    """The same forecasts through different costs or execution, for the sensitivity table."""
    return simulate(portfolio_targets(result["forecasts"], m, cfg), m, cfg, result["end"])


def benchmark_returns(bars: pd.DataFrame, symbol: str, execution: str = "next_close", truncated: pd.DatetimeIndex | None = None) -> pd.Series:
    """A benchmark's daily total return (`all`-adjusted bars), between the same execution prices as the portfolios. On the
    vendor's truncated days it has no usable price either, so it earns zero there and the whole move the next session."""
    b = bars[bars["symbol"] == symbol].copy()
    b["date"] = pd.to_datetime(b["date"]).dt.normalize()
    price = b.set_index("date").sort_index()[{"next_close": "close", "next_open": "open"}[execution]]
    if truncated is not None and len(truncated):
        price = price.where(~price.index.isin(truncated)).ffill()
    return price.pct_change().dropna().rename(symbol)


def abdi_ranaldo(raw: pd.DataFrame) -> pd.DataFrame:
    """Each stock-month's bid-ask spread estimated from daily closes, highs and lows (Abdi and Ranaldo 2017), as a share of
    the price: the square root of 4 x the mean of (c_t - m_t)(c_t - m_{t+1}), with m the mid-range, floored at zero."""
    r = raw[(raw["volume"] > 0) & (raw["trade_count"] > 0)].copy()
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()
    r = r.sort_values(["symbol", "date"])
    c, mid = np.log(r["close"]), (np.log(r["high"]) + np.log(r["low"])) / 2.0
    r["s2"] = 4.0 * (c - mid) * (c - mid.groupby(r["symbol"]).shift(-1))
    month = r["date"].dt.to_period("M")
    out = r.groupby(["symbol", month])["s2"].agg(["mean", "count"]).reset_index()
    out["spread"] = np.sqrt(out["mean"].clip(lower=0.0))
    return out[out["count"] >= 15].rename(columns={"date": "month"})[["symbol", "month", "spread"]]


def scorecard(series: dict, rf: pd.Series | None, cfg: Config, ics: dict | None = None) -> pd.DataFrame:
    """Every portfolio and benchmark in every window it has, on the same dates. `series` maps a name to a frame with `net`
    (and `turnover`) or to a plain return series; each is cut to the sessions the first one traded."""
    first = next(iter(series.values()))
    dates = first.index
    labels = window_of(dates, cfg)
    spans = {w: dates[labels == w] for w in ("development", "validation", "holdout") if (labels == w).any()}
    spans["all scored"] = dates
    rows = []
    for w, days in spans.items():
        for name, s in series.items():
            frame = s if isinstance(s, pd.DataFrame) else None
            ret = (frame["net"] if frame is not None else s).reindex(days)
            row = {"window": w, "portfolio": name, "from": days[0].date(), "to": days[-1].date()}
            row.update(performance(ret, rf, frame["turnover"] if frame is not None else None))
            ic = (ics or {}).get(name)
            if ic is not None:
                x = ic.loc[days[0]:days[-1]]
                row.update(ic_mean=x.mean(), ic_t=newey_west_t(x))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- notebook helpers

def perturbed_after(m: Market, day: pd.Timestamp, seed: int = 1) -> Market:
    """The same market with every price and volume after `day` scrambled, for checking that nothing at `day` moves."""
    rng = np.random.default_rng(seed)
    shock = pd.DataFrame(np.exp(rng.normal(0, 0.2, m.close.shape)), index=m.close.index, columns=m.close.columns)
    shock[m.close.index <= day] = 1.0
    return Market(close=m.close * shock, open=m.open * shock, raw_close=m.raw_close * shock, dollar_volume=m.dollar_volume * shock ** 3,
                  real=m.real, event=m.event, sector=m.sector, truncated=m.truncated)


def no_lookahead_check(m: Market, cfg: Config, day: pd.Timestamp) -> dict:
    """Scramble the future and compare what a decision at `day` sees: features, eligibility and the forecast."""
    other = perturbed_after(m, day)
    fa, ea = build_features(m, cfg)
    fb, eb = build_features(other, cfg)
    same = {"eligibility": bool(ea.loc[:day].equals(eb.loc[:day]))}
    same["features"] = all(np.array_equal(fa[n].loc[:day].to_numpy(), fb[n].loc[:day].to_numpy(), equal_nan=True) for n in fa)
    dates = pd.DatetimeIndex([day])
    pa, _ = walk_forward(fa, ea, forward_returns(m, cfg), MODELS["base_pe_x"], dates, cfg)
    pb, _ = walk_forward(fb, eb, forward_returns(other, cfg), MODELS["base_pe_x"], dates, cfg)
    same["forecast"] = bool(np.allclose(pa.to_numpy(), pb.to_numpy(), equal_nan=True, atol=1e-12))
    return same


def load_protocol(path) -> tuple[Config, dict]:
    """A frozen protocol file: the settings it fixes as a Config, and the whole document."""
    import yaml
    with open(path) as f:
        doc = yaml.safe_load(f)
    return Config(**doc["rules"]), doc


def check_benchmarks(bars: pd.DataFrame, reference: dict, truncated: pd.DatetimeIndex | None = None, tolerance: float = 0.005) -> pd.DataFrame:
    """A benchmark's calendar-year total returns against independently recorded ones (`reference`: symbol to year to return)."""
    rows = []
    for symbol, years in reference.items():
        r = benchmark_returns(bars, symbol, truncated=truncated)
        got = (1.0 + r).groupby(r.index.year).prod() - 1.0
        for year, ref in years.items():
            rows.append({"symbol": symbol, "year": int(year), "this_series": float(got.get(int(year), np.nan)), "reference": float(ref)})
    out = pd.DataFrame(rows)
    out["ok"] = (out["this_series"] - out["reference"]).abs() <= tolerance
    return out


def fetch_benchmarks(path, start: str, end: str, symbols=("SPY", "BIL"), client=None) -> pd.DataFrame:
    """SPY (the market) and BIL (Treasury bills, the risk-free proxy): `all`-adjusted daily bars, kept in `path`."""
    from pathlib import Path

    from src.alpaca_rest import AlpacaREST, daily_frame, rfc3339
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    client = client or AlpacaREST()
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    bars = daily_frame(client.bars(list(symbols), "1Day", rfc3339(lo), rfc3339(hi), "all", feed="sip"))
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(path, index=False)
    return bars


def current_portfolio(name: str, feats: dict, eligible: pd.DataFrame, target: pd.DataFrame, m: Market, cfg: Config, held: set,
                      fit_until: pd.Timestamp | None = None) -> pd.DataFrame:
    """The portfolio the tool would hold from the latest data: names, weights, forecasts, the features behind each, and what
    changes from `held`. It reports no performance. `fit_until` keeps the model off a locked hold-out."""
    day = eligible.index[eligible.sum(axis=1) >= 50][-1]
    forecasts, coefs = walk_forward(feats, eligible, target, MODELS[name], pd.DatetimeIndex([day]), cfg, fit_until)
    f = forecasts.loc[day]
    names = choose_names(f, held, m.sector, cfg)
    return pd.DataFrame({"symbol": names, "date": day, "sector": [m.sector[s] for s in names], "weight": 1.0 / len(names),
                         "forecast": f[names].to_numpy(), "action": ["hold" if s in held else "buy" for s in names],
                         "why": [explain(coefs.iloc[-1], feats, day, s) for s in names]})
