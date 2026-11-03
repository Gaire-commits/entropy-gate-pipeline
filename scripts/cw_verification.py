#!/usr/bin/env python3
"""Is Clark-West inference appropriate for the pooled models and their refitting schemes, with dependence across dates accounted for?

    python scripts/cw_verification.py --sessions sessions.parquet --reference data/reference --reps 300

The literature leaves this open for our scheme. Clark and West (2007) argue that standard normal critical values give a size close to, and a
little under, nominal, but concede the statistic is not asymptotically normal under their preferred conditions, and rolling schemes were better
sized than recursive ones. Ours is pooled across stocks, recursive (expanding), and refit every 21 sessions. So this measures the size directly,
by the false-alarm rate of a predictor known to have no incremental information:

  A. Simulated panels with a known data-generating process: a common volatility factor, stock-level persistence, heavy-tailed proxy noise and
     common shocks. Null scenarios, and one with a small true effect, for power.
  B. The real target, up to the lock, with placebo predictors built from simulated returns. y keeps every real dependence; the placebo cannot
     be related to it. Scored on the last 690 target dates before the lock, the length of the reserved period, with the same training
     history before them. No PE of any real return series is used anywhere.

Dependence across stocks is handled by averaging within each date; dependence across dates by Newey-West on that series of date means. The
rule, fixed before this was run: the Newey-West lag to use is the smallest of the lags tried whose false-alarm rate at the nominal 5% is at most 7.5% in
every null scenario, simulated and real. If no lag qualifies, Clark-West t with normal critical values is not adequate on its own, and a
critical value calibrated from placebos replaces 1.645.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import ROOT

from src import portfolio as pf, volforecast as vf

VOL_LAGS = (5, 10, 21, 42)
PORT_LAGS = (2, 4, 9, 18)


def nw_t(x: np.ndarray, lags: int) -> float:
    return pf.newey_west_t(x, lags=lags)


def _moments(M: np.ndarray, key: np.ndarray, n_keys: int) -> np.ndarray:
    """Per key, the sum of M'M over its rows, as (n_keys, k, k)."""
    k = M.shape[1]
    out = np.zeros((n_keys, k, k))
    for a in range(k):
        for b in range(a, k):
            out[:, a, b] = out[:, b, a] = np.bincount(key, M[:, a] * M[:, b], minlength=n_keys)
    return out


def recursive_forecasts(y, X, z, realized, origin, blocks, ridge: float = 0.0):
    """The two nested fits, refit at each block's date on every row realized by then, from moments so that many placebos are cheap. `realized`
    and `origin` are integer dates per row; `blocks` lists (fit_through, first_origin, last_origin). Returns (f0, f1, scored). Matches
    volforecast.walk_forward at ridge 0 and portfolio.walk_forward with the ridge it uses (see the tests)."""
    n_keys = int(max(realized.max(), origin.max())) + 1
    one = np.ones(len(y))
    M = np.column_stack([one, X, z, y])
    G = np.cumsum(_moments(M, realized, n_keys), axis=0)
    k = X.shape[1]
    f0, f1 = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for fit_through, a, b in blocks:
        m = G[fit_through]
        n = m[0, 0]
        if n < 100:
            continue
        mean = m[0] / n
        cov = m / n - np.outer(mean, mean)
        std = np.sqrt(np.maximum(np.diag(cov), 1e-300))
        corr = cov / np.outer(std, std)
        te = (origin >= a) & (origin <= b)
        for size, out in ((k, f0), (k + 1, f1)):
            pick = np.arange(1, size + 1)
            beta = np.linalg.solve(corr[np.ix_(pick, pick)] + ridge * np.eye(size), corr[pick, k + 2])
            Zt = (M[te][:, pick] - mean[pick]) / std[pick]
            out[te] = mean[k + 2] + std[k + 2] * (Zt @ beta)
    return f0, f1, np.isfinite(f0)


def date_series(y, f0, f1, key, scored):
    """Per target date, the mean across stocks of the adjusted Clark-West difference and of the raw gain."""
    y, f0, f1, key = y[scored], f0[scored], f1[scored], key[scored]
    uniq, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv)
    gain = (y - f0) ** 2 - (y - f1) ** 2
    cw = gain + (f0 - f1) ** 2
    return np.bincount(inv, cw) / n, np.bincount(inv, gain) / n, cw, gain


def rates(rows: list[dict], lags) -> dict:
    """False alarms at nominal 5%, one-sided, for each lag; the same ignoring all dependence (a stock-day t); how often Clark-West passes
    while the raw loss did not demonstrably fall; the mean t."""
    t = pd.DataFrame(rows)
    out = {f"lag {l}": float((t[f"t{l}"] >= 1.645).mean()) for l in lags}
    out["ignoring dependence"] = float((t["naive"] >= 1.645).mean())
    out["passes, raw loss not shown lower"] = float(((t[f"t{lags[2]}"] >= 1.645) & (t[f"raw{lags[2]}"] < 1.96)).mean())
    out[f"mean t, lag {lags[2]}"] = float(t[f"t{lags[2]}"].mean())
    return out


def stats(y, f0, f1, key, scored, lags) -> dict:
    a, g, cw, gain = date_series(y, f0, f1, key, scored)
    out = {f"t{l}": nw_t(a, l) for l in lags}
    out.update({f"raw{l}": nw_t(g, l) for l in lags})
    out["naive"] = float(cw.mean() / (cw.std() / np.sqrt(len(cw))))
    return out


def simulated_panel(rng, N=150, T=1800, effect=0.0, common=True, heavy=True, correlated=False, common_z=False):
    """Log variance with a common factor and stock-level persistence, a noisy proxy of it, its HAR inputs, and a placebo (PE of unrelated returns)."""
    def ar(rho, shape):
        e = rng.normal(size=shape)
        x = np.zeros(shape)
        for t in range(1, shape[0]):
            x[t] = rho * x[t - 1] + e[t]
        return x * np.sqrt(1 - rho ** 2)
    c = (0.5 * ar(0.5, (T, 1)) + 0.5 * ar(0.95, (T, 1)) + 0.7 * ar(0.995, (T, 1))) if common else np.zeros((T, 1))
    s = -8.8 + 0.8 * c + 0.5 * ar(0.97, (T, N)) + rng.normal(0, 0.4, (1, N))
    noise = rng.normal(0, 0.7, (T, N))
    if heavy:
        noise = noise + 0.25 * rng.standard_t(3, (T, N)) + (0.35 * rng.standard_t(3, (T, 1)) if common else 0.0)
    placebo = pf.rolling_pe(pd.DataFrame(rng.normal(size=(T, N))), 3, 1, 63).to_numpy()
    placebo = placebo - np.nanmean(placebo)
    if common_z:
        placebo = placebo + np.nanstd(placebo) * ar(0.98, (T, 1))
    ln_v = s + noise
    if effect:
        ln_v[1:] += effect * np.nan_to_num(placebo[:-1])
    V = np.exp(ln_v)
    frame = pd.DataFrame(V)
    d = ln_v
    w = np.log(frame.rolling(5).mean().to_numpy())
    m = np.log(frame.rolling(22).mean().to_numpy())
    if correlated:
        placebo = placebo + 0.02 * np.nan_to_num(m - np.nanmean(m))
    ok = np.isfinite(placebo[:-1]) & np.isfinite(m[:-1])
    t_idx, n_idx = np.nonzero(ok)
    return ln_v[t_idx + 1, n_idx], np.column_stack([d[t_idx, n_idx], w[t_idx, n_idx], m[t_idx, n_idx]]), placebo[t_idx, n_idx], t_idx, T


def blocks_for(first_origin, last_origin, refit):
    return [(a, a, min(a + refit - 1, last_origin)) for a in range(first_origin, last_origin + 1, refit)]


def rolling_forecasts(y, X, z, origin, first_origin, last_origin, refit, window):
    f0, f1 = np.full(len(y), np.nan), np.full(len(y), np.nan)
    M0 = np.column_stack([np.ones(len(y)), X])
    M1 = np.column_stack([M0, z])
    for a, b in ((a, min(a + refit - 1, last_origin)) for a in range(first_origin, last_origin + 1, refit)):
        tr = (origin + 1 <= a) & (origin + 1 > a - window)
        te = (origin >= a) & (origin <= b)
        for A, out in ((M0, f0), (M1, f1)):
            beta, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
            out[te] = A[te] @ beta
    return f0, f1, np.isfinite(f0)


def part_a(reps: int, seed: int) -> pd.DataFrame:
    scenarios = {"null: independent stocks, normal noise": dict(common=False, heavy=False),
                 "null: common factor, heavy tails": dict(),
                 "null: + placebo has a slow common swing": dict(common_z=True),
                 "null: + placebo correlated with the inputs": dict(common_z=True, correlated=True),
                 "true effect, small": dict(effect=2.0),
                 "true effect, moderate": dict(effect=5.0)}
    out = {}
    for name, kw in scenarios.items():
        rng = np.random.default_rng(seed)
        rows, rows1, rows_roll = [], [], []
        for r in range(reps):
            y, X, z, t, T = simulated_panel(rng, **kw)
            first, last = T - 691, T - 2
            f0, f1, sc = recursive_forecasts(y, X, z[:, None][:, 0], t + 1, t, blocks_for(first, last, 21))
            rows.append(stats(y, f0, f1, t + 1, sc, VOL_LAGS))
            if name == "null: common factor, heavy tails" and r < reps // 3:
                f0, f1, sc = recursive_forecasts(y, X, z, t + 1, t, blocks_for(first, last, 1))
                rows1.append(stats(y, f0, f1, t + 1, sc, VOL_LAGS))
                f0, f1, sc = rolling_forecasts(y, X, z, t, first, last, 21, 504)
                rows_roll.append(stats(y, f0, f1, t + 1, sc, VOL_LAGS))
        out[name] = rates(rows, VOL_LAGS)
        if rows1:
            out["  same, refit every session"] = rates(rows1, VOL_LAGS)
            out["  same, rolling 504-session window"] = rates(rows_roll, VOL_LAGS)
    return pd.DataFrame(out).T


def placebo_pool(rng, shape, size: int = 24) -> list[np.ndarray]:
    return [pf.rolling_pe(pd.DataFrame(rng.normal(size=shape)), 3, 1, 63).to_numpy() for _ in range(size)]


def draw(rng, pool, T, N, kind, base=None):
    """A placebo panel from the pool: a random stock order and a circular shift in time, so draws differ cheaply."""
    z = pool[rng.integers(len(pool))]
    z = np.roll(z[:, rng.permutation(z.shape[1])], rng.integers(T), axis=0)
    z = np.nan_to_num(z - np.nanmean(z))
    if kind != "own swings only":
        e = rng.normal(size=T)
        g = np.zeros(T)
        for t in range(1, T):
            g[t] = 0.98 * g[t - 1] + np.sqrt(1 - 0.98 ** 2) * e[t]
        z = z + z.std() * g[:, None]
    return z


def part_b(args, reps: int, seed: int) -> pd.DataFrame:
    cfg = vf.VolConfig()
    p = vf.load_panel(args.sessions, args.reference, vf.lock_end(cfg), None, ROOT)
    rows = vf.build_rows(p, cfg, pf.load_joined(ROOT / "registry" / "sp500_join_dates.csv"))
    dates = pd.DatetimeIndex(np.sort(pd.concat([rows["date"], rows["target_date"]]).unique()))
    symbols = pd.Index(sorted(rows["symbol"].unique()))
    origin = dates.get_indexer(rows["date"])
    realized = dates.get_indexer(rows["target_date"])
    col = symbols.get_indexer(rows["symbol"])
    y = rows["y"].to_numpy()
    X = rows[vf.LOG_HAR].to_numpy()
    targets = np.unique(realized)
    first_target = targets[-690]
    first, last = origin[realized >= first_target].min(), origin.max()
    print(f"real target, up to {dates.max().date()}: {len(y):,} rows; scored target dates {dates[first_target].date()} to {dates[realized.max()].date()} ({int((targets >= first_target).sum())} dates)")
    blocks = [(a, a, min(a + 20, last)) for a in sorted(np.unique(origin[(origin >= first) & (origin <= last)]))[::21]]
    rng = np.random.default_rng(seed)
    T, N = len(dates), len(symbols)
    pool = placebo_pool(rng, (T, N))
    out = {}
    for name, kind, tilt in (("real y: placebo with its own swings only", "own swings only", 0.0), ("real y: placebo with a slow common swing", "common", 0.0), ("real y: + correlated with the inputs", "common", 0.02)):
        res = []
        for _ in range(reps):
            z = draw(rng, pool, T, N, kind)[origin, col]
            if tilt:
                z = z + tilt * (rows["m"].to_numpy() - rows["m"].mean())
            f0, f1, sc = recursive_forecasts(y, X, z, realized, origin, blocks)
            res.append(stats(y, f0, f1, realized, sc, VOL_LAGS))
        out[name] = rates(res, VOL_LAGS)
    return pd.DataFrame(out).T


def part_c(args, reps: int, seed: int) -> pd.DataFrame:
    """The portfolio's comparison: ridge, rank target, 21-session targets at 7-session decisions, refit every 63 sessions, on the last 98 decisions before the lock."""
    cfg = pf.Config(rebalance_every=21, horizon=21, tranches=3, top_frac=0.2, hold_frac=0.4)
    import portfolio_rules as rules
    m = rules.load_market(args.reference, cfg)
    index = m.close.index
    feats, eligible = pf.build_features(m, cfg)
    target = pf.xs_rank(pf.forward_returns(m, cfg).where(eligible))
    decisions = pf.decision_dates(index, cfg)
    train_days = index[252::cfg.train_sample_every]
    symbols = m.close.columns
    def stack(days, realize):
        out = []
        for d in days:
            mask = eligible.loc[d] & target.loc[d].notna()
            idx = np.flatnonzero(mask.to_numpy())
            if len(idx) < 50:
                continue
            X = np.column_stack([feats[n].loc[d].to_numpy(dtype=float)[idx] for n in pf.BASE])
            ok = np.isfinite(X).all(axis=1)
            out.append((X[ok], target.loc[d].to_numpy()[idx][ok], np.full(ok.sum(), index.get_loc(d)), idx[ok]))
        return tuple(np.concatenate([o[k] for o in out]) for k in range(4))
    Xt, yt, ot, st = stack(train_days, True)
    Xs, ys, os_, ss = stack(decisions, False)
    print(f"portfolio: {len(yt):,} training rows, {len(ys):,} scored rows over {len(np.unique(os_))} decisions; scoring the last 98 decisions (the reserved length)")
    rng = np.random.default_rng(seed)
    T, N = len(index), len(symbols)
    pool = placebo_pool(rng, (T, N), size=12)
    decision_pos = np.unique(os_)
    fit_points = decision_pos[::9]
    score_from = decision_pos[-98]
    def one(z_panel):
        zt, zs = z_panel[ot, st], z_panel[os_, ss]
        f0s, f1s = [], []
        n = len(yt)
        M = np.column_stack([np.ones(n), Xt, zt, yt])
        G = np.cumsum(_moments(M, ot, T), axis=0)
        k = Xt.shape[1]
        F0, F1 = np.full(len(ys), np.nan), np.full(len(ys), np.nan)
        for j, fp in enumerate(fit_points):
            nxt = fit_points[j + 1] if j + 1 < len(fit_points) else T
            te = (os_ >= fp) & (os_ < nxt)
            if not te.any() or nxt <= score_from:
                continue
            g = G[fp - 1 - cfg.horizon]
            nn = g[0, 0]
            mean = g[0] / nn
            cov = g / nn - np.outer(mean, mean)
            std = np.sqrt(np.maximum(np.diag(cov), 1e-300))
            corr = cov / np.outer(std, std)
            for size, out in ((k, F0), (k + 1, F1)):
                pick = np.arange(1, size + 1)
                beta = np.linalg.solve(corr[np.ix_(pick, pick)] + cfg.ridge * np.eye(size), corr[pick, k + 2])
                A = np.column_stack([Xs[te], zs[te]])[:, :size]
                out[te] = mean[k + 2] + std[k + 2] * (((A - mean[pick]) / std[pick]) @ beta)
        sc = np.isfinite(F0) & (os_ >= score_from)
        return stats(ys, F0, F1, os_, sc, PORT_LAGS)
    out = {}
    for name, kind in (("portfolio: placebo rank, own swings only", "own swings only"), ("portfolio: placebo with a slow common swing", "common")):
        res = []
        for _ in range(reps):
            z = pool[rng.integers(len(pool))]
            z = np.roll(z[:, rng.permutation(N)], rng.integers(T), axis=0)
            panel = pd.DataFrame(z, index=index, columns=symbols).where(eligible)
            if kind == "common":
                # a market-wide swing cannot survive a cross-sectional rank, so it enters as a slow change in which stocks rank high
                loading = rng.normal(size=N)
                e = rng.normal(size=T)
                g = np.zeros(T)
                for t in range(1, T):
                    g[t] = 0.98 * g[t - 1] + np.sqrt(1 - 0.98 ** 2) * e[t]
                panel = panel + float(np.nanstd(z)) * np.outer(g, loading)
            res.append(one(np.nan_to_num(pf.xs_rank(panel).to_numpy())))
        out[name] = rates(res, PORT_LAGS)
    return pd.DataFrame(out).T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", default="abc")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    t0 = time.time()
    if "a" in args.parts:
        print(f"A. Simulated panels, 150 stocks, 1,800 sessions, last 690 scored; {args.reps} draws each. Share of draws passing at nominal 5%:")
        print(part_a(args.reps, args.seed).to_string(), f"\n({time.time() - t0:.0f}s)\n", flush=True)
    if "b" in args.parts:
        print(f"B. The real target with placebo predictors, recursive, refit every 21 sessions; {args.reps} draws each. Share of draws passing at nominal 5%:")
        print(part_b(args, args.reps, args.seed).to_string(), f"\n({time.time() - t0:.0f}s)\n", flush=True)
    if "c" in args.parts:
        print(f"C. The portfolio's comparison, real target, placebo predictors; {args.reps} draws each. Lags in decisions:")
        print(part_c(args, args.reps, args.seed).to_string(), f"\n({time.time() - t0:.0f}s)\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
