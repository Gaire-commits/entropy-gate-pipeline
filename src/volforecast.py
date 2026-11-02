"""Does permutation entropy of daily returns improve next-day variance forecasts beyond a HAR baseline?

One registered hypothesis (protocols/volatility_v1.yaml): the baseline against the same baseline plus PE at the origin, on
identical rows and folds, judged by Clark-West on the log of next session's variance. The baseline's form is chosen on the
development window and confirmed on validation without any PE; the comparison runs once, on the reserved test window.

A forecast at the close of t uses data through t. A model refit at the close of T trains only on rows whose target session
is at or before T. The panel comes from src/volpanel.py; nothing past its `end` is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import audit, nyse, portfolio as pf, volpanel


@dataclass(frozen=True)
class VolConfig:
    pe_m: int = 3
    pe_tau: int = 1
    pe_window: int = 63             # patterns, so 65 returns
    warmup: int = 65                # sessions that must mostly hold valid values, the PE window plus its embedding
    floor: float = 1e-8             # (1 bp)^2: a session whose variance is exactly zero has no logarithm
    min_train_sessions: int = 252   # sessions of history before the first forecast
    refit_every: int = 21
    long_window: int = 252
    long_min: int = 63
    validation_start: str = "2022-01-04"
    test_start: str = "2024-01-03"
    test_end: str = "2026-10-12"
    seed: int = 0


LOG_HAR = ["d", "w", "m"]
# In order of simplicity: a tie within 0.5% of the best development QLIKE goes to the earliest.
CANDIDATES = {"log_har": {"names": LOG_HAR}, "log_har_long": {"names": LOG_HAR + ["long"]},
              "log_har_per_stock": {"names": LOG_HAR, "per_stock": True}, "level_har": {"names": ["vd", "vw", "vm"], "level": True}}


def lock_end(cfg: VolConfig, score_reserved: bool = False) -> pd.Timestamp:
    """The last session the panel may hold: the test window's last unless it is being scored, else the day before it opens."""
    return pd.Timestamp(cfg.test_end) if score_reserved else pd.Timestamp(cfg.test_start) - pd.Timedelta(days=1)


def load_panel(sessions_path, reference, end, official_path=None, root: Path = Path(".")) -> pd.DataFrame:
    """The panel from the audit's sessions and the reference data, with nothing after `end` in memory."""
    end = pd.Timestamp(end)

    def upto(frame):
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        return frame[frame["date"] <= end]
    sessions = upto(pd.read_parquet(sessions_path, columns=["symbol", "date", "bars", "slots", "high", "low", "dv_trailing63"]))
    daily = {t: upto(pd.read_parquet(Path(reference) / "daily" / f"daily_sip_{t}.parquet")) for t in ("raw", "split", "all")}
    vendor = volpanel.vendor_frame(daily["raw"], daily["split"], daily["all"])
    actions = audit.normalize_actions(pd.read_parquet(Path(reference) / "corporate_actions.parquet"))
    registry = volpanel.load_registry(root / "registry" / "vol_registry.csv")
    symbols = sorted(pd.read_csv(root / "universe" / "sp500.csv")["symbol"])
    official = upto(pd.read_parquet(official_path)) if official_path else None
    return volpanel.build_panel(sessions, vendor, actions, registry, symbols, volpanel.Rules(), end=end, official=official)


def build_rows(p: pd.DataFrame, cfg: VolConfig, joined: pd.Series | None = None, col: str = "V") -> pd.DataFrame:
    """One row per eligible forecast origin t with a full-day target: the HAR inputs of `col` through t, PE through t, and the
    target at t+1. Half-day targets are left out of fitting and scoring (half-days stay in the inputs). Rows need a PE
    value whichever model reads them, so every model is fitted and scored on the same rows."""
    q = volpanel.add_eligibility(p, volpanel.Rules(), cfg.warmup, joined)
    key = [q["symbol"], q["segment"]]
    x = q[col].clip(lower=cfg.floor)

    def roll(series, window, least, fn="mean"):
        return getattr(series.groupby(key, sort=False).rolling(window, min_periods=least), fn)().reset_index(level=[0, 1], drop=True).sort_index()
    q["vd"], q["vw"], q["vm"] = x, roll(x, 5, 4), roll(x, 22, 18)
    q["d"], q["w"], q["m"] = np.log(q["vd"]), np.log(q["vw"]), np.log(q["vm"])
    q["long"] = roll(np.log(x), cfg.long_window, cfg.long_min)
    wide = q.pivot(index="date", columns="symbol", values="ret")
    q = q.merge(pf.rolling_pe(wide, cfg.pe_m, cfg.pe_tau, cfg.pe_window).stack().rename("pe").reset_index(), on=["date", "symbol"], how="left")
    g = q.groupby("symbol", sort=False)
    q["next"] = g[col].shift(-1).clip(lower=cfg.floor)
    keep = q["eligible"] & q["next"].notna() & q["pe"].notna() & ~q["target_half_day"].eq(True)
    rows = q.loc[keep, ["symbol", "date", "target_date", "d", "w", "m", "long", "vd", "vw", "vm", "pe", "next", "open_source"]].copy()
    rows["y"] = np.log(rows["next"])
    return rows.dropna(subset=["d", "w", "m", "long", "vd", "vw", "vm"]).sort_values(["date", "symbol"]).reset_index(drop=True)


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta, float(np.var(y - A @ beta))


def walk_forward(rows: pd.DataFrame, names: list[str], cfg: VolConfig, first, last, per_stock: bool = False, level: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecasts for origins whose target lies in [first, last], refit every `refit_every` sessions on every row realized by
    then. `f` forecasts log variance; `h` forecasts the level (log models add half the residual variance). Returns the
    forecasts and each pooled refit's coefficients."""
    origins = pd.DatetimeIndex(np.sort(rows["date"].unique()))
    start_at = origins.searchsorted(origins[0]) + cfg.min_train_sessions
    in_span = rows["target_date"].between(pd.Timestamp(first), pd.Timestamp(last))
    wanted = origins[(origins >= origins[min(start_at, len(origins) - 1)]) & origins.isin(rows.loc[in_span, "date"])]
    out, coefs = [], []
    y_all = rows["next"].to_numpy() if level else rows["y"].to_numpy()
    X_all = rows[names].to_numpy(dtype=float)
    dates, targets = rows["date"].to_numpy(), rows["target_date"].to_numpy()
    span = in_span.to_numpy()
    for k in range(0, len(wanted), cfg.refit_every):
        T = wanted[k]
        block_end = wanted[min(k + cfg.refit_every, len(wanted)) - 1]
        tr = targets <= np.datetime64(T)
        te = (dates >= np.datetime64(T)) & (dates <= np.datetime64(block_end)) & span
        if tr.sum() < 100 or not te.any():
            continue
        beta, s2 = _ols(X_all[tr], y_all[tr])
        coefs.append(pd.Series(beta[1:], index=names, name=T))
        pred = beta[0] + X_all[te] @ beta[1:]
        res = rows.loc[te, ["symbol", "date", "target_date", "y", "next"]].copy()
        if per_stock:
            sym = rows["symbol"].to_numpy()
            frame = pd.DataFrame({"sym": sym[te], "pred": pred}, index=np.flatnonzero(te))
            for s, idx in frame.groupby("sym").groups.items():
                own = tr & (sym == s)
                if own.sum() >= 126:
                    b, v = _ols(X_all[own], y_all[own])
                    frame.loc[idx, "pred"] = b[0] + X_all[idx] @ b[1:]
            pred = frame["pred"].to_numpy()
        if level:
            res["h"] = np.maximum(pred, cfg.floor)
            res["f"] = np.log(res["h"])
        else:
            res["f"] = pred
            res["h"] = np.exp(pred + s2 / 2.0)
        out.append(res)
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["symbol", "date", "target_date", "y", "next", "f", "h"])), pd.DataFrame(coefs)


def naive_forecast(rows: pd.DataFrame, first, last) -> pd.DataFrame:
    """The monthly mean carried forward: what any model has to beat."""
    res = rows.loc[rows["target_date"].between(pd.Timestamp(first), pd.Timestamp(last)), ["symbol", "date", "target_date", "y", "next", "m", "vm"]].copy()
    return res.rename(columns={"m": "f", "vm": "h"})


def daily_loss(fc: pd.DataFrame) -> pd.DataFrame:
    """Per target date, the mean across stocks: squared error on log variance, QLIKE, and squared error on the level."""
    ratio = fc["next"] / fc["h"]
    return pd.DataFrame({"target_date": fc["target_date"], "mse_log": (fc["y"] - fc["f"]) ** 2, "qlike": ratio - np.log(ratio) - 1.0,
                         "mse_level": (fc["next"] - fc["h"]) ** 2}).groupby("target_date").mean()


def choose_baseline(losses: dict[str, float], tie: float = 0.005) -> str:
    """The lowest development QLIKE; within `tie` of it, the simplest (the earliest in CANDIDATES)."""
    best = min(losses.values())
    return next(n for n in CANDIDATES if n in losses and losses[n] <= best * (1 + tie))


def gap(small: pd.Series, large: pd.Series, seed: int = 0, block: int = 21) -> dict:
    """The larger model's loss minus the smaller's, per date: the mean, its 95% block-bootstrap interval and its t. Negative
    means the larger model forecasts better."""
    d = (large - small).dropna()
    lo, hi = pf.block_bootstrap(d.to_numpy(), np.mean, block=block, seed=seed)
    return {"gap": float(d.mean()), "lo": lo, "hi": hi, "t": pf.newey_west_t(d, lags=5), "dates": int(len(d))}


def clark_west(small: pd.DataFrame, large: pd.DataFrame) -> pd.Series:
    """Clark-West (2007) adjusted loss difference on log variance, per target date. Positive favours the larger, nesting model."""
    m = small[["symbol", "date", "target_date", "y", "f"]].merge(large[["symbol", "date", "f"]], on=["symbol", "date"], suffixes=("_0", "_1"))
    m["cw"] = (m["y"] - m["f_0"]) ** 2 - (m["y"] - m["f_1"]) ** 2 + (m["f_0"] - m["f_1"]) ** 2
    return m.groupby("target_date")["cw"].mean()


def pe_test(rows: pd.DataFrame, baseline: str, cfg: VolConfig, first, last) -> dict:
    """The one registered comparison: the baseline against the baseline plus PE, same rows, same folds."""
    spec = CANDIDATES[baseline]
    small, _ = walk_forward(rows, spec["names"], cfg, first, last, spec.get("per_stock", False), spec.get("level", False))
    # PE enters on the log scale's linear model whichever baseline won, standardized by nothing: it already lies in [0, 1]
    large, coefs = walk_forward(rows, spec["names"] + ["pe"], cfg, first, last, spec.get("per_stock", False), spec.get("level", False))
    keys = ["symbol", "date"]
    if not small[keys].equals(large[keys]):
        raise ValueError("the two models were scored on different rows")
    cw = clark_west(small, large)
    a, b = daily_loss(small), daily_loss(large)
    pe = coefs["pe"]
    years = {}
    for y, c in cw.groupby(cw.index.year):
        years[int(y)] = {"cw_mean": float(c.mean()), "cw_t": pf.newey_west_t(c, lags=5), "mse_log_gap": float((b["mse_log"] - a["mse_log"]).loc[c.index].mean())}
    return {"baseline": baseline, "rows": int(len(small)), "dates": int(len(cw)), "first": str(cw.index.min().date()), "last": str(cw.index.max().date()),
            "clark_west_mean": float(cw.mean()), "clark_west_t": pf.newey_west_t(cw, lags=5), "mse_log": gap(a["mse_log"], b["mse_log"], cfg.seed), "qlike": gap(a["qlike"], b["qlike"], cfg.seed),
            "mse_level": gap(a["mse_level"], b["mse_level"], cfg.seed), "baseline_mse_log": float(a["mse_log"].mean()), "baseline_qlike": float(a["qlike"].mean()),
            "pe_coef_mean": float(pe.mean()), "pe_coef_same_sign": float((np.sign(pe) == np.sign(pe.mean())).mean()), "by_year": years}


def verdict(result: dict) -> str:
    """The rule, fixed before the comparison ran: Clark-West t at least 1.645 (one-sided 5%). The QLIKE gap's interval says
    whether the result holds under a loss robust to a noisy variance proxy."""
    primary = result["clark_west_t"] >= 1.645
    robust = result["qlike"]["hi"] < 0
    if not primary:
        return "PE: no improvement shown"
    return "PE improves the forecast of log variance" + ("; QLIKE agrees" if robust else "; QLIKE does not confirm it")


def realized_variance(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Per session: the sum of squared 5-minute log returns inside the session (09:30 to the close, 13:00 on a half-day), the
    first from the first bar's open. Empty unless at least 90% of the session's slots hold a bar. Bars as audit.read_bars gives them."""
    ts = bars.index
    day = ts.normalize()
    tod = ts - day
    close_t = pd.TimedeltaIndex(np.where(np.asarray(day.isin(nyse.HALF_DAYS)), nyse.EARLY_CLOSE.to_timedelta64(), nyse.FULL_CLOSE.to_timedelta64()))
    inside = (tod >= nyse.OPEN) & (tod < close_t)
    f = pd.DataFrame({"day": day[inside], "o": bars["open"].to_numpy(float)[inside], "c": bars["close"].to_numpy(float)[inside]})
    f = f[(f["o"] > 0) & (f["c"] > 0)]
    prev = f.groupby("day")["c"].shift(1)
    f["r"] = np.log(f["c"] / prev.fillna(f["o"]))
    out = f.groupby("day")["r"].agg(rv=lambda r: float((r ** 2).sum()), bars="size").reset_index().rename(columns={"day": "date"})
    out["slots"] = np.asarray(nyse.slots(pd.DatetimeIndex(out["date"])))
    out.loc[out["bars"] < 0.9 * out["slots"], "rv"] = np.nan
    out["symbol"] = symbol
    return out[["symbol", "date", "rv", "bars", "slots"]]


def sample_sizes(rows: pd.DataFrame, cfg: VolConfig) -> pd.DataFrame:
    """Rows and stocks per window, by target date. Counts only."""
    windows = {"development": ("2016-01-01", pd.Timestamp(cfg.validation_start) - pd.Timedelta(days=4)),
               "validation": (cfg.validation_start, pd.Timestamp(cfg.test_start) - pd.Timedelta(days=4)), "test": (cfg.test_start, cfg.test_end)}
    out = []
    for name, (a, b) in windows.items():
        x = rows[rows["target_date"].between(pd.Timestamp(a), pd.Timestamp(b))]
        if len(x):
            out.append({"window": name, "first_target": x["target_date"].min().date(), "last_target": x["target_date"].max().date(), "sessions": x["target_date"].nunique(),
                        "stocks": x["symbol"].nunique(), "rows": len(x), "official_open_share": float((x["open_source"] == "official").mean())})
    return pd.DataFrame(out)
