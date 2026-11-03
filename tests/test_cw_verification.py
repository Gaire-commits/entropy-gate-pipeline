"""The verification's fast engine must be the production fit, or its false-alarm rates say nothing about the protocols."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import cw_verification as cw
from src import nyse, volforecast as vf


def test_the_fast_engine_reproduces_the_volatility_walk_forward():
    days = nyse.sessions("2020-01-02", "2021-06-30")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        x = rng.normal(size=(len(days), 4))
        y = x[:, 0] * 0.5 + rng.normal(size=len(days))
        rows.append(pd.DataFrame({"symbol": f"S{i}", "date": days, "target_date": list(days[1:]) + [days[-1] + pd.Timedelta(days=1)], "d": x[:, 0], "w": x[:, 1], "m": x[:, 2], "pe": x[:, 3], "y": y, "next": np.exp(y)}))
    r = pd.concat(rows).iloc[:].sort_values(["date", "symbol"]).reset_index(drop=True)
    r = r[r["date"] < days[-1]].reset_index(drop=True)
    cfg = vf.VolConfig(min_train_sessions=120, refit_every=21)
    first, last = days[200], days[-1]
    small, _ = vf.walk_forward(r, vf.LOG_HAR, cfg, first, last)
    large, _ = vf.walk_forward(r, vf.LOG_HAR + ["pe"], cfg, first, last)
    dates = pd.DatetimeIndex(np.sort(pd.concat([r["date"], r["target_date"]]).unique()))
    origin, realized = dates.get_indexer(r["date"]), dates.get_indexer(r["target_date"])
    wanted = np.unique(origin[dates[realized] >= first])
    wanted = wanted[wanted >= origin.min() + cfg.min_train_sessions]
    blocks = [(a, a, wanted[min(k + 21, len(wanted)) - 1]) for k, a in zip(range(0, len(wanted), 21), wanted[::21])]
    f0, f1, sc = cw.recursive_forecasts(r["y"].to_numpy(), r[vf.LOG_HAR].to_numpy(), r["pe"].to_numpy(), realized, origin, blocks)
    fast = r.loc[sc, ["symbol", "date"]].assign(f0=f0[sc], f1=f1[sc])
    m = small[["symbol", "date", "f"]].merge(large[["symbol", "date", "f"]], on=["symbol", "date"], suffixes=("_s", "_l")).merge(fast, on=["symbol", "date"])
    assert len(m) == len(small) > 1000
    assert np.allclose(m["f_s"], m["f0"]) and np.allclose(m["f_l"], m["f1"])


def test_the_statistic_rejects_a_real_effect_and_rarely_a_placebo():
    rng = np.random.default_rng(1)
    y, X, z, t, T = cw.simulated_panel(rng, N=60, T=900, effect=10.0, common=False, heavy=False)
    first, last = T - 300, T - 2
    f0, f1, sc = cw.recursive_forecasts(y, X, z, t + 1, t, cw.blocks_for(first, last, 21))
    assert cw.stats(y, f0, f1, t + 1, sc, cw.VOL_LAGS)["t21"] > 3
    passed = []
    for _ in range(8):
        y, X, z, t, T = cw.simulated_panel(rng, N=60, T=900, common=False, heavy=False)
        first, last = T - 300, T - 2
        f0, f1, sc = cw.recursive_forecasts(y, X, z, t + 1, t, cw.blocks_for(first, last, 21))
        passed.append(cw.stats(y, f0, f1, t + 1, sc, cw.VOL_LAGS)["t21"] >= 1.645)
    assert sum(passed) <= 2


def test_the_common_swing_has_the_scale_it_claims():
    g = cw.swing(np.random.default_rng(0), 2000)
    assert abs(g.std() - 1) < 1e-12 and abs(g.mean()) < 1e-12 and np.corrcoef(g[1:], g[:-1])[0, 1] > 0.9


def test_a_calibration_job_uses_only_rows_up_to_its_window(tmp_path):
    rng = np.random.default_rng(0)
    T, N = 420, 30
    t, c = np.meshgrid(np.arange(T - 1), np.arange(N), indexing="ij")
    origin, col = t.ravel(), c.ravel()
    X = rng.normal(size=(len(origin), 3))
    y = 0.5 * X[:, 0] + rng.normal(size=len(origin))
    path = tmp_path / "arrays.npz"
    np.savez(path, y=y, X=X, origin=origin, realized=origin + 1, col=col, T=np.array(T), N=np.array(N))
    window = np.arange(300, 360)
    out = cw._calibration_job((str(path), window, "matched to real PE", 2, 1))
    assert len(out) == 2 and all(np.isfinite(r["t21"]) and np.isfinite(r["raw21"]) for r in out)
    y2 = y.copy()
    y2[origin + 1 > window[-1]] += 100.0     # a change after the window must change nothing
    path2 = tmp_path / "arrays2.npz"
    np.savez(path2, y=y2, X=X, origin=origin, realized=origin + 1, col=col, T=np.array(T), N=np.array(N))
    assert cw._calibration_job((str(path2), window, "matched to real PE", 2, 1)) == out
