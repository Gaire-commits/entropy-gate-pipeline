"""The volatility forecaster on synthetic panels: causality, purging, the selection rule, and the test's power and null."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import nyse, portfolio as pf, volforecast as vf, volpanel

CFG = vf.VolConfig(min_train_sessions=120, refit_every=21, long_min=40)
DAYS = nyse.sessions("2019-01-02", "2021-12-31")


def panel(n=60, seed=0, pe_effect=0.0):
    """A panel with persistent log variance; with `pe_effect`, next session's log variance moves with a PE the returns really have."""
    rng = np.random.default_rng(seed)
    T = len(DAYS)
    frames = []
    for i in range(n):
        lv = np.empty(T)
        lv[0] = -9.0
        vol_noise = rng.normal(0, 0.25, T)
        # regimes of more and less ordinal structure in the returns: a strong AR(1) sign in `ordered` stretches
        ordered = (np.sin(np.arange(T) / 40.0 + rng.uniform(0, 6)) > 0)
        r = np.empty(T)
        r[0] = 0.0
        shocks = rng.normal(0, 1, T)
        for t in range(1, T):
            r[t] = (0.9 * r[t - 1] if ordered[t] else 0.0) + shocks[t]
        ret = r * 0.01
        pe = pf.rolling_pe(pd.DataFrame({"x": ret}), 3, 1, 63)["x"].to_numpy()
        pe = np.where(np.isfinite(pe), pe, np.nanmean(pe))
        for t in range(1, T):
            lv[t] = -9.0 + 0.6 * (lv[t - 1] + 9.0) + pe_effect * (pe[t - 1] - 0.9) + vol_noise[t]
        V = np.exp(lv)
        frames.append(pd.DataFrame({"symbol": f"S{i:02d}", "date": DAYS, "V": V, "rho": 0.7 * V, "r_on": np.sqrt(0.3 * V), "ret": ret, "half_day": np.asarray(nyse.is_half_day(DAYS)), "segment": 0,
                                    "in_span": True, "raw_close": 50.0, "dv63": 5e7, "open_source": "official"}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def rows():
    return vf.build_rows(panel(), CFG)


def test_rows_leave_out_half_day_targets_and_have_one_target_each(rows):
    assert not rows["target_date"].isin(nyse.HALF_DAYS).any()
    assert (rows["date"] < rows["target_date"]).all()
    p = panel(n=2)
    r = vf.build_rows(p, CFG)
    one = r[r["symbol"] == "S00"].iloc[50]
    i = DAYS.get_loc(one["date"])
    x = p[p["symbol"] == "S00"].reset_index(drop=True)
    assert one["next"] == pytest.approx(x["V"].iloc[i + 1])
    assert one["d"] == pytest.approx(np.log(x["V"].iloc[i]))
    assert one["w"] == pytest.approx(np.log(x["V"].iloc[i - 4:i + 1].mean()))
    assert one["m"] == pytest.approx(np.log(x["V"].iloc[i - 21:i + 1].mean()))


def test_nothing_at_an_origin_depends_on_later_sessions():
    p = panel(n=6)
    a = vf.build_rows(p, CFG)
    cut = DAYS[400]
    q = p.copy()
    q.loc[q["date"] > cut, ["V", "ret"]] = q.loc[q["date"] > cut, ["V", "ret"]].to_numpy() * 3.0
    q["rho"] = 0.7 * q["V"]
    b = vf.build_rows(q, CFG)
    cols = ["d", "w", "m", "long", "pe"]
    pd.testing.assert_frame_equal(a.loc[a["date"] <= cut, ["symbol", "date"] + cols].reset_index(drop=True), b.loc[b["date"] <= cut, ["symbol", "date"] + cols].reset_index(drop=True))


def test_a_model_trains_only_on_targets_already_realized(rows):
    """Changing the target of a row realized after the refit date must not change that refit's forecasts."""
    first, last = DAYS[300], DAYS[320]
    a, _ = vf.walk_forward(rows, vf.LOG_HAR, CFG, first, last)
    T = a["date"].min()
    changed = rows.copy()
    later = changed["target_date"] > T
    changed.loc[later, "y"] = changed.loc[later, "y"] + 5.0
    b, _ = vf.walk_forward(changed, vf.LOG_HAR, CFG, first, last)
    block = a["date"] <= a["date"].min()
    assert np.allclose(a.loc[block, "f"], b.loc[block, "f"])
    earlier = changed.copy()
    earlier.loc[earlier["target_date"] == T, "y"] += 5.0
    c, _ = vf.walk_forward(earlier, vf.LOG_HAR, CFG, first, last)
    assert not np.allclose(a.loc[block, "f"], c.loc[block, "f"])                 # the row realized at T does count


def test_the_candidates_run_and_the_selection_rule_is_the_stated_one(rows):
    first, last = DAYS[300], DAYS[500]
    losses = {}
    for name, spec in vf.CANDIDATES.items():
        fc, _ = vf.walk_forward(rows, spec["names"], CFG, first, last, spec.get("per_stock", False), spec.get("level", False))
        assert len(fc) and (fc["h"] > 0).all() and fc["f"].notna().all()
        losses[name] = vf.daily_loss(fc)["qlike"].mean()
    nv = vf.daily_loss(vf.naive_forecast(rows, first, last))
    assert vf.daily_loss(vf.walk_forward(rows, vf.LOG_HAR, CFG, first, last)[0])["mse_log"].mean() < nv["mse_log"].mean()
    assert vf.choose_baseline({"log_har": 1.004, "log_har_long": 1.0, "level_har": 2.0}) == "log_har"
    assert vf.choose_baseline({"log_har": 1.01, "log_har_long": 1.0, "level_har": 2.0}) == "log_har_long"


def test_the_comparison_finds_a_planted_pe_effect_and_not_a_missing_one():
    null = vf.pe_test(vf.build_rows(panel(seed=1), CFG), "log_har", CFG, DAYS[300], DAYS[-1])
    real = vf.pe_test(vf.build_rows(panel(seed=1, pe_effect=4.0), CFG), "log_har", CFG, DAYS[300], DAYS[-1])
    assert real["clark_west_t"] > 2.5 and real["pe_coef_same_sign"] == 1.0 and real["pe_coef_mean"] > 0
    assert null["clark_west_t"] < 1.645
    assert vf.verdict(real).startswith("PE improves")
    assert vf.verdict(null) == "PE: no improvement shown"
    assert set(real["by_year"]) <= {2020, 2021}


def test_the_official_open_replaces_the_vendor_open_exactly():
    from tests.test_volpanel import DAYS as D, build, inputs as _  # noqa: F401
    raw, split, adj, sessions, actions, registry = _.__wrapped__()
    v = volpanel.vendor_frame(raw, split, adj)
    symbols = ["S0", "S5"]
    plain = volpanel.build_panel(sessions, v, actions, registry, symbols, volpanel.Rules(market_gap_min_symbols=5), start=D[0], end=D[-1])
    day = D[20]
    official = pd.DataFrame({"symbol": ["S0"], "date": [day], "open_gap_bp": [50.0]})
    with_print = volpanel.build_panel(sessions, v, actions, registry, symbols, volpanel.Rules(market_gap_min_symbols=5), start=D[0], end=D[-1], official=official)
    a = plain[(plain["symbol"] == "S0") & (plain["date"] == day)].iloc[0]
    b = with_print[(with_print["symbol"] == "S0") & (with_print["date"] == day)].iloc[0]
    assert b["r_on"] == pytest.approx(a["r_on"] - 0.005) and b["open_source"] == "official"
    assert b["V"] == pytest.approx(b["r_on"] ** 2 + b["rho"])
    other = with_print[(with_print["symbol"] == "S5") & (with_print["date"] == day)].iloc[0]
    assert other["open_source"] == "vendor"


def test_realized_variance_from_bars():
    day = pd.Timestamp("2024-11-04")
    half = pd.Timestamp("2024-11-29")
    idx = [day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=5 * k) for k in range(78)] + [day + pd.Timedelta(hours=16)]
    idx += [half + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=5 * k) for k in range(42)] + [half + pd.Timedelta(hours=13, minutes=30)]
    close = 100.0 * np.exp(np.cumsum(np.full(len(idx), 0.001)))
    bars = pd.DataFrame({"open": close * np.exp(-0.001), "close": close}, index=pd.DatetimeIndex(idx))
    out = vf.realized_variance(bars, "AAA").set_index("date")
    assert out.loc[day, "rv"] == pytest.approx(78 * 0.001 ** 2)
    assert out.loc[day, "bars"] == 78 and out.loc[half, "bars"] == 42 and out.loc[half, "slots"] == 42
    assert out.loc[half, "rv"] == pytest.approx(42 * 0.001 ** 2)
    thin = vf.realized_variance(bars.iloc[:50], "AAA").set_index("date")
    assert np.isnan(thin.loc[day, "rv"])                                         # too few bars: no value


def test_the_lock():
    cfg = vf.VolConfig()
    assert vf.lock_end(cfg) == pd.Timestamp("2024-01-02")
    assert vf.lock_end(cfg, True) == pd.Timestamp("2026-10-12")


def test_the_stages_run_end_to_end_and_the_reserved_stage_needs_its_flag(monkeypatch, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import vol_forecast as script
    cfg = replace(CFG, validation_start="2021-01-04", test_start="2021-06-01", test_end="2021-12-31")
    monkeypatch.setattr(script.vf, "load_panel", lambda *a, **k: panel(n=55, seed=3))
    args = type("A", (), {"audit": "x", "reference": "y", "official": "z", "rv": None, "score_reserved": False})()
    assert script.reserved_stage(args, cfg) == 2                                 # no flag: nothing runs
    assert "STOPPED" in capsys.readouterr().out
    assert script.baseline_stage(args, cfg) in (0, 2)
    out = capsys.readouterr().out
    assert "chosen on development QLIKE" in out and "validation gate" in out and "pe" not in out.split("validation gate")[1]
    args.score_reserved = True
    assert script.reserved_stage(args, cfg) == 0
    out = capsys.readouterr().out
    assert "PRIMARY" in out and "VERDICT" in out and "ROBUSTNESS" in out


def test_the_reserved_stage_requires_the_official_open(capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import vol_forecast as script
    args = type("A", (), {"audit": "x", "reference": "y", "official": None, "rv": None, "score_reserved": True})()
    assert script.reserved_stage(args, CFG) == 2
    assert "official open" in capsys.readouterr().out
