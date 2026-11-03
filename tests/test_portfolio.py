"""The portfolio tool: no look-ahead, execution and cost accounting, entropy, and recovering a planted signal."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src import entropy, portfolio as pf

CFG = pf.Config(min_train_sessions=120, refit_every=21, min_dollar_volume=1e6)


def market(n_days=520, n_syms=120, seed=0, signal=0.0) -> tuple[pf.Market, np.ndarray]:
    """Random prices; with `signal`, a persistent stock trait moves the next week's returns and shows in its past returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    syms = [f"S{i:03d}" for i in range(n_syms)]
    trait = rng.normal(size=n_syms)
    r = rng.normal(0, 0.015, (n_days, n_syms)) + signal * trait
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=syms)
    open_ = close.shift(1).fillna(close) * np.exp(rng.normal(0, 0.003, close.shape))
    real = pd.DataFrame(True, index=dates, columns=syms)
    return pf.Market(close=close, open=open_, raw_close=close.copy(), dollar_volume=close * 1e6, real=real,
                     event=pd.DataFrame(False, index=dates, columns=syms),
                     sector=pd.Series([f"Sec{i % 5}" for i in range(n_syms)], index=syms)), trait


def test_nothing_at_t_depends_on_prices_after_t():
    m, _ = market()
    day = m.close.index[420]
    other = pf.perturbed_after(m, day)
    a, ea = pf.build_features(m, CFG)
    b, eb = pf.build_features(other, CFG)
    assert ea.loc[:day].equals(eb.loc[:day])
    for name in a:
        pd.testing.assert_frame_equal(a[name].loc[:day], b[name].loc[:day], check_exact=True)


def test_a_forecast_at_t_does_not_depend_on_returns_after_t():
    m, _ = market()
    index = m.close.index
    day = index[420]
    other = pf.perturbed_after(m, day)
    run = {}
    for key, mk in (("a", m), ("b", other)):
        feats, elig = pf.build_features(mk, CFG)
        target = pf.forward_returns(mk, CFG)
        run[key], _ = pf.walk_forward(feats, elig, target, pf.MODELS["base_pe_x"], index[[400, 420]], CFG)
    pd.testing.assert_frame_equal(run["a"], run["b"], check_exact=False, rtol=1e-12)
    assert run["a"].notna().any(axis=None)


def test_training_rows_stop_where_their_target_is_realized():
    """The target of the last training date must be realized by the refit date, and no later date may be usable."""
    m, _ = market()
    index = m.close.index
    i = 420
    last = pf.purge_end(i, CFG)
    fwd = pf.forward_returns(m, CFG)
    growth = (1 + pf.asset_returns(m, CFG.execution)).cumprod()
    s = "S000"
    assert fwd.loc[index[last], s] == pytest.approx(growth[s].iloc[i] / growth[s].iloc[last + 1] - 1)
    assert last + 1 + CFG.horizon == i
    # a shock to returns after the refit date cannot change the fit, even at the last training date's edge
    other = pf.perturbed_after(m, index[i])
    assert pf.forward_returns(other, CFG).loc[index[last], s] == pytest.approx(fwd.loc[index[last], s])
    assert pf.forward_returns(other, CFG).loc[index[last + 1], s] != pytest.approx(fwd.loc[index[last + 1], s])


def test_the_holdout_is_not_scored_unless_asked():
    index = pd.bdate_range("2016-01-04", "2025-06-30")
    cfg = pf.Config()
    locked = pf.decision_dates(index, cfg)
    start = pd.Timestamp(cfg.holdout_start)
    assert index[index.get_loc(locked[-1]) + 1 + cfg.horizon] < start     # the last target is realized before the lock
    assert pf.scored_end(index, cfg) < start
    scored = pf.decision_dates(index, cfg, score_holdout=True)
    assert scored[-1] > start and index[index.get_loc(scored[-1]) + 1 + cfg.horizon] <= index[-1]
    assert pf.scored_end(index, cfg, score_holdout=True) == index[-1]


def test_a_decision_earns_nothing_on_its_own_day_or_the_session_before_it_executes():
    """A jump on the decision day and on the execution day itself (close to close, it happened before the fill) is not earned."""
    m, _ = market(n_days=30, n_syms=3)
    index = m.close.index
    cfg = replace(CFG, cost_bps=0, impact=0)
    d = index[10]
    jump = m.close.copy()
    jump.loc[index[10]:, "S000"] *= 1.5                                       # jump on the decision day
    jump.loc[index[11]:, "S000"] *= 1.5                                       # and on the execution day
    jump.loc[index[12]:, "S000"] *= 1.1                                       # earned: the first session after the fill
    mk = pf.Market(jump, m.open, m.raw_close, m.dollar_volume, m.real, m.event, m.sector)
    out = pf.simulate({d: {"S000": 1.0}}, mk, cfg)
    assert out.index[0] == index[11] and out["gross"].iloc[0] == 0            # the fill day holds nothing yet
    r = pf.asset_returns(mk, cfg.execution)["S000"]
    assert out.loc[index[12], "gross"] == pytest.approx(r.loc[index[12]])
    assert r.loc[index[12]] == pytest.approx(jump["S000"].iloc[12] / jump["S000"].iloc[11] - 1)


def test_costs_and_turnover_are_charged_on_what_is_traded():
    m, _ = market(n_days=40, n_syms=4)
    index = m.close.index
    cfg = replace(CFG, cost_bps=10, impact=0)
    out = pf.simulate({index[5]: {"S000": 0.5, "S001": 0.5}, index[15]: {"S000": 0.5, "S002": 0.5}}, m, cfg)
    assert out.loc[index[6], "cost"] == pytest.approx(1.0 * 10e-4)            # building from cash: every dollar bought
    assert out.loc[index[6], "turnover"] == pytest.approx(0.5)
    assert 0.45 < out.loc[index[16], "turnover"] < 0.55                       # half the book swaps
    assert out.loc[index[16], "cost"] == pytest.approx(2 * out.loc[index[16], "turnover"] * 10e-4)
    assert (out["net"] <= out["gross"] + 1e-12).all()
    with_impact = pf.simulate({index[5]: {"S000": 0.5, "S001": 0.5}}, m, replace(cfg, impact=1.0, aum=1e9))
    assert with_impact.loc[index[6], "cost"] >= out.loc[index[6], "cost"]


def test_a_stock_with_no_bar_cannot_be_traded_and_keeps_its_weight():
    m, _ = market(n_days=40, n_syms=3)
    index = m.close.index
    real = m.real.copy()
    real.loc[index[16], "S000"] = False
    mk = pf.Market(m.close.where(real), m.open.where(real), m.raw_close, m.dollar_volume, real, m.event, m.sector)
    cfg = replace(CFG, cost_bps=0, impact=0)
    out = pf.simulate({index[5]: {"S000": 1.0}, index[15]: {"S001": 1.0}}, mk, cfg)
    assert out.loc[index[16], "turnover"] == 0                                # S000 is held and has no bar: nothing can move
    assert pf.asset_returns(mk, cfg.execution).loc[index[16], "S000"] == 0
    assert mk.close["S000"].ffill().iloc[17] / mk.close["S000"].iloc[15] - 1 == pytest.approx(pf.asset_returns(mk, cfg.execution).loc[index[17], "S000"])


def test_an_event_session_earns_zero_and_leaves_no_return_for_the_features():
    m, _ = market(n_days=40, n_syms=3)
    d = m.close.index[20]
    event = m.event.copy()
    event.loc[d, "S001"] = True
    mk = pf.Market(m.close, m.open, m.raw_close, m.dollar_volume, m.real, event, m.sector)
    assert pf.asset_returns(mk).loc[d, "S001"] == 0
    assert np.isnan(pf.clean_returns(mk).loc[d, "S001"])


def test_rolling_entropy_matches_the_repo_measure_and_uses_no_future():
    rng = np.random.default_rng(3)
    x = rng.normal(size=300)
    frame = pd.DataFrame({"A": x})
    out = pf.rolling_pe(frame, m=3, tau=1, window=63)
    t = 200
    seg = x[t - 64: t + 1]                                                    # 63 patterns of 3 points end at t
    assert out["A"].iloc[t] == pytest.approx(entropy.permutation_entropy(seg, 3, 1, False, True))
    changed = x.copy()
    changed[t + 1:] = rng.normal(size=len(x) - t - 1)
    assert pf.rolling_pe(pd.DataFrame({"A": changed}), 3, 1, 63)["A"].iloc[t] == out["A"].iloc[t]
    assert np.isnan(out["A"].iloc[50])                                        # too few patterns yet


def test_rolling_entropy_low_on_a_trend_high_on_noise_and_skips_ties_and_gaps():
    n = 200
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"trend": np.arange(n, dtype=float), "noise": rng.normal(size=n), "flat": np.zeros(n)})
    gappy = rng.normal(size=n)
    gappy[100:140] = np.nan
    frame["gappy"] = gappy
    out = pf.rolling_pe(frame, 3, 1, 63)
    assert out["trend"].iloc[-1] == pytest.approx(0.0, abs=1e-12)
    assert out["noise"].iloc[-1] > 0.9
    assert out["flat"].isna().all()                                           # all ties: no value rather than a false zero
    assert np.isnan(out["gappy"].iloc[139]) and not np.isnan(out["gappy"].iloc[-1])


def test_a_planted_signal_is_recovered_and_noise_is_not():
    cfg = replace(CFG, refit_every=42)
    results = {}
    for key, signal in (("noise", 0.0), ("planted", 0.002)):
        m, _ = market(signal=signal)
        feats, elig = pf.build_features(m, cfg)
        target = pf.forward_returns(m, cfg)
        dates = pf.decision_dates(m.close.index, cfg, score_holdout=True)
        results[key] = pf.run_model("base", feats, elig, target, m, dates, cfg)
    assert abs(results["noise"]["ic"].mean()) < 0.03
    assert results["planted"]["ic"].mean() > 0.10
    assert results["planted"]["daily"]["net"].mean() > results["noise"]["daily"]["net"].mean()


def test_the_three_models_share_rows_and_coefficients_are_named():
    m, _ = market()
    feats, elig = pf.build_features(m, CFG)
    target = pf.forward_returns(m, CFG)
    dates = pf.decision_dates(m.close.index, CFG, score_holdout=True)[:10]
    masks = {}
    for name, names in pf.MODELS.items():
        f, c = pf.walk_forward(feats, elig, target, names, dates, CFG)
        masks[name] = f.notna().to_numpy()
        assert list(c.columns) == names
        assert set(pf.coefficient_table(c).columns) == {"mean", "std", "same_sign"}
    assert (masks["base"] == masks["base_pe"]).all() and (masks["base"] == masks["base_pe_x"]).all()


def test_the_holdings_rule_buffers_and_caps_sectors():
    syms = [f"S{i:02d}" for i in range(40)]
    forecast = pd.Series(np.arange(40, 0, -1, dtype=float), index=syms)
    sector = pd.Series(["A"] * 20 + ["B"] * 20, index=syms)
    cfg = pf.Config(top_frac=0.1, hold_frac=0.2, sector_cap=1.0)
    assert pf.choose_names(forecast, set(), sector, cfg) == ["S00", "S01", "S02", "S03"]
    # S06 is in the best fifth, so it stays; S20 is not, so it goes
    assert pf.choose_names(forecast, {"S06", "S20"}, sector, cfg) == ["S06", "S00", "S01", "S02"]
    capped = pf.choose_names(forecast, set(), sector, pf.Config(top_frac=0.1, hold_frac=0.2, sector_cap=0.5))
    assert capped == ["S00", "S01", "S20", "S21"]


def test_the_scorecard_figures():
    days = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([0.10, -0.50, 0.10, 0.10], index=days)
    p = pf.performance(r, turnover=pd.Series(0.5, index=days))
    assert p["max_drawdown"] == pytest.approx(-0.5)
    assert p["turnover_ann"] == pytest.approx(0.5 * 252)
    assert p["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    assert p["sortino"] == pytest.approx(r.mean() / np.sqrt(0.25 / 4) * np.sqrt(252))
    assert pf.newey_west_t(np.ones(10)) != pf.newey_west_t(np.arange(10.0))
    lo, hi = pf.block_bootstrap(np.random.default_rng(0).normal(0.5, 1, 500), np.mean, block=5, n_boot=500)
    assert lo < 0.5 < hi
    row = {"clark_west_t": 2.0, "clark_west_critical": 1.645, "loss_gap_hi": -0.001, "ic_gap_lo": 0.001, "net_gap_lo": -0.01}
    assert pf.verdict(row) == "forecast improves; net of costs no improvement shown"
    assert pf.verdict({**row, "loss_gap_hi": 0.001}).startswith("Clark-West detects content")     # significance alone is not an improvement
    assert pf.verdict({**row, "clark_west_t": 1.0}).startswith("forecast: no improvement")
    assert pf.verdict({**row, "clark_west_critical": 2.5}).startswith("forecast: no improvement")


def test_the_lookahead_check_catches_a_leak():
    m, _ = market()
    day = m.close.index[420]
    assert pf.no_lookahead_check(m, CFG, day) == {"eligibility": True, "features": True, "forecast": True}
    real_build = pf.build_features

    def leaky(mk, cfg):
        feats, elig = real_build(mk, cfg)
        feats["xs_ret_5"] = pf.xs_rank(mk.close.pct_change().shift(-1).where(elig))    # tomorrow's return
        return feats, elig
    pf.build_features = leaky
    try:
        assert pf.no_lookahead_check(m, CFG, day)["features"] is False
    finally:
        pf.build_features = real_build


def test_the_live_portfolio_explains_itself_and_respects_a_lock():
    m, _ = market()
    feats, elig = pf.build_features(m, CFG)
    target = pf.forward_returns(m, CFG)
    cut = m.close.index[300]
    out = pf.current_portfolio("base", feats, elig, target, m, CFG, {"S001"}, fit_until=cut)
    assert out["weight"].sum() == pytest.approx(1.0)
    assert set(out["action"]) <= {"hold", "buy"} and out["why"].str.len().gt(0).all()
    assert out["date"].iloc[0] == m.close.index[-1]
    # a model locked at `cut` must not change when returns after it change
    other = pf.perturbed_after(m, m.close.index[-1])
    assert pf.current_portfolio("base", *pf.build_features(other, CFG), pf.forward_returns(other, CFG), other, CFG, set(), fit_until=cut)["symbol"].tolist() \
        == pf.current_portfolio("base", feats, elig, target, m, CFG, set(), fit_until=cut)["symbol"].tolist()


def test_benchmarks_are_fetched_once_and_scored_on_the_portfolios_dates(tmp_path):
    days = pd.bdate_range("2021-12-20", periods=20, tz="UTC") + pd.Timedelta(hours=5)

    class Client:
        calls = 0

        def bars(self, symbols, timeframe, lo, hi, adjustment, feed):
            assert adjustment == "all" and feed == "sip" and timeframe == "1Day"
            Client.calls += 1
            return pd.DataFrame([{"symbol": s, "t": d, "open": 100.0 + i, "high": 102.0 + i, "low": 99.0 + i, "close": 101.0 + i, "volume": 1e6,
                                  "trade_count": 100, "vwap": 100.0} for s in symbols for i, d in enumerate(days)])
    path = tmp_path / "b.parquet"
    bars = pf.fetch_benchmarks(path, "2021-12-20", "2022-01-14", client=Client())
    pf.fetch_benchmarks(path, "2021-12-20", "2022-01-14", client=Client())
    assert Client.calls == 1 and set(bars["symbol"]) == {"SPY", "BIL"}
    spy = pf.benchmark_returns(bars, "SPY")
    assert spy.iloc[0] == pytest.approx(102 / 101 - 1)
    assert pf.benchmark_returns(bars, "SPY", "next_open").iloc[0] == pytest.approx(101 / 100 - 1)
    book = pd.DataFrame({"net": 0.001, "turnover": 0.1}, index=spy.index[2:])
    card = pf.scorecard({"book": book, "market": spy}, pf.benchmark_returns(bars, "BIL"), pf.Config(), {"book": pd.Series(0.02, index=spy.index[2:])})
    assert set(card["window"]) == {"development", "validation", "all scored"}
    assert card.loc[card["portfolio"] == "market", "turnover_ann"].isna().all()
    assert card.loc[(card["portfolio"] == "book") & (card["window"] == "all scored"), "ic_mean"].iloc[0] == pytest.approx(0.02)
    assert card[card["window"] == "all scored"]["from"].nunique() == 1        # both on the book's dates


def test_the_forecast_target_is_what_the_book_earns_after_a_next_close_fill():
    m, _ = market(signal=0.001)
    cfg = replace(CFG, top_frac=0.2, hold_frac=0.2, rebalance_every=5, horizon=5)
    feats, elig = pf.build_features(m, cfg)
    target = pf.forward_returns(m, cfg)
    dates = pf.decision_dates(m.close.index, cfg, score_holdout=True)
    forecasts, _ = pf.walk_forward(feats, elig, target, pf.MODELS["base"], dates, cfg)
    check = pf.reconcile_target(pf.portfolio_targets(forecasts, m, cfg), m, cfg)
    assert len(check) > 10 and check["gap"].abs().max() < 1e-12                  # exact, session for session
    assert (check["book"] - check["unlagged"]).abs().mean() > 1e-3               # and it would notice a target one session off
    assert (check["fill"] > check["decision"]).all() and (check["next_fill"].iloc[:-1].to_numpy() == check["fill"].iloc[1:].to_numpy()).all()


def test_no_target_where_a_fill_cannot_happen():
    m, _ = market(n_days=60, n_syms=3)
    index = m.close.index
    real = m.real.copy()
    real.loc[index[31], "S001"] = False
    mk = pf.Market(m.close.where(real), m.open.where(real), m.raw_close, m.dollar_volume, real, m.event, m.sector)
    fwd = pf.forward_returns(mk, CFG)
    assert np.isnan(fwd.loc[index[30], "S001"])                                  # no bar to enter on
    assert np.isnan(fwd.loc[index[25], "S001"])                                  # no bar to leave on
    assert not np.isnan(fwd.loc[index[30], "S000"]) and not np.isnan(fwd.loc[index[26], "S001"])


def test_nothing_past_the_lock_is_loaded():
    days = pd.bdate_range("2023-12-01", "2024-01-31")
    def bars(mult=1.0):
        return pd.DataFrame([{"symbol": s, "date": d, "open": 10.0 * mult, "high": 11.0 * mult, "low": 9.0 * mult, "close": 10.5 * mult, "volume": 1e6, "trade_count": 100.0}
                             for s in ("AAA", "BBB") for d in days])
    universe = pd.DataFrame({"symbol": ["AAA", "BBB"], "sector": ["X", "Y"]})
    actions = pd.DataFrame(columns=["kind", "symbol", "when", "source_symbol", "old_symbol", "new_symbol"])
    registry = pd.DataFrame(columns=["symbol", "date", "effect", "kind"])
    cfg = pf.Config()
    locked = pf.build_market(bars(), bars(), bars(), actions, registry, universe, until=pf.lock_date(cfg))
    assert locked.close.index.max() == pd.Timestamp("2024-01-02")
    assert pf.lock_date(cfg, True) == pd.Timestamp("2026-10-12")


def test_the_benchmark_guard_and_truncated_days():
    days = pd.bdate_range("2021-12-27", periods=8)
    bars = pd.DataFrame({"symbol": "SPY", "date": days, "open": 100.0, "close": [100, 101, 102, 103, 104, 300, 106, 107.0]})
    r = pf.benchmark_returns(bars, "SPY", truncated=pd.DatetimeIndex([days[5]]))
    assert r.loc[days[5]] == 0 and r.loc[days[6]] == pytest.approx(106 / 104 - 1)
    ok = pf.check_benchmarks(bars, {"SPY": {2022: 107 / 104 - 1}}, pd.DatetimeIndex([days[5]]))
    assert ok["ok"].all()
    assert not pf.check_benchmarks(bars, {"SPY": {2022: 0.20}}, pd.DatetimeIndex([days[5]]))["ok"].any()


def test_the_spread_estimator_finds_a_planted_spread():
    rng = np.random.default_rng(5)
    n, half = 400, 0.002
    eff = np.cumsum(rng.normal(0, 0.01, n))
    close = eff + half * rng.choice([-1, 1], n)
    # high and low bracket the path with the spread on both sides
    high, low = np.maximum(close, eff + 0.008) + half, np.minimum(close, eff - 0.008) - half
    days = pd.bdate_range("2020-01-01", periods=n)
    raw = pd.DataFrame({"symbol": "A", "date": days, "close": np.exp(close), "high": np.exp(high), "low": np.exp(low), "volume": 1.0, "trade_count": 1.0})
    est = pf.abdi_ranaldo(raw)["spread"].median()
    assert 0.5 * (2 * half) < est < 2.5 * (2 * half)


def test_the_rule_search_applies_its_stated_rule():
    import sys
    sys.path.insert(0, "scripts")
    import portfolio_rules as script
    table = pd.DataFrame({"score": [0.30, 0.28, 0.10], "turn_val": [20.0, 5.0, 1.0]})
    assert script.choose(table)["turn_val"] == 5.0                               # tied within 0.05: the lower turnover
    assert script.choose(pd.DataFrame({"score": [0.30, 0.10], "turn_val": [20.0, 1.0]}))["turn_val"] == 20.0
    m, _ = market(n_days=1000, n_syms=60)
    doc = {"grid": {"cadence_sessions": [5, 10], "tranches_by_cadence": {5: 1, 10: 2}, "top_frac": [0.2], "hold_multiple": [1.0, 2.0]},
           "windows": {"development": ["2021-12-15", "2022-04-30"], "validation": ["2022-05-01", "2022-10-20"]},
           "costs": {"cost_bps": 5.0, "impact": 1.0, "aum": 1e7}, "fixed": {"sector_cap": 1.0, "execution": "next_close", "model": "base"}}
    out = script.search(m, doc)
    assert len(out) == 4 and out["score"].is_monotonic_decreasing
    assert (out["score"] == out[["ir_dev", "ir_val"]].min(axis=1)).all()


def test_staggered_tranches_share_the_book_and_each_earns_its_own_target():
    m, _ = market(signal=0.001)
    cfg = replace(CFG, rebalance_every=21, horizon=21, tranches=3, top_frac=0.2, hold_frac=0.2, sector_cap=1.0)
    assert pf.step(cfg) == 7
    with pytest.raises(ValueError):
        pf.step(replace(cfg, tranches=2))
    feats, elig = pf.build_features(m, cfg)
    target = pf.forward_returns(m, cfg)
    dates = pf.decision_dates(m.close.index, cfg, score_holdout=True)
    assert (np.diff([m.close.index.get_loc(d) for d in dates]) == 7).all()
    forecasts, _ = pf.walk_forward(feats, elig, target, pf.MODELS["base"], dates, cfg)
    per = pf.tranche_targets(forecasts, m, cfg)
    book = pf.portfolio_targets(forecasts, m, cfg)
    assert all(sum(w.values()) == pytest.approx(1.0) for w in book.values())
    for k, tr in enumerate(per):
        assert list(tr) == list(forecasts.index[k::3])                           # each tranche every third decision
        check = pf.reconcile_target(tr, m, cfg)
        assert len(check) >= 2 and check["gap"].abs().max() < 1e-12
    # after all three have started, the book holds all three tranches' names in equal thirds
    day = forecasts.index[2]
    expected = {}
    for tr in per:
        latest = tr[max(d for d in tr if d <= day)]
        for s, x in latest.items():
            expected[s] = expected.get(s, 0.0) + x / 3
    assert book[day] == pytest.approx(expected)


def test_a_stock_is_not_eligible_before_it_joined_the_index():
    m, _ = market(n_days=420, n_syms=60)
    days = m.close.index
    member = pd.DataFrame(True, index=days, columns=m.close.columns)
    member["S000"] = days >= days[400]
    mk = pf.Market(m.close, m.open, m.raw_close, m.dollar_volume, m.real, m.event, m.sector, member=member)
    _, plain = pf.build_features(m, CFG)
    _, filtered = pf.build_features(mk, CFG)
    assert plain.loc[days[399], "S000"] and not filtered.loc[days[399], "S000"] and filtered.loc[days[400], "S000"]
    assert filtered["S001"].equals(plain["S001"])
    actions = pd.DataFrame(columns=["kind", "symbol", "when", "source_symbol", "old_symbol", "new_symbol"])
    registry = pd.DataFrame(columns=["symbol", "date", "effect", "kind"])
    bars = pd.DataFrame([{"symbol": s, "date": d, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1e6, "trade_count": 100.0} for s in ("AAA", "BBB") for d in days[:30]])
    built = pf.build_market(bars, bars, bars, actions, registry, pd.DataFrame({"symbol": ["AAA", "BBB"], "sector": ["X", "Y"]}), joined=pd.Series({"AAA": days[10], "BBB": pd.NaT}))
    assert not built.member.loc[days[9], "AAA"] and built.member.loc[days[10], "AAA"] and built.member["BBB"].all()


def test_protocol_checks_measure_quoted_half_spreads():
    import sys
    sys.path.insert(0, "scripts")
    import protocol_checks as script

    class Client:
        def quotes(self, symbols, lo, hi):
            return pd.DataFrame({"symbol": ["A", "A", "A", "B"], "t": pd.Timestamp("2021-03-01T20:58:30Z"), "bid": [99.9, 99.9, 0.0, 49.95], "ask": [100.1, 100.1, 100.0, 50.05]})
    s = script.spreads(Client(), ["A", "B"], ["2021-03-01"]).set_index("symbol")
    assert s.loc["A", "half_spread_bp"] == pytest.approx(10.0)                  # half of 20 bp; the zero bid is dropped
    assert s.loc["B", "half_spread_bp"] == pytest.approx(10.0)
