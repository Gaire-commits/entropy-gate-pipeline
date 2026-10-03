"""The engine has to learn a planted edge, and, just as important, learn to leave noise alone."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import (
    EngineConfig, action_rewards, baseline_positions, build_state, entropy_columns, market_neutral,
    missed_opportunities, model_signal, run_engine, score_positions, threshold_rule,
)
from src.entropy_features import ENTROPY_COLUMNS


def _predictions(edge_bps, n_folds=10, per_fold=3000, seed=0, noise_bps=12.0):
    """Saved-prediction frame for one model. `edge_bps` is how far the model's
    signed confidence s moves the expected return: E[ret] = edge_bps * s."""
    rng = np.random.default_rng(seed)
    frames = []
    day = 0
    for fold in range(n_folds):
        n = per_fold
        s = rng.uniform(-1, 1, n)
        ret_bps = edge_bps * s + noise_bps * rng.normal(size=n)
        frames.append(pd.DataFrame({
            "fold": fold,
            "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(day + np.arange(n) // 50, unit="D"),
            "symbol": "AAA", "bar_index": np.arange(n) % 78,
            "y": (ret_bps > 0).astype(int), "ret": ret_bps / 1e4,
            "prob": 0.5 + 0.5 * s, "arch": "m1", "seed": 0, "n_params": 0, "best_epoch": -1,
            "pass_trend": False, "pass_entropy": False, "pass_entropy_weighted": False,
            "p_trend": 1.0, "has_reading": True,
        }))
        day += n // 50
    return pd.concat(frames, ignore_index=True)


CFG = EngineConfig(cost_bps=2.0, seed=0)
FULL = EngineConfig(cost_bps=2.0, seed=0, feedback="full")
BOTH = pytest.mark.parametrize("cfg", [CFG, FULL], ids=["bandit", "full"])


def test_action_rewards_charge_cost_only_for_trading():
    r = action_rewards(np.array([10.0, -4.0]), cost_bps=2.0)
    assert r.tolist() == [[-12.0, 0.0, 8.0], [2.0, 0.0, -6.0]]


def test_state_is_built_from_confidence_and_never_from_the_return():
    pred = _predictions(edge_bps=4.0, n_folds=2, per_fold=200)
    state = build_state(pred)
    assert {"conf_m1", "slot", "ret_bps", "y"} <= set(state.columns)
    expected = 2 * (pred.sort_values(["date", "symbol", "bar_index"])["prob"].to_numpy() - 0.5)
    np.testing.assert_allclose(np.sort(state["conf_m1"].to_numpy()), np.sort(expected), atol=1e-9)
    assert "ret_bps" not in __import__("src.engine", fromlist=["feature_columns"]).feature_columns(state)


@BOTH
def test_engine_learns_a_clear_planted_edge_and_beats_trading_everything(cfg):
    state = build_state(_predictions(edge_bps=6.0, seed=3))
    late = state["fold"] >= 4
    engine = score_positions(run_engine(state, cfg, log=lambda _: None).query("fold >= 4"))
    follow = score_positions(_with_pos(state[late], baseline_positions(state)["follow_m1"][late]))
    assert engine["net_lo"] > 0                           # a real gain, not noise
    assert engine["net_bps"] > follow["net_bps"] + 0.1    # by skipping the signals that earn less than their cost
    assert 0.3 < engine["trade_rate"] < 0.95


@BOTH
def test_engine_with_a_marginal_edge_never_does_worse_than_staying_flat(cfg):
    """An edge barely above cost may or may not be learned; losing money on it is the failure."""
    for seed in (1, 4):
        state = build_state(_predictions(edge_bps=4.0, seed=seed))
        net = score_positions(run_engine(state, cfg, log=lambda _: None).query("fold >= 4"))["net_bps"]
        assert net > -0.15


def test_full_feedback_learns_a_marginal_edge_from_the_trades_it_did_not_take():
    """With bandit feedback a policy that settles on flat stops seeing what trading would have
    paid, and on these seeds it misses or under-trades an edge barely above cost. Scoring all
    three actions on every past sample shows it the missed trades."""
    for seed in (1, 4):
        state = build_state(_predictions(edge_bps=4.0, seed=seed))
        full = score_positions(run_engine(state, FULL, log=lambda _: None).query("fold >= 4"))
        assert full["net_bps"] > 0.25
        assert full["trade_rate"] > 0.3


@BOTH
def test_engine_learns_to_stay_flat_on_pure_noise(cfg):
    """Trading every signal loses the full cost; a policy that learns loses almost none."""
    for seed in (1, 3):
        state = build_state(_predictions(edge_bps=0.0, seed=seed))
        late = state["fold"] >= 5
        follow = score_positions(_with_pos(state[late], baseline_positions(state)["follow_m1"][late]))
        engine = score_positions(run_engine(state, cfg, log=lambda _: None).query("fold >= 5"))
        assert follow["net_bps"] < -1.4
        assert engine["net_bps"] > -0.5
        assert engine["trade_rate"] < 0.3


def test_the_first_fold_has_no_feedback_and_earns_exactly_zero():
    state = build_state(_predictions(edge_bps=4.0, n_folds=3))
    scored = run_engine(state, CFG, log=lambda _: None)
    first = scored[scored["fold"] == 0]
    assert (first["pos"] == 0).all() and (first["pnl_bps"] == 0).all()


def test_engine_rows_line_up_with_the_state():
    state = build_state(_predictions(edge_bps=4.0, n_folds=3, per_fold=500))
    scored = run_engine(state, CFG, log=lambda _: None)
    np.testing.assert_array_equal(scored.loc[state.index, "ret_bps"], state["ret_bps"])


def test_unknown_feedback_is_rejected():
    state = build_state(_predictions(edge_bps=4.0, n_folds=2, per_fold=200))
    with pytest.raises(ValueError, match="unknown feedback"):
        run_engine(state, EngineConfig(feedback="Full"), log=lambda _: None)


@BOTH
def test_engine_only_uses_feedback_from_earlier_folds(cfg):
    """Rewriting the future must not change what the policy decides now."""
    pred = _predictions(edge_bps=6.0, n_folds=5, per_fold=1500)
    state = build_state(pred)
    changed = state.copy()
    changed.loc[changed["fold"] >= 3, "ret_bps"] = -changed.loc[changed["fold"] >= 3, "ret_bps"]
    a = run_engine(state, cfg, log=lambda _: None)
    b = run_engine(changed, cfg, log=lambda _: None)
    np.testing.assert_array_equal(a[a["fold"] <= 3]["pos"].to_numpy(), b[b["fold"] <= 3]["pos"].to_numpy())


def test_threshold_rule_baseline_finds_the_edge_from_the_past():
    state = build_state(_predictions(edge_bps=6.0, seed=3))
    rule = threshold_rule(state, cost_bps=2.0)
    assert score_positions(rule[rule["fold"] >= 4])["net_bps"] > 0.5


def _gated_predictions(seed=0, share=0.3, edge_bps=8.0):
    """Pure noise, except on gate-approved samples, where the signal pays `edge_bps` per unit of confidence."""
    pred = _predictions(edge_bps=0.0, seed=seed)
    passed = np.random.default_rng(seed + 100).random(len(pred)) < share
    pred["pass_trend"] = passed
    pred["ret"] += np.where(passed, edge_bps * 2 * (pred["prob"] - 0.5), 0.0) / 1e4
    pred["y"] = (pred["ret"] > 0).astype(int)
    return pred


def test_full_feedback_finds_an_edge_that_exists_only_on_gate_approved_samples():
    """The thesis question in miniature: the gate's verdict helps a learner only when the edge
    lives where the gate says. Diluted over all samples the edge loses money, so a bandit learner
    that retreats to flat may never find the pocket; full feedback finds it, and only with the gate."""
    state = build_state(_gated_predictions())
    with_gate = score_positions(run_engine(state, FULL, use_entropy=True, log=lambda _: None).query("fold >= 4"))
    without = score_positions(run_engine(state, FULL, use_entropy=False, log=lambda _: None).query("fold >= 4"))
    assert with_gate["net_lo"] > 0.3
    assert with_gate["net_bps"] > without["net_bps"] + 0.3


def test_missed_opportunities_find_a_planted_edge_and_nothing_in_the_noise():
    late = build_state(_gated_predictions()).query("fold >= 1")
    table = missed_opportunities(late, {"idle": [np.zeros(len(late))]}, cost_bps=2.0, signal=model_signal(late))
    passed, refused = table[table["gate"] == "passed"], table[table["gate"] == "not passed"]
    strongest = passed[passed["strength"] == 5].iloc[0]
    assert strongest["best_action"] == "follow signal" and strongest["best_lo"] > 2.0
    assert strongest["left_idle"] == pytest.approx(strongest["best_net"])   # all of it missed
    assert (refused["best_action"] == "stay flat").all()
    assert (refused["left_idle"] == 0).all()


def test_trading_where_flat_was_best_counts_as_money_left_on_the_table():
    late = build_state(_predictions(edge_bps=0.0, seed=1)).query("fold >= 1")
    every = {"every": [np.sign(late["conf_m1"].to_numpy())]}
    table = missed_opportunities(late, every, cost_bps=2.0, signal=model_signal(late))
    assert (table["best_action"] == "stay flat").all()
    np.testing.assert_allclose(table["left_every"], -table["net_every"])
    assert (table["left_every"] > 1.0).all()


def test_situations_are_graded_by_trained_models_not_rules():
    pred = _predictions(edge_bps=4.0, n_folds=2, per_fold=300)
    rule = pred.assign(arch="always_up", prob=1.0)
    state = build_state(pd.concat([pred, rule], ignore_index=True))
    np.testing.assert_array_equal(model_signal(state), state["conf_m1"].to_numpy())


def _panel_predictions(edge_bps, drift_bps=6.0, n_symbols=12, n_folds=12, days=60, slots=(0, 12, 24, 36),
                       seed=0, noise_bps=12.0, shock_bps=8.0):
    """Many symbols at the same moments. All symbols share each date-and-bar's market move: a
    drift whose sign flips by fold (the coin flip a quarter's market direction is) plus a shock.
    The model's signal is stock-specific, independent of the market, and worth `edge_bps`."""
    rng = np.random.default_rng(seed)
    drift = rng.choice([-drift_bps, drift_bps], n_folds)
    frames, day = [], 0
    for fold in range(n_folds):
        d, b, sym = (a.ravel() for a in np.meshgrid(np.arange(days), slots, np.arange(n_symbols), indexing="ij"))
        market = (drift[fold] + shock_bps * rng.normal(size=(days, len(slots))))[d, np.searchsorted(slots, b)]
        s = rng.uniform(-1, 1, len(d))
        ret_bps = market + edge_bps * s + noise_bps * rng.normal(size=len(d))
        frames.append(pd.DataFrame({
            "fold": fold, "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(day + d, unit="D"),
            "symbol": [f"S{i:02d}" for i in sym], "bar_index": b,
            "y": (ret_bps > 0).astype(int), "ret": ret_bps / 1e4,
            "prob": 0.5 + 0.5 * s, "arch": "m1", "seed": 0, "n_params": 0, "best_epoch": -1,
            "pass_trend": False, "pass_entropy": False, "pass_entropy_weighted": False,
            "p_trend": 1.0, "has_reading": True,
        }))
        day += days
    return pd.concat(frames, ignore_index=True)


def test_market_neutral_removes_the_common_move_and_leaves_the_gate_alone():
    state = build_state(_panel_predictions(edge_bps=4.0, n_folds=3, days=10))
    neutral = market_neutral(state, min_peers=10)
    for col in ("ret_bps", "conf_m1"):
        np.testing.assert_allclose(neutral.groupby(["date", "bar_index"])[col].mean(), 0.0, atol=1e-9)
    assert len(neutral) == len(state)
    np.testing.assert_array_equal(neutral["pass_trend"], state["pass_trend"])
    assert (neutral.groupby("date")["fold"].nunique() == 1).all()   # never mixes two folds' returns


def test_market_neutral_needs_a_wide_cross_section_and_drops_thin_moments():
    state = build_state(_panel_predictions(edge_bps=4.0, n_folds=2, days=10, n_symbols=12))
    with pytest.raises(ValueError, match="wide cross-section"):
        market_neutral(state, min_peers=20)
    thin = state[~((state["date"] == state["date"].min()) & (state["symbol"] > "S02"))]   # one date left with 3 symbols
    kept = market_neutral(thin, min_peers=10)
    assert state["date"].min() not in set(kept["date"]) and len(kept) < len(thin)


def test_raw_engine_chases_each_quarters_market_drift_and_market_neutral_does_not():
    """No stock-specific edge anywhere. Each quarter the market drifts +-6 bps, so on raw returns
    'long' (or 'short') looks right for tens of thousands of samples and then fails next quarter."""
    for seed in (0, 1):
        state = build_state(_panel_predictions(edge_bps=0.0, seed=seed))
        raw = score_positions(run_engine(state, FULL, log=lambda _: None).query("fold >= 3"))
        neutral = score_positions(
            run_engine(market_neutral(state, min_peers=10), FULL, log=lambda _: None).query("fold >= 3"))
        assert raw["net_bps"] < -0.5
        assert neutral["net_bps"] > -0.05 and neutral["trade_rate"] < 0.05


def test_market_neutral_engine_finds_a_stock_specific_edge_that_drift_hides_from_the_raw_engine():
    for seed in (0, 1):
        state = build_state(_panel_predictions(edge_bps=6.0, seed=seed))
        raw = score_positions(run_engine(state, FULL, log=lambda _: None).query("fold >= 3"))
        neutral = score_positions(
            run_engine(market_neutral(state, min_peers=10), FULL, log=lambda _: None).query("fold >= 3"))
        assert neutral["net_lo"] > 0.5
        assert neutral["net_bps"] > raw["net_bps"] + 0.5


def _entropy_predictions(seed=0, edge_bps=8.0, **kw):
    """Pure noise, except where one entropy feature is high: there the signal pays `edge_bps` per
    unit of confidence. The other entropy features are noise. The informative one lives on its
    real scale (about 0.35 +- 0.05, like the hourly trend share), so it only helps if scaled."""
    pred = _predictions(edge_bps=0.0, seed=seed, **kw)
    rng = np.random.default_rng(seed + 200)
    for c in ENTROPY_COLUMNS:
        pred[c] = rng.normal(size=len(pred))
    pred["ent_trend3_hour"] = rng.normal(0.35, 0.05, len(pred))
    high = pred["ent_trend3_hour"].to_numpy() > 0.38
    pred["ret"] += np.where(high, edge_bps * 2 * (pred["prob"] - 0.5), 0.0) / 1e4
    pred["y"] = (pred["ret"] > 0).astype(int)
    return pred


def test_engine_reads_continuous_entropy_when_saved_and_the_gate_flags_otherwise():
    assert entropy_columns(build_state(_entropy_predictions(n_folds=2, per_fold=200)))[0].startswith("ent_")
    old = build_state(_predictions(edge_bps=4.0, n_folds=2, per_fold=200))
    assert entropy_columns(old) == ["pass_trend", "pass_entropy", "pass_entropy_weighted", "p_trend", "has_reading"]


def test_entropy_comes_from_whichever_model_saved_it():
    new = _entropy_predictions(n_folds=2, per_fold=200).assign(arch="m2")
    old = new.drop(columns=ENTROPY_COLUMNS).assign(arch="m1")
    import pandas as pd
    state = build_state(pd.concat([old, new], ignore_index=True))
    assert state["ent_trend3_hour"].notna().all()
    assert {"conf_m1", "conf_m2"} <= set(state.columns)


def test_full_feedback_finds_an_edge_that_only_continuous_entropy_reveals():
    state = build_state(_entropy_predictions())
    with_entropy = score_positions(run_engine(state, FULL, use_entropy=True, log=lambda _: None).query("fold >= 4"))
    without = score_positions(run_engine(state, FULL, use_entropy=False, log=lambda _: None).query("fold >= 4"))
    assert with_entropy["net_lo"] > 0.2
    assert with_entropy["net_bps"] > without["net_bps"] + 0.2


def test_entropy_scaling_comes_from_the_first_fold_only():
    """Rescaling the future's entropy features must not change what the policy decides now."""
    state = build_state(_entropy_predictions(n_folds=5, per_fold=1500))
    changed = state.copy()
    later = changed["fold"] >= 3
    changed.loc[later, "ent_trend3_hour"] = changed.loc[later, "ent_trend3_hour"] * 10
    a = run_engine(state, FULL, log=lambda _: None)
    b = run_engine(changed, FULL, log=lambda _: None)
    np.testing.assert_array_equal(a[a["fold"] <= 2]["pos"].to_numpy(), b[b["fold"] <= 2]["pos"].to_numpy())


def _with_pos(rows, pos, cost=2.0):
    out = rows[["fold", "date", "symbol", "ret_bps", "y"]].copy()
    out["pos"] = pos
    out["pnl_bps"] = out["pos"] * out["ret_bps"] - cost * out["pos"].abs()
    return out
