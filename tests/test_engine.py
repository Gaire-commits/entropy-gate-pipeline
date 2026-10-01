"""The engine has to learn a planted edge, and, just as important, learn to leave noise alone."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import (
    EngineConfig, action_rewards, baseline_positions, build_state, missed_opportunities, model_signal,
    run_engine, score_positions, threshold_rule,
)


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
    with_gate = score_positions(run_engine(state, FULL, use_gate=True, log=lambda _: None).query("fold >= 4"))
    without = score_positions(run_engine(state, FULL, use_gate=False, log=lambda _: None).query("fold >= 4"))
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


def _with_pos(rows, pos, cost=2.0):
    out = rows[["fold", "date", "symbol", "ret_bps", "y"]].copy()
    out["pos"] = pos
    out["pnl_bps"] = out["pos"] * out["ret_bps"] - cost * out["pos"].abs()
    return out
