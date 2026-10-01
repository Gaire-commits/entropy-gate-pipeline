"""The engine has to learn a planted edge, and, just as important, learn to leave noise alone."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import (
    EngineConfig, action_rewards, build_state, run_engine, score_positions, threshold_rule, baseline_positions,
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


def test_engine_learns_a_clear_planted_edge_and_beats_trading_everything():
    state = build_state(_predictions(edge_bps=6.0, seed=3))
    late = state["fold"] >= 4
    engine = score_positions(run_engine(state, CFG, log=lambda _: None).query("fold >= 4"))
    follow = score_positions(_with_pos(state[late], baseline_positions(state)["follow_m1"][late]))
    assert engine["net_lo"] > 0                           # a real gain, not noise
    assert engine["net_bps"] > follow["net_bps"] + 0.1    # by skipping the signals that earn less than their cost
    assert 0.3 < engine["trade_rate"] < 0.95


def test_engine_with_a_marginal_edge_never_does_worse_than_staying_flat():
    """An edge barely above cost may or may not be learned; losing money on it is the failure."""
    for seed in (1, 4):
        state = build_state(_predictions(edge_bps=4.0, seed=seed))
        net = score_positions(run_engine(state, CFG, log=lambda _: None).query("fold >= 4"))["net_bps"]
        assert net > -0.15


def test_engine_learns_to_stay_flat_on_pure_noise():
    """Trading every signal loses the full cost; a policy that learns loses almost none."""
    for seed in (1, 3):
        state = build_state(_predictions(edge_bps=0.0, seed=seed))
        late = state["fold"] >= 5
        follow = score_positions(_with_pos(state[late], baseline_positions(state)["follow_m1"][late]))
        engine = score_positions(run_engine(state, CFG, log=lambda _: None).query("fold >= 5"))
        assert follow["net_bps"] < -1.4
        assert engine["net_bps"] > -0.5
        assert engine["trade_rate"] < 0.3


def test_the_first_fold_has_no_feedback_and_earns_exactly_zero():
    state = build_state(_predictions(edge_bps=4.0, n_folds=3))
    scored = run_engine(state, CFG, log=lambda _: None)
    first = scored[scored["fold"] == 0]
    assert (first["pos"] == 0).all() and (first["pnl_bps"] == 0).all()


def test_engine_only_uses_feedback_from_earlier_folds():
    """Rewriting the future must not change what the policy decides now."""
    pred = _predictions(edge_bps=6.0, n_folds=5, per_fold=1500)
    state = build_state(pred)
    changed = state.copy()
    changed.loc[changed["fold"] >= 3, "ret_bps"] = -changed.loc[changed["fold"] >= 3, "ret_bps"]
    a = run_engine(state, CFG, log=lambda _: None)
    b = run_engine(changed, CFG, log=lambda _: None)
    np.testing.assert_array_equal(a[a["fold"] <= 3]["pos"].to_numpy(), b[b["fold"] <= 3]["pos"].to_numpy())


def test_threshold_rule_baseline_finds_the_edge_from_the_past():
    state = build_state(_predictions(edge_bps=6.0, seed=3))
    rule = threshold_rule(state, cost_bps=2.0)
    assert score_positions(rule[rule["fold"] >= 4])["net_bps"] > 0.5


def _with_pos(rows, pos, cost=2.0):
    out = rows[["fold", "date", "symbol", "ret_bps", "y"]].copy()
    out["pos"] = pos
    out["pnl_bps"] = out["pos"] * out["ret_bps"] - cost * out["pos"].abs()
    return out
