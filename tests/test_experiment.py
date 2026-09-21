"""Scoring, day-level uncertainty, rules, and the sweep's resume behaviour."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["EGP_DEVICE"] = "cpu"

from src.baselines import rule_probability
from src.config import load_config
from src.dataset import build_dataset, walk_forward_splits
from src.experiment import load_predictions, report, run_arch
from src.screening import apply_gate, screen_universe
from src.stats import compare_subsets, coverage_eligible, coverage_mask, summarize, trades
from src.synthetic import bars_from_returns, half_hour_momentum_returns

ROOT = Path(__file__).resolve().parents[1]


def _pred(n_days=60, per_day=5, edge=0.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2024-01-01", periods=n_days), per_day)
    ret = rng.normal(0, 0.001, len(dates))
    prob = np.where(rng.random(len(dates)) < 0.5 + edge, (ret > 0) * 1.0, (ret <= 0) * 1.0)
    return pd.DataFrame({"date": dates, "ret": ret, "y": (ret > 0).astype(int), "prob": prob})


def test_trades_skip_no_view_predictions_and_sign_the_pnl():
    pred = pd.DataFrame({"date": ["d"] * 3, "ret": [0.001, 0.001, -0.002], "y": [1, 1, 0], "prob": [0.9, 0.5, 0.2]})
    t = trades(pred)
    assert len(t) == 2
    assert t["gross_bps"].tolist() == pytest.approx([10.0, 20.0])
    assert t["correct"].all()


def test_perfect_foresight_and_coin_flips_score_as_expected():
    good = summarize(_pred(edge=0.5), n_boot=300)
    coin = summarize(_pred(edge=0.0), n_boot=300)
    assert good["accuracy"] == 1.0 and good["gross_bps"] > 0
    assert coin["accuracy_lo"] < 0.5 < coin["accuracy_hi"]
    assert coin["gross_bps_lo"] < 0 < coin["gross_bps_hi"]


def test_intervals_widen_when_trades_on_a_day_move_together():
    """Same number of trades, but perfectly correlated within each day: far less evidence."""
    rng = np.random.default_rng(3)
    independent = _pred(n_days=60, per_day=20, seed=3)
    clustered = independent.copy()
    day_ret = pd.Series(rng.normal(0, 0.001, 60), index=pd.bdate_range("2024-01-01", periods=60))
    clustered["ret"] = clustered["date"].map(day_ret).to_numpy()
    clustered["prob"] = 1.0
    independent["prob"] = 1.0
    w_ind = np.subtract(*[summarize(independent, n_boot=500)[k] for k in ("gross_bps_hi", "gross_bps_lo")])
    w_clu = np.subtract(*[summarize(clustered, n_boot=500)[k] for k in ("gross_bps_hi", "gross_bps_lo")])
    assert w_clu > 2 * w_ind


def test_compare_subsets_finds_a_real_difference():
    pred = _pred(n_days=120, per_day=4, edge=0.0, seed=5)
    good_days = pred["date"].isin(pred["date"].unique()[::2])
    pred.loc[good_days, "prob"] = (pred.loc[good_days, "ret"] > 0).astype(float)
    out = compare_subsets(pred, good_days.to_numpy(), n_boot=500)
    assert out["share_in"] == pytest.approx(0.5)
    assert out["diff_bps_lo"] > 0


def test_rules():
    data = {"day_return": np.array([0.01, -0.02, 0.0, np.nan]), "window_return": np.array([-1.0, 1.0, 1.0, -1.0])}
    idx = np.arange(4)
    assert rule_probability("always_up", data, idx).tolist() == [1, 1, 1, 1]
    assert rule_probability("momentum_day", data, idx).tolist() == [1.0, 0.0, 0.5, 0.5]
    assert rule_probability("momentum_window", data, idx).tolist() == [0.0, 1.0, 1.0, 0.0]


@pytest.fixture(scope="module")
def gao_setup():
    cfg = load_config(ROOT / "configs/spy_gao.yaml")
    cfg.model.epochs, cfg.model.patience = 40, 40
    cfg.evaluation.bootstrap = 100
    bars = {"SPY": bars_from_returns("SPY", half_hour_momentum_returns(200, beta=1.0, seed=2))}
    screen = apply_gate(screen_universe(bars, cfg.screening))
    data = build_dataset(bars, screen, cfg)
    folds = walk_forward_splits(data["date"], SimpleNamespace(train_days=100, val_days=20, test_days=30, step_days=30))[:2]
    return cfg, data, folds


def test_sweep_resumes_and_scores_every_model_on_the_same_rows(gao_setup, tmp_path):
    cfg, data, folds = gao_setup
    for arch in ("always_up", "momentum_day", "logreg"):
        assert run_arch(data, folds, arch, 0, cfg, tmp_path, log=lambda _: None) == len(folds)
    assert run_arch(data, folds, "logreg", 0, cfg, tmp_path, log=lambda _: None) == 0

    pred = load_predictions(tmp_path)
    rows = pred.groupby("arch").apply(lambda g: frozenset(zip(g["fold"], g["date"], g["symbol"])), include_groups=False)
    assert rows.nunique() == 1

    overall, gate, _ = report(pred, cfg)
    assert set(overall["arch"]) == {"always_up", "momentum_day", "logreg"}
    momentum = overall.set_index("arch").loc["momentum_day"]
    assert momentum["accuracy"] > 0.65 and momentum["gross_bps_lo"] > 0
    assert set(gate["gate"]) == {"trend", "entropy", "entropy_weighted"}


def test_training_deadzone_never_filters_the_test_set(gao_setup, tmp_path):
    cfg, data, folds = gao_setup
    cfg.features.label_deadzone = 0.01
    try:
        run_arch(data, folds, "always_up", 0, cfg, tmp_path, log=lambda _: None)
    finally:
        cfg.features.label_deadzone = 0.0
    pred = load_predictions(tmp_path)
    assert len(pred) == sum(len(f["test"]) for f in folds)
    assert (pred["ret"].abs() < 0.01).any()


# ------------------------------------------------- confidence and within-symbol scoring


def _graded(n_days=120, per_day=8, seed=11):
    """A well-calibrated model: the more confident it is, the more often it is right."""
    rng = np.random.default_rng(seed)
    n = n_days * per_day
    dates = np.repeat(pd.bdate_range("2024-01-01", periods=n_days), per_day)
    ret = rng.normal(0, 0.001, n)
    confidence = 0.45 * rng.random(n)
    right = rng.random(n) < 0.5 + confidence
    up = np.where(right, ret > 0, ret <= 0)
    fold = np.repeat(np.arange(n_days) // 20, per_day)
    return pd.DataFrame({"date": dates, "fold": fold, "ret": ret, "y": (ret > 0).astype(int),
                         "prob": np.where(up, 0.5 + confidence, 0.5 - confidence)})


def test_coverage_mask_keeps_about_the_target_share_of_eligible_rows():
    pred = _graded()
    eligible = coverage_eligible(pred)
    assert 0 < eligible.sum() < len(pred)          # early folds have no cutoff yet
    for level in (1.0, 0.5, 0.2, 0.1):
        keep = coverage_mask(pred, level)
        assert not keep[~eligible].any()
        assert abs(keep.sum() / eligible.sum() - level) < 0.06


def test_coverage_cutoff_ignores_the_folds_it_is_applied_to():
    """A cutoff taken from the whole test period would use the future."""
    pred = _graded()
    late = pred["fold"] == pred["fold"].max()
    before = coverage_mask(pred, 0.5)[late.to_numpy()]
    shifted = pred.copy()
    shifted.loc[late, "prob"] = 0.5 + (shifted.loc[late, "prob"] - 0.5) * 0.01
    after = coverage_mask(shifted, 0.5)[late.to_numpy()]
    assert before.sum() > 0 and after.sum() == 0        # cutoff unchanged, confidences collapsed


def test_confidence_filtering_finds_an_edge_hidden_by_coin_flips():
    pred = _graded()
    everything = summarize(pred[coverage_mask(pred, 1.0)], n_boot=300)
    confident = summarize(pred[coverage_mask(pred, 0.2)], n_boot=300)
    assert confident["accuracy"] > everything["accuracy"] + 0.1
    assert confident["gross_bps"] > everything["gross_bps"]


def test_within_symbol_comparison_removes_a_pure_symbol_effect():
    """One ETF simply earns more, and the gate happens to approve it more often."""
    rng = np.random.default_rng(5)
    n_days, per_day = 150, 4
    dates = np.repeat(pd.bdate_range("2024-01-01", periods=n_days), per_day)
    symbol = np.tile(["RICH", "POOR"], len(dates) // 2)
    ret = np.where(symbol == "RICH", 0.0010, -0.0010) + rng.normal(0, 0.0002, len(dates))
    pred = pd.DataFrame({"date": dates, "symbol": symbol, "ret": ret,
                         "y": (ret > 0).astype(int), "prob": 1.0})
    approved = (symbol == "RICH") & (rng.random(len(dates)) < 0.5)

    raw = compare_subsets(pred, approved, n_boot=400)
    within = compare_subsets(pred, approved, n_boot=400, demean_by="symbol")
    assert raw["diff_bps"] > 12                      # looks like the gate picks winners
    assert abs(within["diff_bps"]) < 1               # it only picked the richer ETF
    assert within["diff_bps_lo"] < 0 < within["diff_bps_hi"]
