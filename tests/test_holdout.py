"""Train on one universe, trade another: no held-out stock in training, peers-only labels, real transfer."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["EGP_DEVICE"] = "cpu"

from src.crosssection import cross_sectional_ic
from src.dataset import build_dataset, walk_forward_splits
from src.experiment import ensemble, load_predictions, run_arch
from src.holdout import apply_target, holdout_split, relative_returns, resolve_roles, restrict_folds
from src.synthetic import BARS, bars_from_returns


def test_roles_default_to_every_symbol_and_load_what_the_roles_name():
    load, train, trade = resolve_roles(SimpleNamespace(universe=["A", "B", "C"]))
    assert load == ["A", "B", "C"] and train == trade == {"A", "B", "C"}
    load, train, trade = resolve_roles(SimpleNamespace(universe=["A", "B"], train_universe=["A"], trade_universe=["B", "Z"]))
    assert load == ["A", "B", "Z"] and train == {"A"} and trade == {"B", "Z"}


def _data(seed=0, n_train=30, n_trade=10, days=20, slots=(0, 12, 24)):
    rng = np.random.default_rng(seed)
    syms = [f"T{i}" for i in range(n_train)] + [f"H{i}" for i in range(n_trade)]
    d, b, s = (a.ravel() for a in np.meshgrid(np.arange(days), slots, np.arange(len(syms)), indexing="ij"))
    return {"symbol": np.array(syms)[s], "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(d, unit="D"),
            "bar_index": b, "ret": rng.normal(0, 0.005, len(d)), "y": np.zeros(len(d), dtype=np.int64)}, \
        set(syms[:n_train]), set(syms[n_train:])


def test_folds_never_train_or_validate_on_a_held_out_stock_and_test_only_on_them():
    data, train, trade = _data()
    idx = np.arange(len(data["ret"]))
    folds = [{"train": idx[:600], "val": idx[600:900], "test": idx[900:]}]
    out = restrict_folds(folds, data["symbol"], train, trade)[0]
    sym = data["symbol"]
    assert set(sym[out["train"]]) <= train and set(sym[out["val"]]) <= train and set(sym[out["test"]]) <= trade
    assert len(out["test"]) and not set(sym[out["train"]]) & trade
    same = restrict_folds(folds, sym, train | trade, train | trade)[0]
    assert all(np.array_equal(same[k], folds[0][k]) for k in ("train", "val", "test"))   # no split: unchanged


def test_relative_returns_compare_each_universe_only_with_itself():
    data, train, trade = _data()
    rel, keep = relative_returns(data, train, trade, min_peers=5)
    frame = pd.DataFrame({"g": np.isin(data["symbol"], list(trade)), "d": data["date"], "b": data["bar_index"], "r": rel})
    np.testing.assert_allclose(frame.groupby(["g", "d", "b"])["r"].mean(), 0.0, atol=1e-12)

    moved = dict(data, ret=data["ret"] + np.where(np.isin(data["symbol"], list(trade)), 0.01, 0.0))
    rel2, _ = relative_returns(moved, train, trade, min_peers=5)
    is_train = np.isin(data["symbol"], list(train))
    np.testing.assert_allclose(rel2[is_train], rel[is_train])           # held-out stocks never set a training label

    _, keep_thin = relative_returns(data, train, trade, min_peers=15)  # 10 held-out stocks < 15 peers
    assert keep_thin[is_train].all() and not keep_thin[~is_train].any()


def test_targets_direction_is_untouched_relative_rewrites_ret_and_y_and_unknown_fails():
    data, train, trade = _data()
    before = data["ret"].copy()
    assert apply_target(data, "direction", train, trade).all() and np.array_equal(data["ret"], before)
    keep = apply_target(data, "relative", train, trade, min_peers=5)
    assert keep.all() and np.array_equal(data["ret_raw"], before)
    assert np.array_equal(data["y"], (data["ret"] > 0).astype(np.int64))
    with pytest.raises(ValueError, match="unknown target"):
        apply_target(data, "vibes", train, trade)


def test_holdout_split_is_disjoint_complete_and_keeps_the_base_lists_sectors():
    base = pd.DataFrame({"symbol": list("ABCDEF"), "sector": ["Tech", "Tech", "Energy", "Health", "Tech", "Energy"]})
    other = pd.DataFrame({"symbol": ["B", "E", "X"], "sector": ["Technology", "Technology", "Technology"]})
    train, held = holdout_split(base, other)
    assert list(held["symbol"]) == ["B", "E"] and list(held["sector"]) == ["Tech", "Tech"]
    assert set(train["symbol"]) == {"A", "C", "D", "F"} and "X" not in set(held["symbol"]) | set(train["symbol"])


def test_nasdaq100_parser_finds_the_ticker_table_and_its_industry_column():
    from build_universe import parse_nasdaq100

    noise = pd.DataFrame({"Year": [2024], "Close": [1.0]})
    table = pd.DataFrame({"Company": ["Adobe Inc.", "Apple Inc."], "Ticker": [" ADBE", "AAPL "],
                          "ICB Industry[1]": ["Technology", "Technology"], "ICB Subsector[1]": ["Software", "Hardware"]})
    out = parse_nasdaq100([noise, table])
    assert list(out["symbol"]) == ["ADBE", "AAPL"] and list(out["sector"]) == ["Technology", "Technology"]


# ---------------------------------------------------------------- end to end: does a signal transfer?

def _stock_returns(sessions, drift_sd, rng, market):
    """Per-bar returns: the market's move, shared by every stock, plus a slowly varying stock-specific
    drift and noise. The drift is what a model can learn from a stock's own window."""
    n = sessions * BARS
    mu, shocks = np.zeros(n), rng.normal(0, drift_sd * np.sqrt(1 - 0.995**2), n)
    for i in range(1, n):
        mu[i] = 0.995 * mu[i - 1] + shocks[i]
    return market + mu + rng.normal(0, 0.001, n)


def _transfer_ic(arch, signal_in_held_out, seed=0, n_train=24, n_trade=12, sessions=200):
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.0008, sessions * BARS)
    bars, train, trade = {}, set(), set()
    for i in range(n_train + n_trade):
        symbol, held = f"S{i:02d}", i >= n_train
        drift = 0.0 if held and not signal_in_held_out else 0.0002
        bars[symbol] = bars_from_returns(symbol, _stock_returns(sessions, drift, rng, market), seed=i, start="2020-01-02")
        (trade if held else train).add(symbol)
    cfg = SimpleNamespace(
        features=SimpleNamespace(channels=["log_return", "session_return"], window=12, horizon=30, embargo=1, stride=12,
                                 allow_overnight=False, normalize="fold_standardize", label_deadzone=0.0),
        model=SimpleNamespace(epochs=10, patience=3, batch_size=256, lr=1e-3, weight_decay=1e-4, dropout=0.2))
    data = build_dataset(bars, None, cfg)
    keep = apply_target(data, "relative", train, trade, min_peers=8)
    folds = walk_forward_splits(data["date"], SimpleNamespace(train_days=100, val_days=20, test_days=40, step_days=40))
    folds = restrict_folds(folds, data["symbol"], train, trade, keep)[:2]
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        run_arch(data, folds, arch, 0, cfg, Path(tmp), log=lambda _: None)
        pred = load_predictions(Path(tmp))
    assert set(pred["symbol"]) <= trade
    return cross_sectional_ic(ensemble(pred), min_peers=8, n_boot=300)


@pytest.mark.parametrize("arch", ["logreg", "gbm"])
def test_a_shared_stock_level_signal_transfers_to_stocks_the_model_never_saw(arch):
    ic = _transfer_ic(arch, signal_in_held_out=True)
    assert ic["ic_lo"] > 0.2 and ic["t"] > 8


@pytest.mark.parametrize("arch", ["logreg", "gbm"])
def test_a_signal_only_in_the_training_stocks_finds_nothing_in_the_held_out_ones(arch):
    ic = _transfer_ic(arch, signal_in_held_out=False)
    assert ic["ic_lo"] < 0 < ic["ic_hi"] and abs(ic["ic"]) < 0.05


@pytest.mark.parametrize("arch", ["logreg", "gbm"])
def test_scoring_more_stocks_leaves_the_held_out_predictions_unchanged(arch):
    """The replication scores every stock; the held-out stocks' predictions must equal the hold-out run's."""
    import tempfile

    rng = np.random.default_rng(5)
    sessions, market = 200, rng.normal(0, 0.0008, 200 * BARS)
    bars = {f"S{i:02d}": bars_from_returns(f"S{i:02d}", _stock_returns(sessions, 0.0002, rng, market), seed=i,
                                           start="2020-01-02") for i in range(30)}
    train, held = {f"S{i:02d}" for i in range(20)}, {f"S{i:02d}" for i in range(20, 30)}
    cfg = SimpleNamespace(
        features=SimpleNamespace(channels=["log_return", "session_return"], window=12, horizon=30, embargo=1, stride=12,
                                 allow_overnight=False, normalize="fold_standardize", label_deadzone=0.0),
        model=SimpleNamespace(epochs=5, patience=3, batch_size=256, lr=1e-3, weight_decay=1e-4, dropout=0.2))
    preds = []
    for trade in (held, train | held):
        data = build_dataset(bars, None, cfg)
        keep = apply_target(data, "relative", train, trade, min_peers=8)
        folds = walk_forward_splits(data["date"], SimpleNamespace(train_days=100, val_days=20, test_days=40, step_days=40))
        folds = restrict_folds(folds, data["symbol"], train, trade, keep)[:2]
        with tempfile.TemporaryDirectory() as tmp:
            run_arch(data, folds, arch, 0, cfg, Path(tmp), log=lambda _: None)
            preds.append(load_predictions(Path(tmp)))
    holdout, everyone = preds
    key = ["fold", "date", "symbol", "bar_index"]
    a = holdout.sort_values(key).reset_index(drop=True)
    b = everyone[everyone["symbol"].isin(held)].sort_values(key).reset_index(drop=True)
    assert len(a) == len(b) and set(everyone["symbol"]) == train | held
    np.testing.assert_allclose(a["prob"].to_numpy(), b["prob"].to_numpy(), rtol=0, atol=1e-6)
    np.testing.assert_allclose(a["ret"].to_numpy(), b["ret"].to_numpy())
