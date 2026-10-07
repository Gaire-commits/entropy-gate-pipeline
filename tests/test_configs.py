"""Every shipped config has to load and name things that exist, before a long run finds out the hard way."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import RULES
from src.config import load_config
from src.data import resolve_universe
from src.ml import TREE_MODELS
from src.models import ARCHITECTURES

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_loads_and_names_things_that_exist(path):
    cfg = load_config(path)
    assert cfg.experiment == path.stem
    known = set(RULES) | set(TREE_MODELS) | set(ARCHITECTURES)
    assert set(cfg.sweep.archs) <= known, f"unknown archs: {set(cfg.sweep.archs) - known}"
    symbols = resolve_universe(ROOT / cfg.data.universe if isinstance(cfg.data.universe, str) else cfg.data.universe)
    assert len(symbols) == len(set(symbols)) and len(symbols) >= 1
    f = cfg.features
    assert f.window + f.embargo + f.horizon <= 78, "must fit inside one 78-bar session"
    assert cfg.evaluation.headline_cost_bps in cfg.evaluation.cost_bps


def test_universe_files_carry_a_sector_for_every_symbol():
    for path in sorted((ROOT / "universe").glob("*.csv")):
        table = pd.read_csv(path)
        assert {"symbol", "sector"} <= set(table.columns), path.name
        assert table["sector"].notna().all() and table["symbol"].is_unique, path.name


def test_cross_asset_universe_spans_several_asset_classes():
    table = pd.read_csv(ROOT / "universe" / "cross_asset.csv")
    assert table["sector"].nunique() >= 6
    assert {"TLT", "GLD", "UUP", "HYG", "SPY"} <= set(table["symbol"])


def test_holdout_config_never_trains_on_what_it_trades_and_can_reuse_the_sp500_screen():
    from src.holdout import TARGETS, resolve_roles

    cfg = load_config(ROOT / "configs" / "ndx_holdout.yaml")
    data = cfg.data
    for key in ("universe", "train_universe", "trade_universe"):
        setattr(data, key, str(ROOT / getattr(data, key)))
    load, train, trade = resolve_roles(data)
    assert not train & trade, "a held-out stock would be trained on"
    assert (train | trade) <= set(load) and len(train) >= 400 and len(trade) >= 80
    assert cfg.features.target in TARGETS and cfg.evaluation.book_per_side * 2 <= len(trade)
    sp500 = load_config(ROOT / "configs" / "sp500_multihour.yaml")
    assert vars(cfg.screening) == vars(sp500.screening), "screen.parquet is only reusable if screening is identical"
    for key in ("window", "embargo", "horizon", "stride", "channels"):
        assert getattr(cfg.features, key) == getattr(sp500.features, key)


def test_replication_config_trains_exactly_like_the_holdout_and_scores_every_stock():
    from src.holdout import resolve_roles

    def roles(name):
        cfg = load_config(ROOT / "configs" / name)
        for key in ("universe", "train_universe", "trade_universe"):
            setattr(cfg.data, key, str(ROOT / getattr(cfg.data, key)))
        return cfg, resolve_roles(cfg.data)

    hold, (_, hold_train, hold_trade) = roles("ndx_holdout.yaml")
    rep, (load, rep_train, rep_trade) = roles("ndx_replication.yaml")
    assert rep_train == hold_train                                   # same training stocks
    assert hold_trade < rep_trade and rep_train < rep_trade          # scores the held-out 85 and the 418
    for block in ("screening", "features", "model", "validation"):
        assert vars(getattr(rep, block)) == vars(getattr(hold, block)), block
    assert "gbm" in rep.sweep.archs
