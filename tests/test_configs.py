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
