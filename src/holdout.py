"""Train on one universe, trade another, and ask which stocks beat their peers.

A held-out universe tests whether a model learned something about stocks in general or only about
the stocks it was trained on. `configs/ndx_holdout.yaml` trains on the S&P 500 without the Nasdaq-100's
members and trades those members (`scripts/build_universe.py --holdout nasdaq100`): the model never sees
a held-out stock's history, in training or in validation. Both universes share the same dates, so this
is a hold-out across stocks, not across time; the walk-forward split still keeps every test day after
every training day.

The relative target asks the question a ranked book trades on: did this stock beat its peers at this
moment? Each sample's return is measured against the average of its own universe at the same date and
bar (training stocks against training stocks, held-out stocks against held-out stocks), so the market's
move cancels from the label and neither universe's average leaks into the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import resolve_universe

TARGETS = ("direction", "relative")


def resolve_roles(data_cfg) -> tuple[list[str], set[str], set[str]]:
    """(symbols to load, training symbols, trading symbols).

    Without `train_universe` / `trade_universe` every symbol in `universe` does both, as before. A
    symbol named in a role but missing from `universe` is loaded as well.
    """
    load = list(resolve_universe(data_cfg.universe))
    train_spec = getattr(data_cfg, "train_universe", None)
    trade_spec = getattr(data_cfg, "trade_universe", None)
    train = set(resolve_universe(train_spec)) if train_spec is not None else set(load)
    trade = set(resolve_universe(trade_spec)) if trade_spec is not None else set(load)
    known = set(load)
    return load + sorted((train | trade) - known), train, trade


def restrict_folds(folds: list[dict], symbols: np.ndarray, train: set[str], trade: set[str],
                   keep: np.ndarray | None = None) -> list[dict]:
    """Training and validation rows from training symbols only; test rows from trading symbols only.

    With both sets equal to every symbol and nothing dropped, the folds come back unchanged.
    """
    in_train, in_trade = np.isin(symbols, list(train)), np.isin(symbols, list(trade))
    if keep is not None:
        in_train, in_trade = in_train & keep, in_trade & keep
    out = []
    for fold in folds:
        f = dict(fold)
        f["train"] = fold["train"][in_train[fold["train"]]]
        f["val"] = fold["val"][in_train[fold["val"]]]
        f["test"] = fold["test"][in_trade[fold["test"]]]
        out.append(f)
    return out


def relative_returns(data: dict, train: set[str], trade: set[str], min_peers: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Each return minus the average of its own universe at the same date and bar.

    Returns (relative return, rows kept). A trading-only symbol is compared with the other trading-only
    symbols; every other symbol with the training universe. A row whose group had fewer than
    `min_peers` symbols at that moment has no meaningful average and is dropped.
    """
    symbols = data["symbol"]
    group = np.where(np.isin(symbols, list(train)), 0, np.where(np.isin(symbols, list(trade)), 1, -1))
    frame = pd.DataFrame({"g": group, "date": data["date"], "bar": data["bar_index"], "ret": data["ret"]})
    by_moment = frame.groupby(["g", "date", "bar"])["ret"]
    keep = (group >= 0) & (by_moment.transform("size").to_numpy() >= min_peers)
    rel = (frame["ret"] - by_moment.transform("mean")).to_numpy()
    return np.where(keep, rel, np.nan), keep


def apply_target(data: dict, target: str, train: set[str], trade: set[str], min_peers: int = 20) -> np.ndarray:
    """Switch `data` to the configured target in place and return the rows that can be used.

    direction (default): did the stock go up. Nothing changes.
    relative: did it beat its universe's average at that moment. `ret` becomes the relative return, so
    every downstream score (bps, accuracy, books) is measured against the peers; the raw return is kept
    as `ret_raw`.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target '{target}'; options: {TARGETS}")
    if target == "direction":
        return np.ones(len(data["y"]), dtype=bool)
    rel, keep = relative_returns(data, train, trade, min_peers)
    data["ret_raw"] = data["ret"]
    data["ret"] = np.where(keep, rel, 0.0)
    data["y"] = (data["ret"] > 0).astype(np.int64)
    return keep


def holdout_split(base: pd.DataFrame, other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training universe, held-out universe) from two constituent lists.

    Training: the base list without the other index's members. Held out: the members of both, with the
    base list's columns (so their sectors use the same classification as the training universe). The
    other index's members outside the base list are left out: they have no comparable history.
    """
    members = set(other["symbol"])
    held = base["symbol"].isin(members)
    return base[~held].reset_index(drop=True), base[held].reset_index(drop=True)
