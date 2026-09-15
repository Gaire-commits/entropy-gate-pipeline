"""Assemble training tensors from cached bars and a screening table.

`use_gate` is the switch behind Experiment 1: the same code path builds the
gated and ungated datasets, so any difference in downstream results is the gate
and not an incidental change in preprocessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_channels, gaf_encode, make_windows, mtf_encode, normalize_windows


def build_dataset(
    bars: dict[str, pd.DataFrame],
    gated: pd.DataFrame | None,
    cfg,
    use_gate: bool = True,
) -> dict[str, np.ndarray]:
    """Build (X, y, ret, dates, symbols) across the universe.

    A window is kept when the symbol passed the gate reading whose trade_date is
    the window's session, i.e. a reading computed at the previous close. Matching
    on the day the reading was computed instead would admit a morning window
    based on entropy that already saw that afternoon.
    """
    if use_gate and gated is None:
        raise ValueError("use_gate=True requires a screening table")

    allowed = None
    if use_gate:
        if "trade_date" not in gated.columns:
            raise ValueError("screening table has no trade_date; re-run scripts/run_screen.py")
        passed = gated[gated["passed"] & gated["trade_date"].notna()]
        allowed = set(zip(passed["symbol"], passed["trade_date"]))

    chunks = {"X": [], "y": [], "ret": [], "date": [], "symbol": []}
    for symbol, df in bars.items():
        features = build_channels(df, cfg.features.channels)
        close = df["close"].to_numpy(dtype=float)
        sessions = df["session_id"].to_numpy() if "session_id" in df else None
        timestamps = df.index.get_level_values("timestamp")

        windows = make_windows(
            features,
            close,
            window=cfg.features.window,
            horizon=cfg.features.horizon,
            embargo=cfg.features.embargo,
            session_id=sessions,
            deadzone=cfg.features.label_deadzone,
            allow_overnight=getattr(cfg.features, "allow_overnight", False),
        )
        if windows["X"].size == 0:
            continue

        signal_dates = timestamps[windows["signal_bar"]].normalize().tz_localize(None)
        keep = np.ones(len(signal_dates), dtype=bool)
        if allowed is not None:
            keep = np.array([(symbol, d) in allowed for d in signal_dates])
        if not keep.any():
            continue

        chunks["X"].append(windows["X"][keep])
        chunks["y"].append(windows["y"][keep])
        chunks["ret"].append(windows["ret"][keep])
        chunks["date"].append(signal_dates[keep].to_numpy())
        chunks["symbol"].append(np.full(int(keep.sum()), symbol))

    if not chunks["X"]:
        raise RuntimeError("no windows survived; loosen the gate or widen the date range")

    X = np.concatenate(chunks["X"])
    X = normalize_windows(X, cfg.features.normalize)
    X = np.transpose(X, (0, 2, 1))

    encoding = getattr(cfg.features, "encoding", "1d")
    if encoding == "gaf":
        X = np.stack([gaf_encode(sample) for sample in X])
    elif encoding == "mtf":
        X = np.stack([mtf_encode(sample) for sample in X])
    elif encoding != "1d":
        raise ValueError(f"unknown encoding '{encoding}'")

    order = np.argsort(np.concatenate(chunks["date"]), kind="stable")
    return {
        "X": X[order].astype(np.float32),
        "y": np.concatenate(chunks["y"])[order],
        "ret": np.concatenate(chunks["ret"])[order],
        "date": np.concatenate(chunks["date"])[order],
        "symbol": np.concatenate(chunks["symbol"])[order],
    }


def walk_forward_splits(dates: np.ndarray, cfg) -> list[dict[str, np.ndarray]]:
    """Rolling train/val/test index sets, split on calendar days.

    Splitting on dates rather than rows keeps every window from one session on
    the same side of a boundary, so no sample in test overlaps a sample in train.
    """
    unique = np.unique(dates)
    tr, va, te, step = cfg.train_days, cfg.val_days, cfg.test_days, cfg.step_days
    span = tr + va + te

    folds = []
    for start in range(0, len(unique) - span + 1, step):
        train_days = unique[start : start + tr]
        val_days = unique[start + tr : start + tr + va]
        test_days = unique[start + tr + va : start + span]
        folds.append(
            {
                "train": np.flatnonzero(np.isin(dates, train_days)),
                "val": np.flatnonzero(np.isin(dates, val_days)),
                "test": np.flatnonzero(np.isin(dates, test_days)),
                "test_start": test_days[0],
                "test_end": test_days[-1],
            }
        )
    return folds
