"""Assemble samples from cached bars and annotate them with the gate's verdict.

Every sample is kept whether or not the gate approved its day. The gate's
decision rides along as columns instead, so one set of predictions answers Q1
directly: compare the same model's trades on approved days against its trades
on the rest. Training a separate model on approved days only would change the
training set at the same time as the selection, and the two effects could not
be told apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_channels, make_windows, normalize_windows, session_return
from .screening import GATE_STATISTICS

GATE_COLUMNS = [f"pass_{name}" for name in GATE_STATISTICS] + ["p_trend"]


def build_dataset(bars: dict[str, pd.DataFrame], screen: pd.DataFrame | None, cfg) -> dict[str, np.ndarray]:
    """Build (X, y, ret, date, symbol, ...) across the universe, time-ordered.

    X is (n, channels, window). Gate columns match a sample to the reading whose
    trade_date is the sample's session, i.e. a reading computed at the previous
    close. `has_reading` is False where no reading exists yet (warm-up days).
    """
    f = cfg.features
    lookup = None
    if screen is not None and len(screen):
        if "trade_date" not in screen.columns or "pass_trend" not in screen.columns:
            raise ValueError("screen table is from an older version; re-run scripts/run_screen.py")
        dated = screen.dropna(subset=["trade_date"])
        lookup = dated.set_index(["symbol", "trade_date"])[GATE_COLUMNS]

    chunks: dict[str, list] = {
        k: [] for k in ("X", "y", "ret", "date", "symbol", "day_return", "window_return", "bar_index")
    }
    gate_chunks: dict[str, list] = {k: [] for k in GATE_COLUMNS + ["has_reading"]}

    for symbol, df in bars.items():
        windows = make_windows(
            build_channels(df, f.channels),
            df["close"].to_numpy(dtype=float),
            window=f.window,
            horizon=f.horizon,
            embargo=f.embargo,
            session_id=df["session_id"].to_numpy(),
            stride=getattr(f, "stride", 1),
            allow_overnight=getattr(f, "allow_overnight", False),
        )
        n = len(windows["y"])
        if n == 0:
            continue

        signal = windows["signal_bar"]
        dates = df.index.get_level_values("timestamp")[signal].normalize().tz_localize(None)

        chunks["X"].append(windows["X"])
        chunks["y"].append(windows["y"])
        chunks["ret"].append(windows["ret"])
        chunks["date"].append(dates.to_numpy())
        chunks["symbol"].append(np.full(n, symbol))
        chunks["day_return"].append(session_return(df)[signal])
        chunks["window_return"].append(windows["window_return"])
        chunks["bar_index"].append(df["bar_index"].to_numpy()[signal])

        if lookup is not None:
            keys = pd.MultiIndex.from_arrays([np.full(n, symbol), dates])
            matched = lookup.reindex(keys)
            gate_chunks["has_reading"].append(matched["p_trend"].notna().to_numpy())
            for col in GATE_COLUMNS:
                values = matched[col].to_numpy()
                if col.startswith("pass_"):
                    values = np.where(pd.isna(values), False, values).astype(bool)
                gate_chunks[col].append(values)

    if not chunks["X"]:
        raise RuntimeError("no samples could be built; check the date range and window geometry")

    X = normalize_windows(np.concatenate(chunks["X"]), f.normalize)
    date = np.concatenate(chunks["date"])
    order = np.argsort(date, kind="stable")

    data = {"X": np.transpose(X, (0, 2, 1)).astype(np.float32)[order]}
    for key in ("y", "ret", "date", "symbol", "day_return", "window_return", "bar_index"):
        data[key] = np.concatenate(chunks[key])[order]
    if lookup is not None:
        for key, parts in gate_chunks.items():
            data[key] = np.concatenate(parts)[order]
    return data


def walk_forward_splits(dates: np.ndarray, cfg) -> list[dict[str, np.ndarray]]:
    """Rolling train/val/test index sets, split on calendar days.

    Splitting on dates rather than rows keeps every window from one session on
    the same side of a boundary, so no sample in test overlaps a sample in train.
    With step_days equal to test_days the test periods tile the history without
    overlapping, so every test day is scored exactly once.
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
