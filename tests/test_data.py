"""The session grid: a missing IEX bar must leave a gap, not shift the clock."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import bar_minutes, completeness, is_cached, save_symbol, to_session_grid


def _raw(days, drop=(), early=()):
    stamps = []
    for day in days:
        close = "12:55" if day in early else "15:55"
        stamps += list(pd.date_range(f"{day} 09:30", f"{day} {close}", freq="5min"))
    ts = pd.DatetimeIndex(stamps).tz_localize("America/New_York")
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": np.arange(len(ts), dtype=float) + 100, "volume": 10.0},
        index=pd.MultiIndex.from_arrays([["SPY"] * len(ts), ts], names=["symbol", "timestamp"]),
    )
    return df.drop(df.index[list(drop)])


def test_full_days_get_78_slots_and_early_closes_get_42():
    grid = to_session_grid(_raw(["2024-11-27", "2024-11-29"], early=["2024-11-29"]))
    assert grid.groupby("session_id").size().tolist() == [78, 42]


def test_missing_bars_become_nan_rows_at_the_right_time():
    grid = to_session_grid(_raw(["2024-03-04"], drop=[10, 11]))
    assert len(grid) == 78
    assert grid["close"].isna().sum() == 2
    missing = grid[grid["close"].isna()].index.get_level_values("timestamp")
    assert [t.strftime("%H:%M") for t in missing] == ["10:20", "10:25"]
    last = grid.index.get_level_values("timestamp")[-1]
    assert last.strftime("%H:%M") == "15:55"


def test_bar_index_counts_from_the_open():
    grid = to_session_grid(_raw(["2024-03-04", "2024-03-05"], drop=[3]))
    assert grid["bar_index"].tolist() == list(range(78)) * 2
    assert grid["session_id"].tolist() == [0] * 78 + [1] * 78


def test_dst_days_keep_wall_clock_times():
    grid = to_session_grid(_raw(["2024-03-08", "2024-03-11"]))
    first_bars = grid.index.get_level_values("timestamp")[::78]
    assert [t.strftime("%H:%M") for t in first_bars] == ["09:30", "09:30"]


def test_completeness_reports_the_share_of_real_bars():
    grid = to_session_grid(_raw(["2024-03-04"], drop=range(39)))
    assert completeness(grid) == 0.5


def test_bar_minutes():
    assert bar_minutes("5Min") == 5
    assert bar_minutes("1Min") == 1


def test_cache_covers_later_listings_and_narrower_requests(tmp_path):
    df = _raw(["2024-03-04"])
    save_symbol(df, tmp_path, "XLC", "5Min", "2016-01-01", "2026-06-30", "iex", "split")
    assert is_cached(tmp_path, "XLC", "5Min", "2016-01-01", "2026-06-30", "iex", "split")
    assert is_cached(tmp_path, "XLC", "5Min", "2020-01-01", "2025-01-01", "iex", "split")
    assert not is_cached(tmp_path, "XLC", "5Min", "2015-01-01", "2026-06-30", "iex", "split")
    assert not is_cached(tmp_path, "XLC", "5Min", "2016-01-01", "2026-06-30", "sip", "split")
    assert not is_cached(tmp_path, "SPY", "5Min", "2016-01-01", "2026-06-30", "iex", "split")
