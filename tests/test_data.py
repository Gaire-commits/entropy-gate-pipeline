"""The session grid: a missing IEX bar must leave a gap, not shift the clock."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (
    bar_minutes, completeness, merge_bars, plan_fetch, resolve_end, save_symbol, to_session_grid,
)


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
    grid = to_session_grid(_raw(["2024-11-27", "2024-11-29", "2024-12-02"], early=["2024-11-29"]))
    assert grid.groupby("session_id").size().tolist() == [78, 42, 78]


def test_a_session_still_in_progress_is_dropped_not_mistaken_for_an_early_close():
    """Downloading at 11am must not leave a half day that looks complete."""
    raw = _raw(["2024-03-04", "2024-03-05"])
    ts = raw.index.get_level_values("timestamp")
    partial = raw[~((ts.normalize() == ts.normalize()[-1]) & (ts.hour >= 11))]
    grid = to_session_grid(partial)
    assert grid["session_id"].nunique() == 1
    assert grid.index.get_level_values("timestamp")[-1].date().isoformat() == "2024-03-04"


def test_a_genuine_early_close_is_kept_when_later_days_follow_it():
    grid = to_session_grid(_raw(["2024-11-29", "2024-12-02"], early=["2024-11-29"]))
    assert grid.groupby("session_id").size().tolist() == [42, 78]


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


def test_explicit_end_dates_are_inclusive_and_rolling_ones_are_not_dates():
    assert resolve_end("2026-09-21") == pd.Timestamp("2026-09-22")
    assert resolve_end("yesterday") == pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    assert resolve_end("today") > resolve_end("yesterday")


def test_merge_bars_appends_new_days_and_lets_the_newest_copy_win():
    old = _raw(["2024-03-04", "2024-03-05"])
    fresh = _raw(["2024-03-05", "2024-03-06"])
    fresh["close"] = -1.0
    merged = merge_bars(old, fresh)
    days = merged.index.get_level_values("timestamp").normalize().unique()
    assert len(days) == 3
    assert not merged.index.duplicated().any()
    on_overlap = merged[merged.index.get_level_values("timestamp").normalize() == days[1]]
    assert (on_overlap["close"] == -1.0).all()
    assert merge_bars(None, fresh) is fresh and merge_bars(old, fresh.iloc[:0]) is old


def _save(tmp_path, end, symbol="SPY", start="2016-01-01"):
    save_symbol(_raw(["2024-03-04", "2024-03-05"]), tmp_path, symbol, "5Min", start, end, "iex", "split")


def test_nothing_cached_downloads_everything(tmp_path):
    assert plan_fetch(tmp_path, "SPY", "5Min", "2016-01-01", "2026-06-30", "iex", "split") == (
        pd.Timestamp("2016-01-01"), pd.Timestamp("2026-07-01"))


def test_a_fully_covered_request_downloads_nothing(tmp_path):
    _save(tmp_path, "2026-06-30")
    assert plan_fetch(tmp_path, "SPY", "5Min", "2016-01-01", "2026-06-30", "iex", "split") is None
    assert plan_fetch(tmp_path, "SPY", "5Min", "2020-01-01", "2025-01-01", "iex", "split") is None


def test_extending_the_end_date_downloads_only_the_new_days(tmp_path):
    _save(tmp_path, "2026-06-30")
    first, stop = plan_fetch(tmp_path, "SPY", "5Min", "2016-01-01", "2026-09-21", "iex", "split")
    assert first == pd.Timestamp("2024-03-05")        # the last cached day, refetched in case it was partial
    assert stop == pd.Timestamp("2026-09-22")


def test_a_rolling_end_always_tops_up(tmp_path):
    _save(tmp_path, "today")
    assert plan_fetch(tmp_path, "SPY", "5Min", "2016-01-01", "today", "iex", "split") is not None


def test_a_different_feed_or_an_earlier_start_downloads_everything(tmp_path):
    _save(tmp_path, "2026-06-30")
    assert plan_fetch(tmp_path, "SPY", "5Min", "2016-01-01", "2026-06-30", "sip", "split")[0] == pd.Timestamp("2016-01-01")
    assert plan_fetch(tmp_path, "SPY", "5Min", "2015-01-01", "2026-06-30", "iex", "split")[0] == pd.Timestamp("2015-01-01")


def test_a_later_listing_is_not_redownloaded_on_every_run(tmp_path):
    """XLC's first bar is in 2018, but the request covered 2016."""
    _save(tmp_path, "2026-06-30", symbol="XLC")
    assert plan_fetch(tmp_path, "XLC", "5Min", "2016-01-01", "2026-06-30", "iex", "split") is None
