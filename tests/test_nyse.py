"""The hard-coded NYSE calendar: session counts per year, half-days, closures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import nyse

# Sessions per year in the SIP cache (first data audit, 2026-10-27), which the calendar must reproduce.
CACHE_SESSIONS = {2016: 252, 2017: 251, 2018: 251, 2019: 252, 2020: 253, 2021: 252, 2022: 251, 2023: 250,
                  2024: 252, 2025: 250}


@pytest.mark.parametrize("year,count", sorted(CACHE_SESSIONS.items()))
def test_session_counts_match_the_cache(year, count):
    assert len(nyse.sessions(f"{year}-01-01", f"{year}-12-31")) == count


def test_2026_through_the_cache_end():
    assert len(nyse.sessions("2026-01-01", "2026-10-12")) == 195
    assert len(nyse.sessions("2026-01-01", "2026-10-13")) == 196


def test_closures_and_half_days():
    days = nyse.sessions("2016-01-01", "2026-12-31")
    for closed in ("2018-12-05", "2025-01-09", "2022-06-20", "2020-07-03", "2021-12-24"):
        assert pd.Timestamp(closed) not in days
    assert pd.Timestamp("2021-12-31") in days            # New Year's Day 2022 fell on a Saturday: no closure
    assert nyse.HALF_DAYS.isin(days).all()                 # every half-day is a session
    assert nyse.is_half_day("2024-11-29") and not nyse.is_half_day("2024-11-27")
    assert nyse.slots("2024-12-24") == 42 and nyse.slots("2024-12-23") == 78
    assert nyse.close_time("2025-07-03") == pd.Timedelta(hours=13)
    assert not any(nyse.is_half_day(nyse.sessions("2026-01-01", "2026-10-12")))   # 2026 had none by the cache end


def test_outside_the_calendar_is_refused():
    with pytest.raises(ValueError):
        nyse.sessions("2015-12-01", "2016-01-10")
