"""The NYSE trading calendar, 2016-2026: full-day closures and 1:00 pm early closes.

Hard-coded from the exchange's published holiday schedules, because the bars alone cannot tell a
half-day from a quiet afternoon: after a 1:00 pm close, trading goes on as extended-hours trades, which
minute bars include, so the cache's clock-based 09:30-16:00 filter keeps them (Alpaca's daily bars do
not count them in the open, high, low or close). The data check is in scripts/data_audit_v2.py: each
year's session count here matches the sessions in the SIP cache, and every listed half-day is tested
against the bars.

Includes the two unscheduled closures in the period (national days of mourning, 2018-12-05 and
2025-01-09) and Juneteenth from 2022.
"""

from __future__ import annotations

import pandas as pd

HOLIDAYS = pd.DatetimeIndex([
    "2016-01-01", "2016-01-18", "2016-02-15", "2016-03-25", "2016-05-30", "2016-07-04", "2016-09-05",
    "2016-11-24", "2016-12-26",
    "2017-01-02", "2017-01-16", "2017-02-20", "2017-04-14", "2017-05-29", "2017-07-04", "2017-09-04",
    "2017-11-23", "2017-12-25",
    "2018-01-01", "2018-01-15", "2018-02-19", "2018-03-30", "2018-05-28", "2018-07-04", "2018-09-03",
    "2018-11-22", "2018-12-05", "2018-12-25",
    "2019-01-01", "2019-01-21", "2019-02-18", "2019-04-19", "2019-05-27", "2019-07-04", "2019-09-02",
    "2019-11-28", "2019-12-25",
    "2020-01-01", "2020-01-20", "2020-02-17", "2020-04-10", "2020-05-25", "2020-07-03", "2020-09-07",
    "2020-11-26", "2020-12-25",
    "2021-01-01", "2021-01-18", "2021-02-15", "2021-04-02", "2021-05-31", "2021-07-05", "2021-09-06",
    "2021-11-25", "2021-12-24",
    "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20", "2022-07-04", "2022-09-05",
    "2022-11-24", "2022-12-26",
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", "2023-06-19", "2023-07-04",
    "2023-09-04", "2023-11-23", "2023-12-25",
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04",
    "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03",
    "2026-09-07", "2026-11-26", "2026-12-25",
])

HALF_DAYS = pd.DatetimeIndex([
    "2016-11-25",
    "2017-07-03", "2017-11-24",
    "2018-07-03", "2018-11-23", "2018-12-24",
    "2019-07-03", "2019-11-29", "2019-12-24",
    "2020-11-27", "2020-12-24",
    "2021-11-26",
    "2022-11-25",
    "2023-07-03", "2023-11-24",
    "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-11-27", "2026-12-24",
])

FIRST, LAST = pd.Timestamp("2016-01-01"), pd.Timestamp("2026-12-31")
OPEN = pd.Timedelta(hours=9, minutes=30)
FULL_CLOSE = pd.Timedelta(hours=16)
EARLY_CLOSE = pd.Timedelta(hours=13)


def sessions(start, end) -> pd.DatetimeIndex:
    """Every NYSE session from `start` to `end`, both inclusive (dates, no time)."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if start < FIRST or end > LAST:
        raise ValueError(f"the calendar covers {FIRST.date()} to {LAST.date()}, not {start.date()} to {end.date()}")
    days = pd.bdate_range(start, end)
    return days[~days.isin(HOLIDAYS)]


def is_half_day(dates) -> pd.Series | bool:
    """True on a 1:00 pm early close."""
    if isinstance(dates, (pd.Timestamp, str)):
        return pd.Timestamp(dates).normalize() in HALF_DAYS
    return pd.DatetimeIndex(dates).normalize().isin(HALF_DAYS)


def close_time(dates):
    """The session's close as a time of day: 16:00, or 13:00 on a half-day."""
    half = is_half_day(dates)
    if isinstance(half, bool):
        return EARLY_CLOSE if half else FULL_CLOSE
    return pd.TimedeltaIndex([EARLY_CLOSE if h else FULL_CLOSE for h in half])


def slots(dates, minutes: int = 5):
    """5-minute bars in the session: 78 on a full day, 42 on a half-day."""
    step = pd.Timedelta(minutes=minutes)
    half = is_half_day(dates)
    full, early = (FULL_CLOSE - OPEN) // step, (EARLY_CLOSE - OPEN) // step
    if isinstance(half, bool):
        return early if half else full
    return pd.Index([early if h else full for h in half])
