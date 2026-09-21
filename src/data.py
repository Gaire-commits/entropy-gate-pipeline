"""Alpaca historical bars -> local parquet cache -> a fixed intraday grid.

Credentials are read from the environment (or a local .env) and never passed as
arguments, so no key can end up in a config file, a notebook, or a traceback.

Free-tier notes: the `iex` feed carries only IEX-routed volume, a few percent of
the consolidated tape, so volume is a sample rather than the true figure. A bar
only exists when a trade printed on IEX, so quiet intervals come back missing.
`to_session_grid` puts those gaps back as NaN rows: without that, a missing bar
silently shifts every later bar, and "the last 6 bars of the day" stops meaning
3:30 to 4:00.
"""

from __future__ import annotations

import json
import os
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

EASTERN = "America/New_York"
OPEN = pd.Timedelta(hours=9, minutes=30)
FULL_CLOSE = pd.Timedelta(hours=16)
EARLY_CLOSE = pd.Timedelta(hours=13)


def _load_dotenv(path: str | Path = ".env") -> None:
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def get_client():
    """Build an Alpaca data client from env credentials."""
    from alpaca.data.historical import StockHistoricalDataClient

    _load_dotenv()
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Alpaca credentials not found. Copy .env.example to .env and fill in "
            "APCA_API_KEY_ID and APCA_API_SECRET_KEY, or export them in your shell."
        )
    return StockHistoricalDataClient(key, secret)


def resolve_end(end) -> pd.Timestamp:
    """Exclusive upper bound for a download, from a config `end` value.

    An explicit date is inclusive: "2026-09-21" includes that whole session.
    "today" runs up to the current moment, so a run after the close picks the
    session up. "yesterday" stops at this midnight and never touches a session
    still in progress.
    """
    text = str(end).strip().lower()
    now = pd.Timestamp.now(tz=EASTERN).tz_localize(None)
    if text in ("today", "now"):
        return now
    if text == "yesterday":
        return now.normalize()
    return pd.Timestamp(end).normalize() + pd.Timedelta(days=1)


def is_rolling_end(end) -> bool:
    return str(end).strip().lower() in ("today", "now", "yesterday")


def merge_bars(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append newly downloaded bars to cached ones, newest copy winning on overlap."""
    if existing is None or existing.empty:
        return new
    if new is None or new.empty:
        return existing
    combined = pd.concat([existing, new]).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def bar_minutes(spec: str) -> int:
    """'5Min' -> 5. Intraday grids only make sense for minute bars."""
    digits = "".join(c for c in spec if c.isdigit()) or "1"
    word = "".join(c for c in spec if c.isalpha()).lower()
    if word != "min":
        raise ValueError(f"intraday pipeline needs minute bars, got '{spec}'")
    return int(digits)


def parse_timeframe(spec: str):
    """'5Min' -> TimeFrame(5, Minute)."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    units = {
        "min": TimeFrameUnit.Minute,
        "hour": TimeFrameUnit.Hour,
        "day": TimeFrameUnit.Day,
        "week": TimeFrameUnit.Week,
        "month": TimeFrameUnit.Month,
    }
    digits = "".join(c for c in spec if c.isdigit()) or "1"
    word = "".join(c for c in spec if c.isalpha()).lower()
    if word not in units:
        raise ValueError(f"unknown timeframe '{spec}'; expected e.g. 1Min, 5Min, 1Hour, 1Day")
    return TimeFrame(int(digits), units[word])


def fetch_symbol(
    client,
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "5Min",
    feed: str = "iex",
    adjustment: str = "split",
) -> pd.DataFrame:
    """Download one symbol in yearly chunks, so a long history never rides on one request."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest

    frames = []
    cursor, stop = pd.Timestamp(start), pd.Timestamp(end)
    while cursor < stop:
        chunk_end = min(cursor + pd.DateOffset(years=1), stop)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=parse_timeframe(timeframe),
            start=cursor.to_pydatetime(),
            end=chunk_end.to_pydatetime(),
            feed=DataFeed(feed),
            adjustment=Adjustment(adjustment),
        )
        df = client.get_stock_bars(request).df
        if not df.empty:
            frames.append(df)
        cursor = chunk_end
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def to_eastern(df: pd.DataFrame) -> pd.DataFrame:
    """Move the timestamp level of the index to US/Eastern."""
    ts = df.index.get_level_values("timestamp")
    ts = ts.tz_convert(EASTERN) if ts.tz is not None else ts.tz_localize("UTC").tz_convert(EASTERN)
    return df.set_index(
        pd.MultiIndex.from_arrays([df.index.get_level_values("symbol"), ts], names=["symbol", "timestamp"])
    )


def regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 09:30-16:00 ET bars.

    Pre- and post-market bars have their own microstructure; leaving them in
    makes the entropy screen read the session boundary as structure.
    """
    ts = df.index.get_level_values("timestamp")
    mask = (ts.time >= time(9, 30)) & (ts.time < time(16, 0))
    return df[mask]


def to_session_grid(df: pd.DataFrame, minutes: int = 5) -> pd.DataFrame:
    """Reindex one symbol onto a complete bar grid per trading day, NaN where IEX had no trade.

    Adds `session_id` and `bar_index` (0 = the 09:30 bar). A day whose last
    observed bar starts before 13:00 is treated as an early close; every real
    early close on the US calendar ends at 13:00.

    A final session still in progress is dropped. Downloading at 11am would
    otherwise leave a half day that looks like a complete one, and "the last
    half-hour" would quietly mean 10:30. The cost is dropping a genuine early
    close when it happens to be the very last day in the file.
    """
    if df.empty:
        return df
    symbol = df.index.get_level_values("symbol")[0]
    ts = df.index.get_level_values("timestamp")
    naive = ts.tz_convert(EASTERN).tz_localize(None) if ts.tz is not None else ts

    days = naive.normalize()
    last_bar = pd.Series(naive, index=naive).groupby(days).max()
    if (last_bar.iloc[-1] - last_bar.index[-1]) < pd.Timedelta(hours=15, minutes=30):
        keep_rows = np.asarray(days < last_bar.index[-1])
        df, naive = df[keep_rows], naive[keep_rows]
        last_bar = last_bar.iloc[:-1]
        if last_bar.empty:
            return df
    early = (last_bar - last_bar.index) < EARLY_CLOSE

    step = pd.Timedelta(minutes=minutes)
    offsets = pd.timedelta_range(OPEN, FULL_CLOSE - step, freq=step)
    grid = last_bar.index.values[:, None] + offsets.values[None, :]
    keep = ~(early.values[:, None] & (offsets.values[None, :] >= EARLY_CLOSE.to_timedelta64()))

    grid_index = pd.DatetimeIndex(grid[keep].ravel())
    values = df.drop(columns=[c for c in ("session_id", "bar_index") if c in df.columns])
    values = values.set_axis(naive, axis=0).reindex(grid_index)

    out_days = grid_index.normalize()
    values["session_id"] = pd.factorize(out_days)[0]
    values["bar_index"] = ((grid_index - out_days) - OPEN) // step
    values.index = pd.MultiIndex.from_arrays(
        [np.full(len(grid_index), symbol), grid_index.tz_localize(EASTERN)], names=["symbol", "timestamp"]
    )
    return values


def cache_path(cache_dir: str | Path, symbol: str, timeframe: str) -> Path:
    return Path(cache_dir) / timeframe / f"{symbol}.parquet"


def _meta_path(cache_dir: str | Path, symbol: str, timeframe: str) -> Path:
    return cache_path(cache_dir, symbol, timeframe).with_suffix(".meta.json")


def read_meta(cache_dir, symbol, timeframe) -> dict | None:
    meta = _meta_path(cache_dir, symbol, timeframe)
    if not meta.exists() or not cache_path(cache_dir, symbol, timeframe).exists():
        return None
    return json.loads(meta.read_text())


def plan_fetch(cache_dir, symbol, timeframe, start, end, feed, adjustment) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """What to download for one symbol, or None when the cache already covers it.

    Nothing cached, or cached with a different feed or adjustment: download the
    whole range. Otherwise download only from the last cached day onward and
    merge. The last cached day is fetched again because it may have been saved
    while its session was still in progress. Coverage is judged from what was
    requested, not from the first bar returned, so a fund that launched after
    `start` does not get re-downloaded on every run.
    """
    stop = resolve_end(end)
    m = read_meta(cache_dir, symbol, timeframe)
    usable = m and m.get("feed") == feed and m.get("adjustment") == adjustment and pd.Timestamp(m["start"]) <= pd.Timestamp(start)
    if not usable:
        return pd.Timestamp(start), stop
    if not is_rolling_end(end) and pd.Timestamp(m["end"]) >= stop:
        return None
    raw = pd.read_parquet(cache_path(cache_dir, symbol, timeframe))
    last_day = raw.index.get_level_values("timestamp").max().tz_localize(None).normalize()
    return last_day, stop


def save_symbol(df, cache_dir, symbol, timeframe, start, end, feed, adjustment) -> Path:
    """Save bars and record the range that was requested, keeping the earliest start ever covered."""
    prior = read_meta(cache_dir, symbol, timeframe)
    if prior and prior.get("feed") == feed and prior.get("adjustment") == adjustment:
        start = min(pd.Timestamp(prior["start"]), pd.Timestamp(start))
    path = cache_path(cache_dir, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    _meta_path(cache_dir, symbol, timeframe).write_text(
        json.dumps({"start": str(pd.Timestamp(start)), "end": str(resolve_end(end)), "feed": feed, "adjustment": adjustment})
    )
    return path


def load_raw(cache_dir: str | Path, symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Cached bars exactly as downloaded, before gridding."""
    path = cache_path(cache_dir, symbol, timeframe)
    return pd.read_parquet(path) if path.exists() else None


def load_symbol(cache_dir: str | Path, symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Cached bars for one symbol on the complete session grid, or None."""
    path = cache_path(cache_dir, symbol, timeframe)
    if not path.exists():
        return None
    raw = pd.read_parquet(path)
    if raw.empty:
        return None
    return to_session_grid(raw, bar_minutes(timeframe))


def load_universe(cache_dir: str | Path, symbols: list[str], timeframe: str) -> dict[str, pd.DataFrame]:
    out = {}
    for sym in symbols:
        df = load_symbol(cache_dir, sym, timeframe)
        if df is not None and not df.empty:
            out[sym] = df
    return out


def completeness(df: pd.DataFrame) -> float:
    """Share of grid slots that have a real bar."""
    return float(df["close"].notna().mean()) if len(df) else 0.0
