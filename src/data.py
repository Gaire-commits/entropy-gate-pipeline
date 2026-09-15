"""Alpaca historical bars -> local parquet cache.

Credentials are read from the environment (or a local .env) and never passed as
arguments, so no key can end up in a config file, a notebook, or a traceback.

Free-tier notes: the `iex` feed carries only IEX-routed volume, roughly a few
percent of consolidated tape, so its volume series is a sample rather than the
true figure. Prices track well for liquid names and degrade for thin ones --
which matters here, because thin names are exactly where the entropy screen is
most likely to be fooled by stale quotes.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
EASTERN = "America/New_York"


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
    spec = spec.strip()
    digits = "".join(c for c in spec if c.isdigit()) or "1"
    word = "".join(c for c in spec if c.isalpha()).lower()
    if word not in units:
        raise ValueError(f"unknown timeframe '{spec}'; expected e.g. 1Min, 5Min, 1Hour, 1Day")
    return TimeFrame(int(digits), units[word])


def fetch_bars(
    symbols: list[str],
    start: str | datetime,
    end: str | datetime,
    timeframe: str = "5Min",
    feed: str = "iex",
    adjustment: str = "split",
) -> pd.DataFrame:
    """Download bars for symbols. Returns a long frame indexed by (symbol, timestamp)."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest

    client = get_client()
    request = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=parse_timeframe(timeframe),
        start=pd.Timestamp(start).to_pydatetime(),
        end=pd.Timestamp(end).to_pydatetime(),
        feed=DataFeed(feed),
        adjustment=Adjustment(adjustment),
    )
    bars = client.get_stock_bars(request)
    df = bars.df
    if df.empty:
        return df
    return df.sort_index()


def to_eastern(df: pd.DataFrame) -> pd.DataFrame:
    """Move the timestamp level of the index to US/Eastern."""
    ts = df.index.get_level_values("timestamp")
    ts = ts.tz_convert(EASTERN) if ts.tz is not None else ts.tz_localize("UTC").tz_convert(EASTERN)
    return df.set_index(
        pd.MultiIndex.from_arrays([df.index.get_level_values("symbol"), ts], names=["symbol", "timestamp"])
    )


def regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 09:30-16:00 ET bars.

    Overnight and pre-market bars have their own microstructure; leaving them in
    makes the entropy screen read the session boundary as structure.
    """
    ts = df.index.get_level_values("timestamp")
    mask = (ts.time >= pd.Timestamp(MARKET_OPEN).time()) & (ts.time < pd.Timestamp(MARKET_CLOSE).time())
    return df[mask]


def add_session_id(df: pd.DataFrame) -> pd.DataFrame:
    """Integer id per trading date, so windows can be kept inside one session."""
    dates = df.index.get_level_values("timestamp").normalize()
    out = df.copy()
    out["session_id"] = pd.factorize(dates)[0]
    return out


def cache_path(cache_dir: str | Path, symbol: str, timeframe: str) -> Path:
    return Path(cache_dir) / timeframe / f"{symbol}.parquet"


def save_symbol(df: pd.DataFrame, cache_dir: str | Path, symbol: str, timeframe: str) -> Path:
    path = cache_path(cache_dir, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def load_symbol(cache_dir: str | Path, symbol: str, timeframe: str) -> pd.DataFrame | None:
    path = cache_path(cache_dir, symbol, timeframe)
    return pd.read_parquet(path) if path.exists() else None


def load_universe(cache_dir: str | Path, symbols: list[str], timeframe: str) -> dict[str, pd.DataFrame]:
    out = {}
    for sym in symbols:
        df = load_symbol(cache_dir, sym, timeframe)
        if df is not None and not df.empty:
            out[sym] = df
    return out
