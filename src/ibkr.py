"""Turn Interactive Brokers captures into what the rest of the pipeline reads.

Two jobs, both pure pandas so they run and test without a TWS connection:

- `ticks_to_bars`: quote and trade ticks -> 5-minute bars on the same session grid
  the Alpaca data uses, so the gate and the feature code run on them unchanged.
- `replay_book`: depth messages -> the order book, using the same rules the live
  recorder applies, so a saved capture can be rebuilt exactly.

Bars are built from the quote *midpoint*, not from trade prices. Trade prices bounce
between bid and ask, which is order the gate can detect but nobody can trade; the
midpoint does not bounce. That is a different price series from the trade bars the
models were trained on, so compare the two feeds on the same day before trusting
one to stand in for the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import EASTERN, OPEN, FULL_CLOSE, to_session_grid

TIME_COLUMNS = ("exch_ts", "recv_ts", "timestamp")


def read_ticks(path_or_frame, tz: str = EASTERN) -> pd.DataFrame:
    """Read a tick CSV (or frame) and return timestamps as tz-aware US/Eastern.

    Understands both recorder formats. The current one writes ISO-8601 UTC
    (`exch_ts`, else `recv_ts`). The first version wrote naive local times in a
    `timestamp` column, which are taken to be `tz`. Wall-clock times that do not
    exist or occur twice (the DST hours, when markets are closed) are dropped.
    """
    df = pd.read_csv(path_or_frame) if not isinstance(path_or_frame, pd.DataFrame) else path_or_frame.copy()
    column = next((c for c in TIME_COLUMNS if c in df.columns and df[c].notna().any()), None)
    if column is None:
        raise ValueError(f"no timestamp column; expected one of {TIME_COLUMNS}")
    stamps = pd.to_datetime(df[column], format="ISO8601", errors="coerce")
    if stamps.dt.tz is None:
        stamps = stamps.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    else:
        stamps = stamps.dt.tz_convert(tz)
    df["timestamp"] = stamps
    return df.dropna(subset=["timestamp"]).reset_index(drop=True)


def quotes(ticks: pd.DataFrame) -> pd.DataFrame:
    """One row per quote update: timestamp, symbol, bid, ask, sizes, mid, spread in bps.

    The recorder writes every update as a bid row immediately followed by its ask
    row, so a new update starts at each bid row. Pairing that way, instead of
    carrying each side forward on its own, avoids inventing a quote made of a fresh
    bid and the previous ask. Empty (price 0) and crossed quotes are dropped.
    """
    q = ticks[ticks["tick_type"].isin(["bid", "ask"])].copy()
    q["update"] = (q["tick_type"] == "bid").groupby(q["symbol"]).cumsum()
    bid = q[q["tick_type"] == "bid"].set_index(["symbol", "update"])
    ask = q[q["tick_type"] == "ask"].set_index(["symbol", "update"])
    out = (
        bid[["timestamp", "price", "size"]].rename(columns={"price": "bid", "size": "bid_size"})
        .join(ask[["price", "size"]].rename(columns={"price": "ask", "size": "ask_size"}), how="inner")
        .reset_index()
        .drop(columns="update")
    )
    out = out[(out["bid"] > 0) & (out["ask"] >= out["bid"])].copy()
    out["mid"] = (out["bid"] + out["ask"]) / 2
    out["spread_bps"] = (out["ask"] - out["bid"]) / out["mid"] * 1e4
    return out


def ticks_to_bars(ticks: pd.DataFrame, minutes: int = 5, regular_hours_only: bool = True) -> pd.DataFrame:
    """5-minute bars per symbol: midpoint OHLC, trade volume, and the spread.

    A bar is stamped with its start time, like the Alpaca bars. Columns beyond
    open/high/low/close/volume (`spread_bps`, `quotes`, `trades`) ride along; the
    pipeline ignores them but they are the raw material for microstructure features.
    Bars with trades but no valid quote are dropped, since there is no midpoint.
    """
    freq = f"{minutes}min"

    def bar_start(ts: pd.Series) -> pd.Series:
        return ts.dt.tz_convert(EASTERN).dt.tz_localize(None).dt.floor(freq)

    q = quotes(ticks)
    if q.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "spread_bps", "quotes", "trades"])
    q["bar"] = bar_start(q["timestamp"])
    grouped = q.groupby(["symbol", "bar"])
    bars = grouped["mid"].agg(open="first", high="max", low="min", close="last")
    bars["spread_bps"] = grouped["spread_bps"].mean()
    bars["quotes"] = grouped.size()

    t = ticks[ticks["tick_type"] == "trade"].copy()
    t["bar"] = bar_start(t["timestamp"])
    tg = t.groupby(["symbol", "bar"])
    bars["volume"] = tg["size"].sum().reindex(bars.index).fillna(0.0)
    bars["trades"] = tg.size().reindex(bars.index).fillna(0).astype(int)

    if regular_hours_only:
        into_day = bars.index.get_level_values("bar") - bars.index.get_level_values("bar").normalize()
        bars = bars[(into_day >= OPEN) & (into_day < FULL_CLOSE)]

    bars = bars.sort_index()
    symbols = bars.index.get_level_values("symbol")
    stamps = bars.index.get_level_values("bar").tz_localize(EASTERN, ambiguous="NaT", nonexistent="NaT")
    bars.index = pd.MultiIndex.from_arrays([symbols, stamps], names=["symbol", "timestamp"])
    bars = bars[bars.index.get_level_values("timestamp").notna()]
    return bars[["open", "high", "low", "close", "volume", "spread_bps", "quotes", "trades"]]


def capture_grids(ticks: pd.DataFrame, minutes: int = 5) -> dict[str, pd.DataFrame]:
    """Regular-hours bars on the complete session grid, one frame per symbol.

    Same shape as `data.load_universe`, so the result feeds `screen_universe` and
    `build_dataset` directly. A session still in progress is dropped.
    """
    bars = ticks_to_bars(ticks, minutes, regular_hours_only=True)
    grids = {}
    for symbol, frame in bars.groupby(level="symbol", sort=False):
        grid = to_session_grid(frame, minutes)
        if not grid.empty:
            grids[symbol] = grid
    return grids


def apply_depth_op(levels: list, position: int, operation: int, level: tuple, max_rows: int | None = None) -> None:
    """Apply one IB depth message to one side of the book, in place.

    `position` is a row in a ranked list. An insert (0) pushes the rows below it
    down one place, an update (1) replaces the row, and a delete (2) pulls the rows
    below it up. Storing levels by position in a dict, and overwriting on every
    message, lets the book drift out of step after the first insert or delete.
    """
    if operation == 0:
        levels.insert(min(position, len(levels)), level)
    elif operation == 1:
        if position < len(levels):
            levels[position] = level
        else:
            levels.append(level)
    elif operation == 2:
        if position < len(levels):
            levels.pop(position)
    else:
        raise ValueError(f"unknown depth operation {operation}")
    if max_rows is not None:
        del levels[max_rows:]


def replay_book(depth: pd.DataFrame, symbol: str | None = None, max_rows: int | None = None) -> dict[str, list]:
    """Rebuild the book after the last message, as {"bid": [...], "ask": [...]} of (price, size, market_maker).

    Needs the `operation` column and the rows in the order they were recorded.
    """
    if "operation" not in depth.columns:
        raise ValueError(
            "depth file has no `operation` column, so inserts, updates and deletes cannot be told apart "
            "and the book cannot be rebuilt (files from the first recorder version have this problem)"
        )
    rows = depth if symbol is None else depth[depth["symbol"] == symbol]
    book = {"bid": [], "ask": []}
    for r in rows.itertuples(index=False):
        maker = "" if pd.isna(r.market_maker) else r.market_maker
        apply_depth_op(book[r.side], int(r.position), int(r.operation), (r.price, r.size, maker), max_rows)
    return book
