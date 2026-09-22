"""IBKR captures: ticks -> bars the pipeline can read, and depth that can be rebuilt from disk."""

import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ibkr import apply_depth_op, capture_grids, quotes, read_ticks, replay_book, ticks_to_bars
from src.screening import screen_universe

ET = "America/New_York"


def _ticks(rows):
    df = pd.DataFrame(rows, columns=["timestamp", "symbol", "tick_type", "price", "size", "exchange"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(ET)
    return df


def _pair(ts, symbol, bid, ask, bid_size=100, ask_size=100):
    return [(ts, symbol, "bid", bid, bid_size, ""), (ts, symbol, "ask", ask, ask_size, "")]


# ------------------------------------------------------------------ quotes and bars

def test_quotes_pair_each_bid_with_the_ask_that_follows_it():
    ticks = _ticks(
        _pair("2026-09-21 10:00:01", "SPY", 100.00, 100.02)
        + _pair("2026-09-21 10:00:02", "SPY", 100.01, 100.03)
    )
    q = quotes(ticks)
    assert q[["bid", "ask"]].values.tolist() == [[100.00, 100.02], [100.01, 100.03]]
    assert q["mid"].tolist() == pytest.approx([100.01, 100.02])
    assert q["spread_bps"].iloc[0] == pytest.approx(0.02 / 100.01 * 1e4)


def test_quotes_drop_crossed_empty_and_orphaned_rows():
    ticks = _ticks(
        [("2026-09-21 10:00:00", "SPY", "ask", 100.5, 10, "")]            # ask with no bid before it
        + _pair("2026-09-21 10:00:01", "SPY", 100.05, 100.00)            # crossed
        + _pair("2026-09-21 10:00:02", "SPY", 0.0, 100.00)               # empty side
        + _pair("2026-09-21 10:00:03", "SPY", 100.00, 100.02)            # fine
    )
    q = quotes(ticks)
    assert len(q) == 1 and q["bid"].iloc[0] == 100.00


def test_quote_pairing_is_per_symbol_even_when_symbols_interleave():
    rows = []
    for i in range(3):
        rows += _pair(f"2026-09-21 10:00:0{i}", "AAA", 10.00 + i, 10.01 + i)
        rows += _pair(f"2026-09-21 10:00:0{i}", "BBB", 50.00 + i, 50.02 + i)
    q = quotes(_ticks(rows))
    assert (q[q.symbol == "AAA"]["ask"] - q[q.symbol == "AAA"]["bid"]).round(2).eq(0.01).all()
    assert (q[q.symbol == "BBB"]["ask"] - q[q.symbol == "BBB"]["bid"]).round(2).eq(0.02).all()


def test_bars_use_the_midpoint_and_sum_trade_volume():
    ticks = _ticks(
        _pair("2026-09-21 10:01:00", "SPY", 100.00, 100.02)     # mid 100.01 (open)
        + _pair("2026-09-21 10:02:00", "SPY", 100.10, 100.12)   # mid 100.11 (high)
        + _pair("2026-09-21 10:03:00", "SPY", 99.90, 99.92)     # mid 99.91  (low)
        + _pair("2026-09-21 10:04:30", "SPY", 100.04, 100.06)   # mid 100.05 (close)
        + [("2026-09-21 10:02:30", "SPY", "trade", 100.11, 40, "ARCA"),
           ("2026-09-21 10:03:30", "SPY", "trade", 99.91, 60, "ARCA")]
        + _pair("2026-09-21 10:05:10", "SPY", 101.00, 101.02)   # next bar
    )
    bars = ticks_to_bars(ticks)
    first = bars.iloc[0]
    assert bars.index[0][1] == pd.Timestamp("2026-09-21 10:00", tz=ET)
    assert (first["open"], first["high"], first["low"], first["close"]) == pytest.approx((100.01, 100.11, 99.91, 100.05))
    assert first["volume"] == 100 and first["trades"] == 2 and first["quotes"] == 4
    assert bars.iloc[1]["volume"] == 0 and bars.iloc[1]["open"] == pytest.approx(101.01)


def test_regular_hours_filter_and_extended_hours_option():
    ticks = _ticks(_pair("2026-09-21 09:31:00", "SPY", 100, 100.02) + _pair("2026-09-21 19:56:00", "SPY", 100, 100.30))
    assert len(ticks_to_bars(ticks)) == 1
    assert len(ticks_to_bars(ticks, regular_hours_only=False)) == 2


def test_bars_are_tz_aware_eastern_and_symbols_stay_separate():
    ticks = _ticks(_pair("2026-09-21 10:01:00", "AAA", 10, 10.01) + _pair("2026-09-21 10:02:00", "BBB", 50, 50.02))
    bars = ticks_to_bars(ticks)
    assert set(bars.index.get_level_values("symbol")) == {"AAA", "BBB"}
    assert str(bars.index.get_level_values("timestamp").tz) == ET
    assert bars.loc["AAA"]["close"].iloc[0] == pytest.approx(10.005)


def test_bars_with_trades_but_no_quote_are_dropped():
    ticks = _ticks(_pair("2026-09-21 10:01:00", "SPY", 100, 100.02) + [("2026-09-21 10:12:00", "SPY", "trade", 100.0, 5, "X")])
    assert len(ticks_to_bars(ticks)) == 1


# ------------------------------------------------------------------ reading files

def test_reads_the_first_recorder_format_as_local_eastern(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("timestamp,symbol,tick_type,price,size,exchange\n2026-09-21T19:58:23,SPY,bid,773.84,2000,\n")
    ticks = read_ticks(path)
    assert ticks["timestamp"].iloc[0] == pd.Timestamp("2026-09-21 19:58:23", tz=ET)


def test_reads_the_current_format_from_utc_and_prefers_exchange_time(tmp_path):
    path = tmp_path / "new.csv"
    path.write_text("recv_ts,exch_ts,symbol,tick_type,price,size,exchange\n"
                    "2026-09-21T14:30:00.900Z,2026-09-21T14:30:00.000Z,SPY,bid,773.84,2000,\n")
    ticks = read_ticks(path)
    assert ticks["timestamp"].iloc[0] == pd.Timestamp("2026-09-21 10:30:00", tz=ET)


def test_the_real_first_capture_produces_extended_hours_bars_only():
    path = Path.home() / "Downloads/ibkr_microstructure_data/tick_by_tick.csv"
    if not path.exists():
        pytest.skip("first capture not on this machine")
    ticks = read_ticks(path)
    assert ticks_to_bars(ticks).empty                       # 19:54-19:58 is after the close
    extended = ticks_to_bars(ticks, regular_hours_only=False)
    assert set(extended.index.get_level_values("symbol")) == {"AAPL", "SPY"}


# ------------------------------------------------------------------ into the gate

def _sessions(n, symbol="SPY", seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-09-01", periods=n)
    starts = [d + pd.Timedelta(hours=9, minutes=30 + 5 * i) for d in days for i in range(78)]
    mid = 700 * np.exp(np.cumsum(rng.normal(0, 0.0008, len(starts))))
    rows = []
    for t, m in zip(starts, mid):
        rows += _pair(t + pd.Timedelta(seconds=20), symbol, round(m - 0.005, 3), round(m + 0.005, 3))
        rows += _pair(t + pd.Timedelta(seconds=200), symbol, round(m - 0.005, 3), round(m + 0.005, 3))
        rows.append((t + pd.Timedelta(seconds=100), symbol, "trade", round(m, 2), 10, "ARCA"))
    return _ticks(rows)


GATE = SimpleNamespace(embedding_dim=4, delay=1, window=390, series="log_return",
                       n_surrogates=19, min_bars_required=300, seed=0)


def test_capture_grids_match_the_shape_the_pipeline_expects():
    grids = capture_grids(_sessions(3))
    df = grids["SPY"]
    assert len(df) == 3 * 78 and df["session_id"].nunique() == 3
    assert {"open", "high", "low", "close", "volume", "bar_index", "session_id"} <= set(df.columns)
    assert df["bar_index"].tolist() == list(range(78)) * 3
    assert df["close"].notna().all()


def test_the_first_gate_reading_needs_six_sessions():
    """Five sessions of returns plus the one that gets scored. The notebook says this to the user."""
    assert screen_universe(capture_grids(_sessions(5)), GATE).empty
    seven = screen_universe(capture_grids(_sessions(7)), GATE)
    assert len(seven) == 2


def test_a_session_still_in_progress_is_not_scored():
    ticks = _sessions(6)
    ticks = ticks[ticks["timestamp"] < ticks["timestamp"].max().normalize() + pd.Timedelta(hours=11)]
    assert capture_grids(ticks)["SPY"]["session_id"].nunique() == 5


# ------------------------------------------------------------------ depth

def test_insert_pushes_rows_down_and_delete_pulls_them_up():
    book = []
    apply_depth_op(book, 0, 0, (10.00, 1, "a"))
    apply_depth_op(book, 1, 0, (9.99, 2, "b"))
    apply_depth_op(book, 0, 0, (10.01, 3, "c"))           # new best; the old rows move down
    assert [lvl[0] for lvl in book] == [10.01, 10.00, 9.99]
    apply_depth_op(book, 0, 2, (10.01, 3, "c"))           # best is removed; the rows move back up
    assert [lvl[0] for lvl in book] == [10.00, 9.99]
    apply_depth_op(book, 1, 1, (9.99, 7, "b"))
    assert book[1][1] == 7


def test_the_book_is_trimmed_to_the_rows_that_were_requested():
    book = []
    for i in range(5):
        apply_depth_op(book, 0, 0, (float(i), 1, ""), max_rows=3)
    assert len(book) == 3


def test_replay_needs_the_operation_column():
    with pytest.raises(ValueError, match="operation"):
        replay_book(pd.DataFrame({"symbol": ["SPY"], "side": ["bid"], "position": [0],
                                  "price": [1.0], "size": [1], "market_maker": [""]}))


# ------------------------------------------------------------------ the recorder itself

ibapi = pytest.importorskip("ibapi")
from src.ibkr_capture import CaptureApp, market_open_now  # noqa: E402


@pytest.fixture
def app(tmp_path):
    return CaptureApp(tmp_path, max_rows=10)


def _drive_depth(app, rng, steps=300):
    """A stream of depth messages that inserts, updates and deletes on both sides."""
    app.depth_req_map[1] = "SPY"
    for side in (0, 1):
        for pos in range(6):
            app.updateMktDepthL2(1, pos, "ARCA", 0, side, 100.0 - pos * 0.01 * (1 if side else -1), 100 + pos, False)
    for _ in range(steps):
        side, op = int(rng.integers(0, 2)), int(rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2]))
        book = app.books["SPY"]["bid" if side == 1 else "ask"]
        pos = int(rng.integers(0, max(len(book), 1)))
        if op != 0 and not book:
            op = 0
        app.updateMktDepthL2(1, pos, "ARCA", op, side, round(100 + float(rng.normal(0, 0.05)), 2), int(rng.integers(1, 500)), False)


def test_saved_depth_replays_into_exactly_the_book_the_recorder_held(app):
    _drive_depth(app, np.random.default_rng(3))
    live = app.get_current_book("SPY")
    app.flush_to_csv()
    saved = pd.read_csv(next(app.output_dir.glob("depth_*.csv")))
    assert set(saved["operation"]) == {0, 1, 2}
    assert replay_book(saved, "SPY", max_rows=10) == live
    assert len(live["bid"]) > 3 and len(live["ask"]) > 3        # a real book, not an empty one that trivially matches


def test_tick_callbacks_write_trades_and_paired_quotes_with_utc_times(app):
    app.tick_req_map.update({2: "SPY", 3: "SPY"})
    app.tickByTickAllLast(2, 1, int(time.time()), 773.9, 55, None, "EDGEA", "")
    app.tickByTickBidAsk(3, int(time.time()), 773.81, 773.92, 2000, 2000, None)
    assert app.flush_to_csv() == (0, 3)
    ticks = pd.read_csv(next(app.output_dir.glob("ticks_*.csv")))
    assert ticks["tick_type"].tolist() == ["trade", "bid", "ask"]
    assert ticks["recv_ts"].str.endswith("Z").all() and ticks["exch_ts"].str.endswith("Z").all()
    assert len(quotes(read_ticks(ticks))) == 1


def test_each_file_gets_its_header_once_and_flush_appends(app):
    app.tick_req_map[3] = "SPY"
    for _ in range(2):
        app.tickByTickBidAsk(3, int(time.time()), 1.0, 1.1, 1, 1, None)
        app.flush_to_csv()
    lines = next(app.output_dir.glob("ticks_*.csv")).read_text().splitlines()
    assert lines[0].startswith("recv_ts") and len(lines) == 1 + 4


def test_autoflush_writes_without_being_asked(app):
    app.tick_req_map[3] = "SPY"
    app.start_autoflush(interval=0.05)
    app.tickByTickBidAsk(3, int(time.time()), 1.0, 1.1, 1, 1, None)
    time.sleep(0.4)
    assert next(app.output_dir.glob("ticks_*.csv")).read_text().count("\n") == 3
    app.stop_autoflush()


def test_errors_are_explained_and_routine_notices_are_quiet(app, capsys):
    app.depth_req_map[1] = "SPY"
    app.error(1, 10092, "Deep market data is not supported for this combination of security type/exchange")
    app.error(-1, 2104, "Market data farm connection is OK")
    out = capsys.readouterr().out
    assert "10092" in out and "isSmartDepth=True" in out and "2104" not in out


def test_depth_is_requested_as_smart_depth_and_direct_depth_names_its_exchange(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "reqMktDepth", lambda *a: calls.append(a))
    monkeypatch.setattr(app, "reqTickByTickData", lambda *a: None)
    app.subscribe("SPY")
    app.subscribe("AAPL", depth_exchange="ISLAND")
    (_, smart_contract, rows, is_smart, _), (_, direct_contract, _, direct_flag, _) = calls
    assert smart_contract.exchange == "SMART" and is_smart is True and rows == 10
    assert direct_contract.exchange == "ISLAND" and direct_flag is False


def test_market_hours_check():
    et = ZoneInfo(ET)
    assert market_open_now(datetime(2026, 9, 21, 10, 0, tzinfo=et))
    assert not market_open_now(datetime(2026, 9, 21, 19, 58, tzinfo=et))     # the first capture
    assert not market_open_now(datetime(2026, 9, 19, 11, 0, tzinfo=et))      # Saturday
