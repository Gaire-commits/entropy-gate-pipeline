"""Gate timing. A reading computed at a session's close must not admit windows
from that same session, because live, that reading does not exist yet."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import build_dataset
from src.screening import apply_gate, screen_symbol

BARS = 78
SESSIONS = 8


def _bars(symbol="TEST", seed=0):
    rng = np.random.default_rng(seed)
    n = BARS * SESSIONS
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    days = pd.bdate_range("2024-03-04", periods=SESSIONS)
    offsets = pd.timedelta_range("09:30:00", periods=BARS, freq="5min")
    ts = pd.DatetimeIndex([d + o for d in days for o in offsets]).tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
            "volume": rng.lognormal(10, 0.5, n),
            "session_id": np.repeat(np.arange(SESSIONS), BARS),
        },
        index=pd.MultiIndex.from_arrays([np.full(n, symbol), ts], names=["symbol", "timestamp"]),
    )


def _screen(df):
    table = screen_symbol(df, window=2 * BARS, m=3, min_bars_required=100)
    table.insert(1, "symbol", "TEST")
    return table


def _cfg():
    return SimpleNamespace(
        features=SimpleNamespace(
            channels=["log_return", "hl_range", "close_open"],
            window=24, horizon=6, embargo=1, label_deadzone=0.0,
            normalize="window_zscore", encoding="1d", allow_overnight=False,
        )
    )


def test_reading_applies_to_the_following_session():
    table = _screen(_bars())
    dated = table.dropna(subset=["trade_date"])
    assert (dated["trade_date"] > dated["session_date"]).all()

    session_days = sorted(table["session_date"].unique())
    for _, row in dated.iterrows():
        assert row["trade_date"] == session_days[session_days.index(row["session_date"]) + 1]


def test_last_reading_has_no_trade_date_in_sample():
    table = _screen(_bars())
    assert pd.isna(table.iloc[-1]["trade_date"])
    assert table["trade_date"].iloc[:-1].notna().all()


def test_a_reading_never_admits_windows_from_the_day_it_was_computed():
    """The leak this guards against: an 11am window admitted by entropy measured at 4pm."""
    df = _bars()
    table = apply_gate(_screen(df), select_quantile=1.0)
    computed_on = table["session_date"].iloc[2]
    table["passed"] = table["session_date"] == computed_on

    data = build_dataset({"TEST": df}, table, _cfg(), use_gate=True)
    sample_days = set(pd.to_datetime(data["date"]).normalize())

    assert computed_on not in sample_days
    assert sample_days == {table["trade_date"].iloc[2]}


def test_stale_screen_table_without_trade_date_is_rejected():
    df = _bars()
    table = apply_gate(_screen(df)).drop(columns="trade_date")
    with pytest.raises(ValueError, match="trade_date"):
        build_dataset({"TEST": df}, table, _cfg(), use_gate=True)
