"""The feed comparison: IEX's grid has gaps and a sliver of the volume; consolidated bars have both."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.feed_compare import compare_feeds, gridded, summarize
from src.synthetic import ar_returns, bars_from_returns


def _raw(symbol="SPY", sessions=8, seed=0):
    grid = bars_from_returns(symbol, ar_returns(sessions, 0.0, seed=seed), seed=seed, start="2024-03-04")
    return grid.drop(columns=["session_id", "bar_index"]).dropna()


def _iex_from(sip_raw, seed=0, keep=0.65, volume_share=0.03):
    rng = np.random.default_rng(seed)
    thin = sip_raw[rng.random(len(sip_raw)) < keep].copy()
    thin["volume"] = np.round(thin["volume"] * volume_share * rng.uniform(0.5, 1.5, len(thin)))
    thin["close"] = thin["close"] * (1 + rng.normal(0, 2e-5, len(thin)))      # a stray cent: the same trades, priced alike
    return thin


def test_consolidated_bars_fill_the_grid_and_carry_far_more_volume():
    sip = _raw()
    iex = _iex_from(sip)
    out = compare_feeds(gridded(iex), gridded(sip))
    assert out["sip_completeness"] > 0.99 and 0.55 < out["iex_completeness"] < 0.75
    assert out["iex_sessions"] == out["sip_sessions"] == 8
    assert 20 < out["volume_ratio"] < 60
    assert out["close_diff_bps_median"] < 1.0 and out["return_corr"] > 0.9


def test_a_feed_with_no_bars_is_reported_not_raised():
    out = compare_feeds(None, gridded(_raw()))
    assert out["iex_sessions"] == 0 and out["sip_sessions"] == 8 and np.isnan(out["volume_ratio"])
    assert gridded(pd.DataFrame()) is None


def test_the_script_writes_a_report_and_survives_one_feed_refusing(tmp_path, monkeypatch):
    import feed_check

    def fake_fetch(client, symbol, start, end, timeframe, feed, adjustment):
        if feed == "sip" and start.startswith("2016") and symbol == "HAS":
            raise RuntimeError("subscription does not permit querying this data")
        sip = _raw(symbol, seed=hash(symbol) % 100)
        return _iex_from(sip) if feed == "iex" else sip

    monkeypatch.setattr(feed_check, "get_client", lambda: object())
    monkeypatch.setattr(feed_check, "fetch_symbol", fake_fetch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["feed_check.py", "--symbols", "SPY", "HAS"])
    assert feed_check.main() == 0
    text = (tmp_path / "outputs" / "feed_check.md").read_text()
    assert "2016 (earliest SIP)" in text and "| recent |" in text
    assert "subscription does not permit" in text and "SIP history from 2016 available on this account:** yes" in text
