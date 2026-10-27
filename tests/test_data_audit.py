"""The read-only data audit: counts on bars with known gaps, an unadjusted split, and a thinner second feed."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import save_symbol
from src.synthetic import BARS, bars_from_returns

import data_audit as audit


def write(cache: Path, symbol: str, grid: pd.DataFrame, feed: str) -> None:
    raw = grid.drop(columns=["session_id", "bar_index"]).dropna()
    save_symbol(raw, cache, symbol, "5Min", "2024-01-02", "2024-06-28", feed, "split")


def make(tmp_path):
    rng = np.random.default_rng(0)
    sip, iex = tmp_path / "sip", tmp_path / "iex"
    for i in range(4):
        r = rng.normal(0, 0.001, 60 * BARS)
        g = bars_from_returns(f"S{i}", r, seed=i, start="2024-01-02")
        if i == 0:                                   # a split the adjustment missed: prices halve from day 30
            cut = g.index[29 * BARS + 0]
            g.loc[g.index >= cut, ["open", "high", "low", "close"]] /= 2
        if i == 1:                                   # ten missing market days in a row
            g = g[~g["session_id"].between(20, 29)]
        write(sip, f"S{i}", g, "sip")
        thin = g.copy()
        thin["high"] = thin["high"] - 0.05           # IEX sees a subset of the trades
        thin["low"] = thin["low"] + 0.05
        thin["volume"] = thin["volume"] * 0.05
        write(iex, f"S{i}", thin.iloc[::2], "iex")
    pd.DataFrame({"symbol": [f"S{i}" for i in range(4)], "name": "x", "sector": "x"}).to_csv(tmp_path / "u.csv", index=False)
    return sip, iex


def test_audit_finds_the_split_the_gap_and_the_feed_difference(tmp_path):
    sip, iex = make(tmp_path)
    s, _ = audit.load_feed(str(sip), [f"S{i}" for i in range(4)])
    i, _ = audit.load_feed(str(iex), [f"S{i}" for i in range(4)])
    cal = audit.calendar(s)
    cov = audit.coverage(s, cal)
    sym = cov["symbols"].set_index("symbol")
    assert sym.loc["S1", "longest_run"] >= 9 and sym.loc["S0", "longest_run"] == 0
    j = audit.jumps(s)
    assert (j[j["symbol"] == "S0"]["split_like_ratio"] == 2).any()
    c = audit.feed_comparison(s, i)
    assert c["iex_high_above_sip_share"] == 0 and c["median_iex_to_sip_volume"] < 0.2
    assert audit.variance_parts(s)["overnight_share_of_ovn_plus_park"].between(0, 1).all()


def test_script_writes_a_report(tmp_path, monkeypatch):
    sip, iex = make(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["data_audit.py", "--sip", str(sip), "--iex", str(iex), "--universe", str(tmp_path / "u.csv")])
    assert audit.main() == 0
    text = (tmp_path / "outputs" / "data_audit" / "report.md").read_text()
    assert "## 3. Suspected unadjusted splits" in text and "## Not done here" in text
