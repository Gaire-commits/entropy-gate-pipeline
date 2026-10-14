"""E11: the mid-cap list parser, the replication rule, and the two-universe script end to end."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.horizons import hold_frames, replication_verdict
from src.synthetic import ar_returns, bars_from_returns


def test_sp400_parser_finds_the_constituents_table_among_other_tables():
    from build_universe import parse_sp400

    noise = pd.DataFrame({"Year": [2024], "Change": ["added"]})
    changes = pd.DataFrame({"Date": ["2024-03-01"], "Ticker": ["XYZ"], "Reason": ["merger"]})
    table = pd.DataFrame({"Symbol": [" ACM", "AIT "], "Security": ["AECOM", "Applied Industrial"],
                          "GICS Sector": ["Industrials", "Industrials"], "GICS Sub-Industry": ["Eng", "Trading"]})
    out = parse_sp400([noise, changes, table])
    assert list(out["symbol"]) == ["ACM", "AIT"] and list(out["sector"]) == ["Industrials", "Industrials"]


def test_replication_verdict_follows_the_rule():
    up, down = {"lo": 0.001, "hi": 0.006, "diff": 0.004}, {"lo": -0.008, "hi": -0.001, "diff": -0.004}
    flat = {"lo": -0.002, "hi": 0.003, "diff": 0.0}
    size_up = {"lo": 0.001, "hi": 0.005}
    v = replication_verdict({"t": 5.0}, up, down, size_up)
    assert v["level"] == "replicates in full" and v["size"] == "stronger in mid caps" and v["claims"] == 3
    v = replication_verdict({"t": 5.0}, flat, down, flat)
    assert v["level"] == "replicates in part" and "1-hour IC is not above" in v["reason"] and v["size"].startswith("the same")
    v = replication_verdict({"t": 5.0}, up, flat, {"lo": -0.006, "hi": -0.001})
    assert v["level"] == "replicates in part" and "full-day IC is not below" in v["reason"] and v["size"] == "weaker in mid caps"
    v = replication_verdict({"t": 1.5}, up, down, size_up)
    assert v["level"] == "does not replicate" and v["claims"] == 0


def test_hold_frames_share_one_set_of_moments_across_holds():
    bars = {f"S{i:02d}": bars_from_returns(f"S{i:02d}", ar_returns(8, 0.0, seed=i), seed=i, start="2024-01-02") for i in range(30)}
    frames = hold_frames(bars, {s: "A" for s in bars}, holds=(6, 12, 24, 48), min_peers=5)
    assert set(frames) == {6, 12, 24, 48, "day"}
    keys = {h: set(zip(f["date"], f["bar_index"])) for h, f in frames.items() if h != "day"}
    assert keys[6] == keys[12] == keys[24] == keys[48]
    assert set(frames[6]["bar_index"]) == {6, 12, 18, 24} and (frames["day"]["prob"] == frames["day"]["own"]).all()


def test_script_runs_on_two_universes_and_reports_a_verdict(tmp_path, monkeypatch):
    import yaml
    import second_sample as script

    (tmp_path / "bars" / "5Min").mkdir(parents=True)
    names = {"mid": [f"M{i:02d}" for i in range(40)], "big": [f"B{i:02d}" for i in range(40)]}
    for k, group in enumerate(names.values()):
        for i, s in enumerate(group):
            grid = bars_from_returns(s, ar_returns(40, 0.0, seed=100 * k + i), seed=100 * k + i, start="2024-01-02")
            grid.drop(columns=["session_id", "bar_index"]).dropna().to_parquet(tmp_path / "bars" / "5Min" / f"{s}.parquet")
        pd.DataFrame({"symbol": group + (["B00"] if k == 0 else []), "sector": [f"G{j % 4}" for j in range(len(group) + (k == 0))]}).to_csv(
            tmp_path / f"{'mid' if k == 0 else 'big'}.csv", index=False)                       # B00 is on both lists
    cfg = yaml.safe_load((ROOT / "configs" / "sip_e11_midcap.yaml").read_text())
    cfg["data"].update(universe=str(tmp_path / "mid.csv"), cache_dir=str(tmp_path / "bars"))
    cfg["second"].update(compare_universe=str(tmp_path / "big.csv"))
    cfg["holds"].update(split="2024-02-01", min_peers=10)
    cfg["evaluation"]["bootstrap"] = 100
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["second_sample.py", "--config", "c.yaml"])
    assert script.main() == 0
    text = (tmp_path / "outputs" / "sip_e11_midcap" / "second_sample.md").read_text()
    assert "**Replication: DOES NOT REPLICATE**" in text and "**Size: THE SAME IN MID CAPS AND LARGE CAPS**" in text
    assert "The three claims, on mid caps" in text and "Every hold, both samples" in text
    assert "1 stock(s) on both lists (B00) kept in the S&P 500 sample only" in text
