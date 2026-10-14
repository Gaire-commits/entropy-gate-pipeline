"""E10: the overnight hold must be the next day's price, and growth must be judged on the same moments."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capacity import paired_ic_difference
from src.horizons import OFFSETS, collect_overnight, growth_verdict
from src.synthetic import ar_returns, bars_from_returns


def _bars(n=30, sessions=6, seed0=0):
    return {f"S{i:02d}": bars_from_returns(f"S{i:02d}", ar_returns(sessions, 0.0, seed=seed0 + i), seed=seed0 + i,
                                           start="2024-01-02") for i in range(n)}


def test_overnight_rows_use_the_signal_bar_the_next_days_exit_and_never_the_same_days_close():
    bars = _bars()
    frame = collect_overnight(bars, min_peers=5)
    assert set(frame["bar_index"]) == set(OFFSETS)
    assert frame["date"].nunique() == 5                       # six sessions: the last one has no next day
    sym = "S03"
    df = bars[sym]
    ts = df.index.get_level_values("timestamp")
    day = ts.tz_localize(None).normalize() if ts.tz is not None else ts.normalize()
    close = df["close"].to_numpy()
    first = np.flatnonzero(df["bar_index"].to_numpy() == 0)
    row = frame[(frame["symbol"] == sym) & (frame["bar_index"] == 12)].sort_values("date").iloc[1]
    d = int(np.flatnonzero(day[first] == row["date"])[0])
    entry, exit_ = close[first[d] + 13], close[first[d + 1] + 13]
    assert row["ret"] == pytest.approx(np.log(exit_ / entry), abs=1e-6)
    window = np.log(close[first[d] + 12] / close[first[d] + 6])
    market = [np.log(bars[s]["close"].to_numpy()[first[d] + 12] / bars[s]["close"].to_numpy()[first[d] + 6]) for s in bars]
    assert row["own"] == pytest.approx(-(window - np.mean(market)), abs=1e-5)


def test_a_gap_in_the_entry_or_exit_bar_drops_the_stock_not_the_day():
    bars = _bars(n=12)
    sym = "S01"
    df = bars[sym].copy()
    nxt = np.flatnonzero(df["bar_index"].to_numpy() == 0)[1] + 13           # day two, the exit bar of day one's offset 12
    df.iloc[nxt, df.columns.get_loc("close")] = np.nan
    bars[sym] = df
    frame = collect_overnight(bars, min_peers=5)
    gone = frame[(frame["symbol"] == sym) & (frame["bar_index"] == 12)]
    assert pd.Timestamp("2024-01-02") not in set(gone["date"])
    assert (frame[frame["bar_index"] == 12]["date"] == pd.Timestamp("2024-01-02")).sum() == 11


def _moments(strength, days=500, n=60, seed=0):
    rng = np.random.default_rng(seed)
    d, s = np.repeat(np.arange(days * 4), n), np.tile(np.arange(n), days * 4)
    sig = rng.normal(size=days * 4 * n)
    return pd.DataFrame({"date": pd.Timestamp("2018-01-01") + pd.to_timedelta(d // 4, unit="D"), "bar_index": 6 + 6 * (d % 4),
                         "symbol": s.astype(str), "prob": sig}).assign(ret=lambda f: strength * f["prob"] + rng.normal(size=len(f)))


def test_growth_is_judged_on_the_same_moments_and_found_when_it_is_there():
    base, same, grown = _moments(0.04, seed=1), None, None
    rng = np.random.default_rng(2)
    same = base.assign(ret=0.04 * base["prob"] + rng.normal(size=len(base)))
    grown = base.assign(ret=0.09 * base["prob"] + rng.normal(size=len(base)))
    half = base["date"] < base["date"].unique()[250]
    def judge(longer):
        p = paired_ic_difference(longer, base, min_peers=10, n_boot=300)
        e = paired_ic_difference(longer[half], base[half], min_peers=10, n_boot=200, level=0.95)
        l = paired_ic_difference(longer[~half], base[~half], min_peers=10, n_boot=200, level=0.95)
        return growth_verdict({"t": 5.0}, {}, p, e, l)["level"]
    assert judge(grown) == "keeps growing with the hold"
    assert judge(same) == "does not keep growing"


def test_growth_verdict_tiers():
    ok = {"diff": 0.01, "lo": 0.004, "hi": 0.016}
    assert growth_verdict({"t": 1.0}, {}, ok, {}, {})["level"] == "no reversal at these signal times"
    assert growth_verdict({"t": 5.0}, {}, {"diff": 0.003, "lo": -0.001, "hi": 0.007}, {}, {})["level"] == "does not keep growing"
    assert growth_verdict({"t": 5.0}, {}, ok, {"diff": 0.01}, {"diff": -0.002})["level"] == "not stable"
    assert growth_verdict({"t": 5.0}, {}, ok, {"diff": 0.01}, {"diff": 0.004})["level"] == "keeps growing with the hold"
    fade = {"diff": -0.01, "lo": -0.016, "hi": -0.004}
    assert growth_verdict({"t": 5.0}, {}, fade, {"diff": -0.01}, {"diff": -0.003})["level"] == "fades with the hold"


def test_script_runs_end_to_end_and_reports_both_decisive_comparisons(tmp_path, monkeypatch):
    import yaml

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import holds as script

    (tmp_path / "bars" / "5Min").mkdir(parents=True)
    symbols = [f"S{i:02d}" for i in range(40)]
    for i, s in enumerate(symbols):
        grid = bars_from_returns(s, ar_returns(40, 0.0, seed=i), seed=i, start="2024-01-02")
        grid.drop(columns=["session_id", "bar_index"]).dropna().to_parquet(tmp_path / "bars" / "5Min" / f"{s}.parquet")
    pd.DataFrame({"symbol": symbols, "sector": [f"G{i % 4}" for i in range(40)]}).to_csv(tmp_path / "u.csv", index=False)
    cfg = yaml.safe_load((root / "configs" / "sip_e10_holds.yaml").read_text())
    cfg["data"].update(universe=str(tmp_path / "u.csv"), cache_dir=str(tmp_path / "bars"))
    cfg["holds"].update(split="2024-02-01", min_peers=10)
    cfg["evaluation"]["bootstrap"] = 100
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["holds.py", "--config", "c.yaml"])
    assert script.main() == 0
    text = (tmp_path / "outputs" / "sip_e10_holds" / "holds.md").read_text()
    assert text.count("NO REVERSAL AT THESE SIGNAL TIMES") == 2
    assert "4 hours against 30 minutes" in text.replace("4 hours against", "4 hours against") or "4 hours" in text
    assert "a full day" in text and "Every hold" in text and "each signal time" in text
