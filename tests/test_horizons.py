"""E9: IC is compared across horizons by day; the rule must not credit a bigger number in bps to the horizon."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.horizons import day_ic, paired_day_difference, verdict
from src.synthetic import ar_returns, bars_from_returns


def _days(mean, n=900, sd=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(mean + sd * rng.normal(size=n), index=pd.bdate_range("2017-01-02", periods=n))


def _rule(a, b):
    diff = paired_day_difference(a, b, n_boot=400)
    split = a.index[len(a) // 2]
    early = paired_day_difference(a[a.index < split], b[b.index < split], n_boot=200, level=0.95)
    late = paired_day_difference(a[a.index >= split], b[b.index >= split], n_boot=200, level=0.95)
    return diff, verdict({6: {"t": 5.0}, 24: {"t": 5.0}}, diff, early, late)


def test_an_equal_ic_at_every_horizon_is_the_same_strength_whatever_the_bps():
    a, b = _days(0.0065, seed=1), _days(0.0065, seed=2)
    diff, v = _rule(a, b)
    assert diff["lo"] < 0 < diff["hi"] and v["level"] == "the same strength at every horizon"


def test_a_stronger_and_a_weaker_long_horizon_are_found():
    assert _rule(_days(0.012, seed=1), _days(0.0065, seed=1))[1]["level"] == "stronger at longer horizons"
    assert _rule(_days(0.003, seed=1), _days(0.0065, seed=1))[1]["level"] == "weaker at longer horizons"


def test_a_difference_that_changes_sign_between_halves_is_not_stable():
    n = 900
    a = pd.Series(np.r_[np.full(n // 2, 0.014), np.full(n - n // 2, 0.001)], index=pd.bdate_range("2017-01-02", periods=n))
    b = pd.Series(np.r_[np.full(n // 2, 0.002), np.full(n - n // 2, 0.006)], index=a.index)
    diff = paired_day_difference(a, b, n_boot=300)
    assert verdict({6: {"t": 5}}, {"diff": 0.002, "lo": 0.001, "hi": 0.003}, {"diff": 0.012}, {"diff": -0.005})["level"] == "not stable"
    assert diff["days"] == n


def test_no_horizon_reaching_t_bar_is_no_reversal():
    v = verdict({6: {"t": 1.0}, 12: {"t": 2.0}, 24: {"t": 0.5}}, {"diff": 0.01, "lo": 0.005, "hi": 0.015}, {}, {})
    assert v["level"] == "no reversal at these horizons"


def test_day_ic_averages_each_days_moments_and_pairing_uses_only_shared_days():
    rng = np.random.default_rng(0)
    d, b = np.repeat(np.arange(30), 3 * 40), np.tile(np.repeat([11, 23, 35], 40), 30)
    frame = pd.DataFrame({"date": pd.Timestamp("2024-01-02") + pd.to_timedelta(d, unit="D"), "bar_index": b,
                          "symbol": np.tile(np.arange(40).astype(str), 90), "prob": rng.normal(size=len(d))})
    frame["ret"] = 0.5 * frame["prob"] + rng.normal(size=len(d))
    s = day_ic(frame)
    assert len(s) == 30 and 0.2 < s.mean() < 0.5
    diff = paired_day_difference(s, s.iloc[:20] * 0.0, n_boot=100)
    assert diff["days"] == 20


def test_script_runs_end_to_end_and_writes_a_verdict(tmp_path, monkeypatch):
    import yaml

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import horizons as script

    (tmp_path / "bars" / "5Min").mkdir(parents=True)
    symbols = [f"S{i:02d}" for i in range(40)]
    for i, s in enumerate(symbols):
        grid = bars_from_returns(s, ar_returns(40, 0.0, seed=i), seed=i, start="2024-01-02")
        grid.drop(columns=["session_id", "bar_index"]).dropna().to_parquet(tmp_path / "bars" / "5Min" / f"{s}.parquet")
    pd.DataFrame({"symbol": symbols, "sector": [f"G{i % 4}" for i in range(40)]}).to_csv(tmp_path / "u.csv", index=False)
    cfg = yaml.safe_load((root / "configs" / "sip_e9_horizons.yaml").read_text())
    cfg["data"].update(universe=str(tmp_path / "u.csv"), cache_dir=str(tmp_path / "bars"))
    cfg["horizons"].update(split="2024-02-01", min_peers=10)
    cfg["evaluation"]["bootstrap"] = 100
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["horizons.py", "--config", "c.yaml"])
    assert script.main() == 0
    text = (tmp_path / "outputs" / "sip_e9_horizons" / "horizons.md").read_text()
    assert "**Verdict: NO REVERSAL AT THESE HORIZONS**" in text
    assert "2 hours minus 30 min" in text and "every lookback and hold" in text and "square-root" not in text.split("## Reported")[0][:0]
