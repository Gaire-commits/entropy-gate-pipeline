"""LIVE_PLAN step 1: the forward test's length, analytic and from historical windows."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forward_power import combine, days_needed, window_pass_rates


def test_days_needed_matches_the_textbook_formula_and_scales_with_the_square():
    z = norm.ppf(0.95) + norm.ppf(0.8)
    assert days_needed(0.01, 0.08) == pytest.approx((z * 8) ** 2)
    assert days_needed(0.005, 0.08) == pytest.approx(4 * days_needed(0.01, 0.08))
    assert days_needed(0.0, 0.08) == np.inf and days_needed(-0.01, 0.08) == np.inf


def test_windows_pass_at_the_rate_the_formula_predicts():
    rng = np.random.default_rng(0)
    sd, ic = 0.08, 0.01
    n = int(round(days_needed(ic, sd)))
    days = pd.Series(rng.normal(ic, sd, 40 * n))
    r = window_pass_rates(days, (n,), step=n).iloc[0]
    assert 0.65 < r["share_above"] < 0.95
    assert r["share_rule"] <= r["share_above"]


def test_windows_on_noise_pass_about_five_percent_and_too_long_windows_are_empty():
    rng = np.random.default_rng(1)
    days = pd.Series(rng.normal(0, 0.08, 20000))
    r = window_pass_rates(days, (100, 30000), step=100)
    assert r.iloc[0]["share_above"] == pytest.approx(0.05, abs=0.03)
    assert r.iloc[1]["windows"] == 0 and np.isnan(r.iloc[1]["share_above"])


def test_combine_averages_over_shared_days_only():
    a = pd.Series([0.1, 0.2, 0.3], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    b = pd.Series([0.3, 0.0], index=pd.to_datetime(["2024-01-03", "2024-01-05"]))
    c = combine(a, b)
    assert list(c.index) == [pd.Timestamp("2024-01-03")] and c.iloc[0] == pytest.approx(0.25)


def test_script_runs_on_two_universes(tmp_path, monkeypatch):
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import forward_power as script
    from src.synthetic import ar_returns, bars_from_returns

    root = Path(__file__).resolve().parents[1]
    (tmp_path / "bars" / "5Min").mkdir(parents=True)
    for k, prefix in enumerate(("M", "B")):
        group = [f"{prefix}{i:02d}" for i in range(30)]
        for i, s in enumerate(group):
            grid = bars_from_returns(s, ar_returns(30, -0.1, seed=100 * k + i), seed=100 * k + i, start="2024-01-02")
            grid.drop(columns=["session_id", "bar_index"]).dropna().to_parquet(tmp_path / "bars" / "5Min" / f"{s}.parquet")
        pd.DataFrame({"symbol": group, "sector": [f"G{j % 3}" for j in range(30)]}).to_csv(tmp_path / f"{prefix}.csv", index=False)
    cfg = yaml.safe_load((root / "configs" / "sip_e11_midcap.yaml").read_text())
    cfg["data"].update(universe=str(tmp_path / "M.csv"), cache_dir=str(tmp_path / "bars"))
    cfg["second"].update(compare_universe=str(tmp_path / "B.csv"))
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["forward_power.py", "--config", "c.yaml", "--min-peers", "10"])
    assert script.main() == 0
    text = (tmp_path / "outputs" / "forward_power" / "power.md").read_text()
    assert "How long must the forward test run?" in text and "both, averaged" in text
    days = pd.read_csv(tmp_path / "outputs" / "forward_power" / "days.csv")
    assert set(days["universe"]) == {"S&P 500", "mid caps", "both, averaged"} and days["geometry"].nunique() == 3
