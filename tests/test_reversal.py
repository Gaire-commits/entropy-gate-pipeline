"""E6: the conditional test must find a state that sets the reversal's strength, and not be fooled by volatility."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.leadlag import collect
from src.reversal import STATES, TAIL, collect_states, conditional_ic, day_states, verdict
from src.synthetic import ar_returns, bars_from_returns


def _frame(kind="uniform", days=80, per_day=6, n=180, seed=0):
    """Stock-moments with a reversal of strength b. uniform: the same everywhere. entropy: stronger where
    pe3 is low. vol_only: stronger where vol is low, and pe3 merely moves with vol."""
    rng = np.random.default_rng(seed)
    m = days * per_day
    d, s = np.repeat(np.arange(m), n), np.tile(np.arange(n), m)
    vol = rng.normal(size=m * n)
    pe = 0.7 * vol + np.sqrt(1 - 0.49) * rng.normal(size=m * n)
    driver = {"uniform": np.zeros(m * n), "entropy": pe, "vol_only": vol}[kind]
    b = 0.08 * (1 - 0.8 * np.tanh(driver))
    signal = rng.normal(size=m * n)
    ret = b * signal + rng.normal(size=m * n)
    return pd.DataFrame({"date": pd.Timestamp("2022-01-03") + pd.to_timedelta(d // per_day, unit="D"),
                         "bar_index": 6 * (d % per_day + 1), "symbol": s.astype(str), "signal": signal, "ret": ret,
                         "pe3": pe, "vol": vol})


def _judge(frame):
    single = conditional_ic(frame, "pe3", n_boot=400)
    matched = conditional_ic(frame, "pe3", control="vol", n_boot=400, min_group=15)
    half = frame["date"] < frame["date"].unique()[len(frame["date"].unique()) // 2]
    early, late = conditional_ic(frame[half], "pe3", n_boot=200), conditional_ic(frame[~half], "pe3", n_boot=200)
    return single, matched, verdict(single, matched, early, late)


def test_a_reversal_that_depends_on_entropy_is_found_and_survives_matching_on_volatility():
    single, matched, v = _judge(_frame("entropy"))
    assert single["hi"] < 0 and matched["hi"] < 0 and single["low"] > single["high"]
    assert v["level"] == "entropy conditions the reversal (stronger when entropy is low)"


def test_a_uniform_reversal_is_not_conditioned_by_entropy():
    single, _, v = _judge(_frame("uniform"))
    assert single["lo"] < 0 < single["hi"] and single["low"] > 0.04 and single["high"] > 0.04
    assert v["level"] == "entropy does not condition the reversal"


def test_a_reversal_set_by_volatility_alone_is_not_credited_to_entropy():
    single, matched, v = _judge(_frame("vol_only"))
    assert single["hi"] < 0                                   # entropy looks like it matters...
    assert matched["lo"] < 0 < matched["hi"] or abs(matched["diff"]) < 0.5 * abs(single["diff"])
    assert v["level"] == "volatility, not entropy"            # ...until stocks of similar volatility are compared


def test_verdict_requires_the_same_sign_in_both_halves():
    single, matched = {"diff": -0.01, "lo": -0.02, "hi": -0.005}, {"diff": -0.008, "lo": -0.015, "hi": -0.001}
    assert verdict(single, matched, {"diff": -0.01}, {"diff": 0.002})["level"] == "not stable"
    assert verdict(single, matched, {"diff": -0.01}, {"diff": -0.004})["level"].startswith("entropy conditions")
    assert verdict(single, {"diff": 0.003, "lo": 0.001, "hi": 0.006}, {}, {})["level"] == "volatility, not entropy"


def _bars(n=14, sessions=8):
    return {f"S{i:02d}": bars_from_returns(f"S{i:02d}", ar_returns(sessions, 0.0, seed=i), seed=i, start="2024-01-02")
            for i in range(n)}


def test_states_come_from_bars_and_the_signal_is_e5s_own_move_reversed():
    bars = _bars()
    frame = collect_states(bars)
    assert set(["date", "bar_index", "symbol", "ret", "signal"] + STATES) <= set(frame.columns)
    ll = collect(bars, {s: "A" for s in bars}, min_sector=3)
    both = frame.merge(ll, on=["date", "bar_index", "symbol"], suffixes=("", "_e5"))
    assert len(both) == len(frame) == len(ll)
    np.testing.assert_allclose(both["signal"], both["own"], atol=1e-7)
    np.testing.assert_allclose(both["ret"], both["ret_e5"], atol=1e-7)
    first = frame["date"] == frame["date"].min()
    assert frame.loc[first & (frame["bar_index"] < 60), "pe3"].isna().all()      # not yet 60 returns of history
    later = frame[~first]
    assert later["pe3"].notna().all() and later["pe3"].between(0, 1).all() and (later["vol"] > 0).all()


def test_states_and_signals_never_use_a_bar_after_the_signal_bar():
    rng = np.random.default_rng(0)
    ext = np.cumsum(rng.normal(0, 1e-3, (TAIL + 78, 10)), axis=0)
    slots = np.full(10, 78)
    base = day_states(ext, slots)
    moved = ext.copy()
    moved[TAIL + 40:] += 0.05                                 # everything after bar 39 changes
    after = day_states(moved, slots)
    early = base["bar_index"] <= 39
    for k in ["signal"] + STATES:
        np.testing.assert_array_equal(base[k][early], after[k][: early.sum()])


def test_script_runs_end_to_end_and_writes_a_verdict(tmp_path, monkeypatch):
    import yaml

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import reversal as script

    (tmp_path / "bars" / "5Min").mkdir(parents=True)
    symbols = [f"S{i:02d}" for i in range(100)]
    for i, s in enumerate(symbols):
        grid = bars_from_returns(s, ar_returns(60, 0.0, seed=i), seed=i, start="2024-01-02")   # 12 days are too few for a 99% interval
        grid.drop(columns=["session_id", "bar_index"]).dropna().to_parquet(tmp_path / "bars" / "5Min" / f"{s}.parquet")
    pd.DataFrame({"symbol": symbols, "sector": "X"}).to_csv(tmp_path / "u.csv", index=False)
    cfg = yaml.safe_load((root / "configs" / "sip_e6_reversal.yaml").read_text())
    cfg["data"].update(universe=str(tmp_path / "u.csv"), cache_dir=str(tmp_path / "bars"))
    cfg["reversal"].update(split="2024-02-15", min_group=20, min_group_matched=8)
    cfg["evaluation"]["bootstrap"] = 100
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["reversal.py", "--config", "c.yaml"])
    assert script.main() == 0
    text = (tmp_path / "outputs" / "sip_e6_reversal" / "reversal.md").read_text()
    assert "**Verdict: ENTROPY DOES NOT CONDITION THE REVERSAL**" in text
    assert "permutation entropy (registered)" in text and "## Reported" in text


# ---------------------------------------------------------------- E7: the reversal as a slope

from src.reversal import slope_by_group, slope_verdict  # noqa: E402


def _moves(kind="uniform", days=120, per_day=6, n=200, seed=0):
    """Signals in return units with fat tails. uniform: 2% of any lag is recovered. large: 4% of a large
    lag and nothing of a small one. small: the opposite. none: nothing is recovered."""
    rng = np.random.default_rng(seed)
    m = days * per_day
    d, s = np.repeat(np.arange(m), n), np.tile(np.arange(n), m)
    signal = 1e-3 * rng.standard_t(4, m * n)
    big = np.abs(signal) > np.quantile(np.abs(signal), 0.6)
    b = {"uniform": 0.02, "large": np.where(big, 0.04, 0.0), "small": np.where(big, 0.0, 0.25), "none": 0.0}[kind]
    ret = b * signal + 3e-4 * rng.normal(size=m * n) + 5e-4 * rng.normal(size=m)[d]     # plus a market move
    return pd.DataFrame({"date": pd.Timestamp("2022-01-03") + pd.to_timedelta(d // per_day, unit="D"),
                         "bar_index": 6 * (d % per_day + 1), "symbol": s.astype(str),
                         "signal": signal, "ret": ret, "size": np.abs(signal)})


def _slopes(frame):
    overall = slope_by_group(frame, by=None, n_boot=400)
    by_size = slope_by_group(frame, n_boot=400)
    half = frame["date"] < frame["date"].unique()[len(frame["date"].unique()) // 2]
    early, late = slope_by_group(frame[half], n_boot=200), slope_by_group(frame[~half], n_boot=200)
    return overall, by_size, slope_verdict(overall, by_size, early, late)


def test_a_uniform_reversal_has_the_same_slope_whatever_the_size_though_its_ic_does_not():
    """The trap E7 exists for: the correlation rises with the size of the move, the slope does not."""
    frame = _moves("uniform")
    overall, by_size, v = _slopes(frame)
    assert overall["slope"][0] == pytest.approx(0.02, abs=0.003)
    assert by_size["diff_lo"] < 0 < by_size["diff_hi"] and v["level"] == "a uniform reversal"
    ic = conditional_ic(frame, "size", n_boot=300)
    assert ic["lo"] > 0                                        # the IC comparison "finds" a difference that is not there


def test_a_reversal_only_after_large_moves_is_found_and_so_is_its_opposite():
    _, by_size, v = _slopes(_moves("large"))
    assert by_size["diff_lo"] > 0 and by_size["slope"][-1] == pytest.approx(0.04, abs=0.006)
    assert v["level"] == "not uniform: a larger share is recovered after large moves"
    _, by_size, v = _slopes(_moves("small"))
    assert by_size["diff_hi"] < 0 and v["level"] == "not uniform: a smaller share is recovered after large moves"


def test_no_reversal_is_reported_as_none_and_the_market_move_cancels():
    overall, _, v = _slopes(_moves("none"))
    assert overall["lo"][0] < 0 < overall["hi"][0] and v["level"] == "no reversal as a slope"


def test_slope_verdict_needs_the_same_sign_in_both_halves():
    overall, by_size = {"lo": [0.01]}, {"diff": 0.02, "diff_lo": 0.005, "diff_hi": 0.04}
    assert slope_verdict(overall, by_size, {"diff": 0.03}, {"diff": -0.01})["level"] == "not stable"
    assert slope_verdict(overall, by_size, {"diff": 0.03}, {"diff": 0.01})["level"].startswith("not uniform: a larger")


def test_slope_script_runs_end_to_end_and_writes_a_verdict(tmp_path, monkeypatch):
    import yaml

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import reversal_slope as script

    (tmp_path / "bars" / "5Min").mkdir(parents=True)
    symbols = [f"S{i:02d}" for i in range(60)]
    for i, s in enumerate(symbols):
        grid = bars_from_returns(s, ar_returns(40, 0.0, seed=i), seed=i, start="2024-01-02")
        grid.drop(columns=["session_id", "bar_index"]).dropna().to_parquet(tmp_path / "bars" / "5Min" / f"{s}.parquet")
    pd.DataFrame({"symbol": symbols, "sector": "X"}).to_csv(tmp_path / "u.csv", index=False)
    cfg = yaml.safe_load((root / "configs" / "sip_e7_slope.yaml").read_text())
    cfg["data"].update(universe=str(tmp_path / "u.csv"), cache_dir=str(tmp_path / "bars"))
    cfg["slope"].update(split="2024-02-01")
    cfg["evaluation"]["bootstrap"] = 100
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["reversal_slope.py", "--config", "c.yaml"])
    assert script.main() == 0
    text = (tmp_path / "outputs" / "sip_e7_slope" / "slope.md").read_text()
    assert "**Verdict: NO REVERSAL AS A SLOPE**" in text and "largest third" in text and "By fifths of size" in text


def test_a_flag_is_used_as_the_group_it_names_not_cut_into_equal_halves():
    """The bug in the first E7 report: the largest tenth was cut as if the flag were a quantity,
    which put half of all rows in the "largest tenth"."""
    frame = _moves("large")
    flag = (frame.groupby(["date", "bar_index"])["size"].rank(pct=True) > 0.9).astype(int)
    right = slope_by_group(frame.assign(decile=flag), "decile", 2, n_boot=100, precut=True)
    assert right["n"][1] == int(flag.sum()) and right["n"][1] < 0.11 * len(frame)
    fifths = slope_by_group(frame, "size", 5, n_boot=100)
    assert right["typical_move_bps"][1] > fifths["typical_move_bps"][-1]      # a tenth is more extreme than a fifth
    wrong = slope_by_group(frame.assign(decile=flag), "decile", 2, n_boot=100)
    assert wrong["n"][1] > 0.45 * len(frame)                                  # what the report did
