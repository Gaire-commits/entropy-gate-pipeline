"""E5: the gap must find a planted sector lead, and must not mistake a stock's own reversal for one."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capacity import paired_ic_difference
from src.crosssection import cross_sectional_ic
from src.leadlag import SIGNALS, _peer_mean, collect, day_signals, pick_leaders, shuffle_labels, verdict
from src.synthetic import ar_returns, bars_from_returns

BARS, N, SECTORS = 78, 60, 6


def _frame(kind="noise", days=60, seed=0):
    """lead: half the stocks of each sector get its move at once, the other half 7 bars later (inside the
    hold). same_time: sectors move together with no delay. bounce: each stock's price carries its own
    passing wobble that fades over a few bars, so its own return reverses."""
    rng = np.random.default_rng(seed)
    sector = np.repeat(np.arange(SECTORS), N // SECTORS)
    shuffled = shuffle_labels(sector, 1)
    leaders = pick_leaders(rng.random(N), sector, 3)
    slots = np.full(N, BARS)
    parts = []
    for d in range(days):
        f = rng.normal(0, 8e-4, (BARS + 7, SECTORS))
        e = rng.normal(0, 1e-3, (BARS, N))
        if kind == "lead":
            late = np.arange(N) % 2 == 1
            r = np.where(late, f[:-7, sector], f[7:, sector]) + e
        elif kind == "same_time":
            r = f[7:, sector] + e
        elif kind == "bounce":
            x = np.zeros((BARS + 1, N))
            for t in range(1, BARS + 1):
                x[t] = 0.8 * x[t - 1] + rng.normal(0, 1.5e-3, N)
            r = e + np.diff(x, axis=0)
        else:
            r = e
        close = 100 * np.exp(np.cumsum(r, axis=0))
        day = day_signals(close, slots, sector, shuffled, leaders)
        parts.append(pd.DataFrame({"date": pd.Timestamp("2022-01-03") + pd.Timedelta(days=d),
                                   "bar_index": day["bar_index"], "symbol": day["stock"].astype(int).astype(str),
                                   **{k: day[k] for k in ["ret"] + SIGNALS}}))
    return pd.concat(parts, ignore_index=True)


def _score(frame):
    gap = cross_sectional_ic(frame.assign(prob=frame["gap"]), n_boot=300)
    paired = paired_ic_difference(frame.assign(prob=frame["gap"]), frame.assign(prob=frame["gap_shuffled"]),
                                  n_boot=300, level=0.95)
    return gap, paired


def test_peer_mean_leaves_the_stock_itself_out_and_needs_enough_peers():
    own = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    groups, valid = np.array([0, 0, 0, 1, 1]), np.ones(5, bool)
    np.testing.assert_allclose(_peer_mean(own, groups, valid, 2, 1), [2.5, 2.0, 1.5, 20.0, 10.0])
    assert np.isnan(_peer_mean(own, groups, valid, 2, 2)[3:]).all()          # one other is not enough
    lead = np.array([True, False, False, True, True])
    np.testing.assert_allclose(_peer_mean(own, groups, valid, 2, 1, lead)[:3], [np.nan, 1.0, 1.0])


def test_signals_never_read_a_bar_after_the_signal_bar_and_skip_stale_prices():
    rng = np.random.default_rng(0)
    sector = np.repeat(np.arange(2), 6)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, (BARS, 12)), axis=0))
    args = (np.full(12, BARS), sector, shuffle_labels(sector, 1), np.ones(12, bool))
    base = day_signals(close, *args)
    later = close.copy()
    later[40:] *= 1.05                                           # everything after bar 39 changes
    moved = day_signals(later, *args)
    early = base["bar_index"] <= 39
    for k in SIGNALS:
        np.testing.assert_array_equal(base[k][early], moved[k][: early.sum()])
    stale = close.copy()
    stale[10, 0] = np.nan                                        # stock 0 did not trade in bar 10
    out = day_signals(stale, *args)
    hit = (out["stock"] == 0) & (out["bar_index"] == 12)         # bar 10 is inside that moment's lookback
    assert not hit.any() and ((out["stock"] == 0) & (out["bar_index"] == 24)).any()


def test_a_sector_that_leads_is_found_and_real_sectors_beat_shuffled_ones():
    gap, paired = _score(_frame("lead"))
    assert gap["t"] > 6 and paired["lo"] > 0


def test_noise_and_same_time_sector_moves_give_nothing_to_trade():
    for kind in ("noise", "same_time"):
        gap, _ = _score(_frame(kind))
        assert abs(gap["t"]) < 3, kind


def test_a_stocks_own_reversal_is_not_mistaken_for_lead_lag():
    gap, paired = _score(_frame("bounce"))
    assert gap["t"] > 3 and not paired["lo"] > 0
    assert verdict(gap, paired, {"ic": 0.01}, {"ic": 0.01}, {}, 2.0)["level"] == "own reversal, not lead-lag"


def test_verdict_grades_a_real_effect_by_stability_and_cost():
    gap, paired = {"t": 5.0}, {"lo": 0.001}
    assert verdict({"t": 1.0}, paired, {}, {}, {}, 2.0)["level"] == "no catch-up effect"
    assert verdict(gap, paired, {"ic": 0.01}, {"ic": -0.01}, {}, 2.0)["level"] == "not stable"
    assert verdict(gap, paired, {"ic": 0.01}, {"ic": 0.01}, {"gross_lo": 1.0}, 2.0)["level"] == "catches up, does not pay"
    assert verdict(gap, paired, {"ic": 0.01}, {"ic": 0.01}, {"gross_lo": 2.5}, 2.0)["level"] == "catches up and pays"


def test_collect_runs_over_cached_bars_and_picks_leaders_from_past_volume_only():
    bars = {f"S{i:02d}": bars_from_returns(f"S{i:02d}", ar_returns(25, 0.0, seed=i), seed=i, start="2024-01-02")
            for i in range(12)}
    sectors = {s: ("A" if i < 6 else "B") for i, s in enumerate(bars)}
    frame = collect(bars, sectors, min_sector=3)
    assert set(["date", "bar_index", "symbol", "ret"] + SIGNALS) <= set(frame.columns) and len(frame) > 2000
    first_days = frame["date"] < frame["date"].unique()[20]
    assert frame.loc[first_days, "gap_leaders"].isna().all()     # no 20-day volume history yet: no leaders
    assert frame.loc[~first_days, "gap_leaders"].notna().any() and frame["gap"].notna().all()


def test_script_runs_end_to_end_and_writes_a_verdict(tmp_path, monkeypatch):
    import yaml

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import leadlag as script

    (tmp_path / "bars" / "5Min").mkdir(parents=True)
    symbols = [f"S{i:02d}" for i in range(24)]
    for i, s in enumerate(symbols):
        grid = bars_from_returns(s, ar_returns(40, 0.0, seed=i), seed=i, start="2024-01-02")
        grid.drop(columns=["session_id", "bar_index"]).dropna().to_parquet(tmp_path / "bars" / "5Min" / f"{s}.parquet")
    pd.DataFrame({"symbol": symbols, "sector": [f"G{i % 3}" for i in range(24)]}).to_csv(tmp_path / "u.csv", index=False)
    cfg = yaml.safe_load((root / "configs" / "sip_e5_leadlag.yaml").read_text())
    cfg["data"].update(universe=str(tmp_path / "u.csv"), cache_dir=str(tmp_path / "bars"))
    cfg["leadlag"].update(split="2024-02-01", min_peers=10)
    cfg["evaluation"]["bootstrap"] = 100
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["leadlag.py", "--config", "c.yaml"])
    assert script.main() == 0
    text = (tmp_path / "outputs" / "sip_e5_leadlag" / "leadlag.md").read_text()
    assert "**Verdict: NO CATCH-UP EFFECT**" in text and "shuffled sectors (control)" in text and "## Reported" in text
