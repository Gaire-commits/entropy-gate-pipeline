"""The online agent: the stream must equal the batch pipeline, see nothing early, and explore on purpose."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import build_dataset
from src.online import (
    EntropyControl, Moment, OnlineConfig, PageHinkley, RunningStandardizer, book_pnl, compare_to_batch,
    run_agent, stream_events,
)
from src.synthetic import ar_returns, bars_from_returns

FIELDS = ["open", "high", "low", "close", "volume"]
ALL_CHANNELS = ["log_return", "session_return", "hl_range", "volume_z", "close_open", "signed_volume"]


def _features(**kw):
    base = dict(channels=ALL_CHANNELS, window=12, embargo=1, horizon=30, stride=12, allow_overnight=False,
                normalize="fold_standardize", label_deadzone=0.0)
    return SimpleNamespace(**{**base, **kw})


def _bars(n_symbols=20, sessions=30, gap_rate=0.04, seed=1, early_close_day=None):
    """Synthetic bars with real-data blemishes: 5-minute gaps with no trade, symbols missing whole
    days, and optionally an early close (39 bars) for everyone."""
    rng = np.random.default_rng(seed)
    bars = {}
    for i in range(n_symbols):
        df = bars_from_returns(f"S{i:02d}", ar_returns(sessions, 0.05, seed=i), seed=i, start="2024-01-02")
        days = df.index.get_level_values("timestamp").normalize()
        unique = days.unique()
        gaps = rng.random(len(df)) < gap_rate
        df.loc[gaps, FIELDS] = np.nan
        if i % 5 == 0 and len(unique) > 15:
            df = df[~days.isin(unique[[7, 15]])]
            days = df.index.get_level_values("timestamp").normalize()
        if early_close_day is not None:
            df = df[~((days == unique[early_close_day]) & (df["bar_index"].to_numpy() >= 39))]
        bars[f"S{i:02d}"] = df
    return bars


# ---------------------------------------------------------------- the stream equals the batch pipeline

@pytest.mark.parametrize("channels", [ALL_CHANNELS, ["log_return", "session_return", "hl_range", "volume_z"]])
def test_stream_reproduces_every_batch_sample_with_gaps_and_missing_days(channels):
    bars = _bars()
    f = _features(channels=channels)
    moments, _ = stream_events(bars, f, min_peers=5, keep_windows=True)
    data = build_dataset(bars, None, SimpleNamespace(features=f))
    check = compare_to_batch(moments, data)
    assert check["missing_in_stream"] == 0 and check["batch"] > 800
    assert check["max_window_diff"] < 1e-6 and check["max_return_diff"] < 1e-12
    # the stream decides on a few more: those whose entry or exit bar had no trade, which batch drops
    assert check["extra_with_finite_return"] == 0


def test_an_early_close_leaves_no_decision_whose_exit_would_fall_after_it():
    bars = _bars(early_close_day=10)
    f = _features()
    moments, _ = stream_events(bars, f, min_peers=5, keep_windows=True)
    day = sorted({m.date for m in moments.values()})
    early = pd.Timestamp(bars["S01"].index.get_level_values("timestamp").normalize().unique()[10].date())
    assert all(pd.Timestamp(m.date) != early for m in moments.values())
    assert compare_to_batch(moments, build_dataset(bars, None, SimpleNamespace(features=f)))["missing_in_stream"] == 0
    assert len(day) > 20


def test_the_stream_is_causal_cutting_off_the_future_changes_nothing_before_it():
    bars = _bars(seed=3)
    f = _features()
    full, _ = stream_events(bars, f, min_peers=5)
    cut_day = sorted({m.date for m in full.values()})[15]
    cut = {s: df[df.index.get_level_values("timestamp").tz_localize(None).normalize() <= pd.Timestamp(cut_day)]
           for s, df in bars.items()}
    part, _ = stream_events(cut, f, min_peers=5)
    assert len(part) and all(part[k].date <= cut_day for k in part)
    for k, m in part.items():
        twin = full[k]
        assert np.array_equal(m.symbols, twin.symbols) and np.array_equal(m.F, twin.F)
        np.testing.assert_array_equal(m.ret, twin.ret)


def test_a_return_is_released_exactly_at_its_exit_bar_and_never_earlier():
    bars = _bars(n_symbols=8, sessions=6, gap_rate=0.0)
    f = _features()
    moments, events = stream_events(bars, f, min_peers=5)
    decided = {mid: t for t, kind, mid in events if kind == "decide"}
    for t, kind, mid in events:
        if kind == "outcome":
            assert t - decided[mid] == f.embargo + f.horizon
    kinds = [(t, k) for t, k, _ in events]
    for (t1, k1), (t2, k2) in zip(kinds, kinds[1:]):          # at one bar: outcomes, then decisions
        assert t1 < t2 or (k1, k2) != ("decide", "outcome")


def test_running_normalization_uses_only_earlier_moments():
    rng = np.random.default_rng(0)
    scaler = RunningStandardizer(3, halflife=50)
    assert np.all(scaler.transform(rng.normal(size=(5, 3))) == 0)       # nothing seen yet: no information
    for _ in range(400):
        scaler.update(rng.normal(5.0, 2.0, size=(50, 3)))
    np.testing.assert_allclose(scaler.mean, 5.0, atol=0.3)
    np.testing.assert_allclose(np.sqrt(scaler.var), 2.0, atol=0.3)
    before = scaler.mean.copy()
    z = scaler.transform(np.full((2, 3), 1e6))
    assert np.all(z == 5) and np.array_equal(scaler.mean, before)       # transform never updates


def test_cross_section_normalization_needs_enough_peers():
    bars = _bars(n_symbols=8)
    moments, _ = stream_events(bars, _features(), min_peers=20)
    assert not moments
    moments, _ = stream_events(bars, _features(), normalize="running")
    assert moments


def test_unsupported_settings_fail_loudly():
    with pytest.raises(ValueError, match="allow_overnight"):
        stream_events(_bars(n_symbols=3, sessions=3), _features(allow_overnight=True))
    with pytest.raises(ValueError, match="channels"):
        stream_events(_bars(n_symbols=3, sessions=3), _features(channels=["vibes"]))
    with pytest.raises(ValueError, match="feedback"):
        run_agent({}, [], 4, OnlineConfig(feedback="psychic"))


# ---------------------------------------------------------------- exploration control

def test_page_hinkley_catches_a_drop_quickly_and_stays_quiet_on_noise():
    rng = np.random.default_rng(0)
    quiet = PageHinkley()
    assert sum(quiet.update(x) for x in rng.normal(size=5000)) <= 1
    ph = PageHinkley()
    for x in rng.normal(size=500):
        ph.update(x)
    hits = [i for i, x in enumerate(rng.normal(-1.5, 1.0, size=200)) if ph.update(x)]
    assert hits and hits[0] < 40


def test_entropy_control_raises_beta_when_too_sure_lowers_it_when_too_random_and_can_be_off():
    c = EntropyControl(target=0.5, beta=0.05)
    assert c.step(0.1) > 0.05
    c = EntropyControl(target=0.5, beta=0.05)
    assert c.step(1.0) < 0.05
    c.alarm()
    assert c.current_target() == pytest.approx(0.9)
    for _ in range(c.boost_moments):
        c.tick()
    assert c.current_target() == pytest.approx(0.5)
    off = EntropyControl(enabled=False)
    assert off.step(0.0) == 0.0


# ---------------------------------------------------------------- the agent

def _moments(n_mom=1500, n=80, d=6, edge=6.0, noise=60.0, flip_at=None, seed=0, lag=3):
    """Moments where E[return] = edge * feature 0 (bps), optionally flipping sign mid-stream.
    Each outcome arrives `lag` moments after its decision, as overlapping holds do."""
    rng = np.random.default_rng(seed)
    moments, events = {}, []
    symbols = np.array([f"S{i}" for i in range(n)])
    for k in range(n_mom):
        F = rng.normal(size=(n, d)).astype(np.float32)
        e = edge if flip_at is None or k < flip_at else -edge
        moments[k] = Moment(mid=k, date=np.datetime64("2024-01-01") + k // 3, bar_index=11 + 12 * (k % 3),
                            symbols=symbols, F=F, ret=(e * F[:, 0] + noise * rng.normal(size=n)) / 1e4)
    for t in range(n_mom + lag):
        if t >= lag:
            events.append((t, "outcome", t - lag))
        if t < n_mom:
            events.append((t, "decide", t))
    return moments, events


def _net(dec, book="greedy", last=None, cost=2.0):
    d = dec if last is None else dec[dec["mid"] > dec["mid"].max() - last]
    pos = d[f"pos_{book}"].to_numpy()
    return float(np.mean(pos * d["ret_bps"].to_numpy() - cost * np.abs(pos))), float(np.mean(pos != 0))


def test_a_return_is_read_only_at_its_outcome_event(monkeypatch):
    """Every read of a moment's return is logged with how many decisions had been made by then;
    it must be exactly the number of decide events before that moment's outcome event."""
    from src import online

    moments, events = _moments(n_mom=200, edge=0.0)
    log, decided = [], [0]

    class Watched(Moment):
        @property
        def ret(self):
            log.append((self.mid, decided[0]))
            return self._ret

        @ret.setter
        def ret(self, value):
            self._ret = value

    watched = {k: Watched(**vars(m)) for k, m in moments.items()}
    original = online.OnlineAgent.decide

    def counting(self, mid, F):
        decided[0] += 1
        return original(self, mid, F)

    monkeypatch.setattr(online.OnlineAgent, "decide", counting)
    run_agent(watched, events, 6, OnlineConfig(min_buffer=500))
    expected, before = {}, 0
    for _, kind, mid in events:
        if kind == "decide":
            before += 1
        else:
            expected[mid] = before
    assert len(log) == len(expected) and all(n == expected[mid] for mid, n in log)


@pytest.mark.parametrize("feedback", ["full", "bandit"])
def test_agent_learns_a_planted_cross_sectional_signal(feedback):
    moments, events = _moments(edge=6.0)
    dec, trace = run_agent(moments, events, 6, OnlineConfig(feedback=feedback))
    net, rate = _net(dec[dec["scored"]])
    assert net > 1.5 and rate > 0.4
    assert dec["scored"].sum() < len(dec) and trace["entropy"].between(0, np.log(3) + 1e-6).all()


def test_agent_makes_nothing_on_noise_and_the_placebo_world_looks_the_same():
    moments, events = _moments(edge=0.0)
    dec, _ = run_agent(moments, events, 6, OnlineConfig(feedback="full"))
    assert _net(dec[dec["scored"]])[0] < 0.2
    signal, ev = _moments(edge=6.0)
    real, _ = run_agent(signal, ev, 6, OnlineConfig(feedback="full"))
    placebo, _ = run_agent(signal, ev, 6, OnlineConfig(feedback="full"), placebo_seed=1)
    assert _net(real[real["scored"]])[0] > 1.5 and _net(placebo[placebo["scored"]])[0] < 0.2


def test_placebo_shuffles_returns_only_across_stocks_within_a_moment():
    moments, events = _moments(n_mom=300, edge=6.0)
    real, _ = run_agent(moments, events, 6, OnlineConfig(min_buffer=10**9))
    fake, _ = run_agent(moments, events, 6, OnlineConfig(min_buffer=10**9), placebo_seed=4)
    a = real.groupby("mid")["ret_bps"].apply(lambda r: np.sort(r.to_numpy()))
    b = fake.groupby("mid")["ret_bps"].apply(lambda r: np.sort(r.to_numpy()))
    assert all(np.allclose(x, y) for x, y in zip(a, b))
    assert not np.allclose(real["ret_bps"], fake["ret_bps"])


@pytest.mark.parametrize("feedback", ["full", "bandit"])
def test_entropy_control_lets_the_agent_find_a_signal_again_after_it_flips(feedback):
    """Without it the policy locks in (softmax saturates) and, once trading loses, settles on flat,
    which teaches a bandit learner nothing. With it the agent keeps sampling trades and relearns."""
    results = {}
    for enabled in (True, False):
        nets = []
        for seed in range(2):
            moments, events = _moments(edge=6.0, flip_at=750, seed=10 + seed)
            cfg = OnlineConfig(feedback=feedback, seed=seed, entropy=EntropyControl(enabled=enabled))
            dec, _ = run_agent(moments, events, 6, cfg)
            nets.append(_net(dec, last=300)[0])
        results[enabled] = nets
    assert min(results[True]) > 1.0, results
    assert max(results[False]) < 0.5, results


def test_the_frozen_book_stops_learning_at_the_end_of_warm_up():
    moments, events = _moments(edge=6.0, flip_at=750, seed=11)
    dec, _ = run_agent(moments, events, 6, OnlineConfig(feedback="full"))
    assert (dec.loc[~dec["scored"], "pos_frozen"] == 0).all()
    after = dec[dec["mid"] > 1000]
    assert _net(after, "frozen")[0] < -1.0 < 1.0 < _net(after, "greedy")[0]


def test_book_pnl_charges_cost_per_position_on_scored_rows_only():
    dec = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"] * 3), "ret_bps": [10.0, -4.0, 7.0],
                        "scored": [True, True, False], "pos_greedy": [1.0, -1.0, 1.0]})
    out = book_pnl(dec, "greedy", 2.0)
    assert out["pnl_bps"].tolist() == [8.0, 2.0]


def test_replay_script_runs_end_to_end_on_a_cached_universe(tmp_path, monkeypatch, capsys):
    import yaml

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import online_replay

    (tmp_path / "data" / "bars" / "5Min").mkdir(parents=True)
    symbols = [f"S{i:02d}" for i in range(24)]
    for i, s in enumerate(symbols):
        grid = bars_from_returns(s, ar_returns(40, 0.05, seed=i), seed=i, start="2024-01-02")
        grid.drop(columns=["session_id", "bar_index"]).dropna().to_parquet(tmp_path / "data" / "bars" / "5Min" / f"{s}.parquet")
    pd.DataFrame({"symbol": symbols, "sector": "X"}).to_csv(tmp_path / "u.csv", index=False)
    pd.DataFrame({"symbol": symbols[:5], "sector": "X"}).to_csv(tmp_path / "few.csv", index=False)
    cfg = yaml.safe_load((root / "configs" / "online_sp500.yaml").read_text())
    cfg["data"].update(universe=str(tmp_path / "u.csv"), cache_dir=str(tmp_path / "data" / "bars"))
    cfg["online"].update(warmup_moments=20, min_buffer=200, placebos=1, placebos_other=0, check_symbols=6)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["online_replay.py", "--config", "c.yaml", "--only", "few.csv"])
    assert online_replay.main() == 0
    report = (tmp_path / "outputs" / "online_sp500" / "online_few.md").read_text()
    assert "reproduces every one" in report and "## Verdict" in report and "**A null**" in report
    for name in ("full", "bandit", "bandit_no_entropy"):
        assert (tmp_path / "outputs" / "online_sp500" / f"online_trace_{name}_few.csv").exists()
