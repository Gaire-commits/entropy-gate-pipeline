"""E12: correlated peers, the residual, the random-peer control, planted followers and the script."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.capacity import paired_ic_difference
from src.peers import collect, peer_mean, pick_peers, random_peers, relative_returns, residualize, verdict
from src.synthetic import BARS, bars_from_returns


def group_returns(n_sessions: int, groups: int = 3, size: int = 10, lag: int = 0, phi: float = 0.0,
                  scale: float = 0.001, seed: int = 0) -> dict[str, np.ndarray]:
    """Stocks in groups that share a factor. With `lag`, half of each group (the followers) also take the
    factor `lag` bars later, so their peers' recent move tells where they are going. `phi` adds each
    stock's own bar-to-bar reversal (negative) or trend."""
    rng = np.random.default_rng(seed)
    n = n_sessions * BARS
    out = {}
    for g in range(groups):
        f = rng.normal(0, scale, n)
        for i in range(size):
            e = rng.normal(0, scale, n)
            r = f + e
            if lag and i >= size // 2:
                r = r + np.r_[np.zeros(lag), f[:-lag]]
            if phi:
                for t in range(1, n):
                    r[t] += phi * e[t - 1]
            out[f"G{g}S{i:02d}"] = r
    return out


def to_bars(returns: dict[str, np.ndarray]) -> dict[str, pd.DataFrame]:
    return {s: bars_from_returns(s, r, seed=k, start="2024-01-02") for k, (s, r) in enumerate(returns.items())}


def test_pick_peers_finds_the_planted_groups_and_skips_thin_stocks():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(500, 3))
    hist = np.column_stack([f[:, g] + 0.7 * rng.normal(size=500) for g in range(3) for _ in range(6)])
    hist[:300, 0] = np.nan                                   # stock 0 traded in only 40% of the bars
    peers, rho, ok = pick_peers(hist, k=4)
    assert not ok[0] and (peers[0] == -1).all() and 0 not in peers
    for i in range(1, 18):
        assert set(peers[i] // 6) == {i // 6} and i not in peers[i]
    assert np.all(np.diff(rho[1:], axis=1) <= 1e-12)          # best first


def test_random_peers_are_distinct_others_among_the_eligible():
    ok = np.array([True] * 8 + [False] * 2)
    r = random_peers(ok, 3, np.random.default_rng(1))
    for i in range(8):
        assert len(set(r[i])) == 3 and i not in r[i] and ok[r[i]].all()
    assert (r[8:] == -1).all()


def test_peer_mean_needs_enough_valid_peers():
    own = np.array([1.0, 2.0, 3.0, 4.0])
    valid = np.array([True, True, False, True])
    peers = np.array([[1, 2, 3], [0, 2, -1], [0, 1, 3], [0, 1, 2]])
    m = peer_mean(own, valid, peers, min_valid=2)
    assert m[0] == pytest.approx(3.0) and np.isnan(m[1]) and m[2] == pytest.approx(7 / 3) and m[3] == pytest.approx(1.5)


def test_relative_returns_remove_the_market_bar_by_bar():
    close = np.exp(np.cumsum(np.array([[0.0, 0.0, 0.0], [0.01, 0.02, np.nan], [0.0, -0.01, 0.01]]), axis=0))
    r = relative_returns(close)
    assert r[0, :2] == pytest.approx([-0.005, 0.005]) and np.isnan(r[0, 2]) and np.isnan(r[1, 2])


def test_residual_is_blind_to_the_own_move():
    rng = np.random.default_rng(0)
    n = 40 * 50
    frame = pd.DataFrame({"date": np.repeat(pd.date_range("2024-01-01", periods=40), 50), "bar_index": 6,
                          "own": rng.normal(size=n)})
    frame["copy"] = 2 * frame["own"] + 1
    frame["mix"] = frame["own"] + rng.normal(size=n)
    assert residualize(frame, "copy").abs().max() < 1e-6
    res = residualize(frame, "mix")
    by = pd.DataFrame({"r": res, "o": frame.groupby("date")["own"].rank()}).groupby(frame["date"]).corr().xs("r", level=1)["o"]
    assert by.abs().max() < 1e-6


def test_verdict_follows_the_rule():
    up, down, flat = {"diff": 0.003, "lo": 0.001, "hi": 0.005}, {"diff": -0.003, "lo": -0.005, "hi": -0.001}, \
        {"diff": 0.0, "lo": -0.002, "hi": 0.002}
    assert verdict(flat, up, up)["level"] == "correlated peers add nothing"
    v = verdict(up, up, {"diff": 0.001})
    assert v["level"] == "stocks follow their correlated peers" and v["live"]
    assert verdict(up, up, {"diff": -0.001})["level"] == "not stable"
    v = verdict(down, down, down)
    assert v["level"] == "stocks move against their correlated peers" and not v["live"]


def _primary(frame: pd.DataFrame, k: int = 5):
    f = frame[np.isfinite(frame["ret_6"])].assign(ret=lambda x: x["ret_6"]).reset_index(drop=True)
    pick = lambda c: f[["date", "bar_index", "symbol", "ret"]].assign(prob=residualize(f, c).to_numpy())
    return paired_ic_difference(pick(f"peers_{k}"), pick(f"random_{k}"), min_peers=10, n_boot=300, level=0.99)


def test_planted_followers_are_found_and_beat_random_peers():
    bars = to_bars(group_returns(90, lag=6, seed=1))
    frame, refreshes = collect(bars, {s: s[:2] for s in bars}, peer_days=20)
    assert refreshes["same_sector"].mean() > 0.95 and refreshes["peer_corr"].mean() > 0.3
    assert frame["date"].min() > pd.Timestamp("2024-01-02") + pd.Timedelta(days=27)   # moments wait for 20 days of history
    d = _primary(frame)
    assert d["lo"] > 0, d


@pytest.mark.parametrize("phi", [0.0, -0.3])
def test_no_false_following_from_shared_moves_or_the_own_reversal(phi):
    bars = to_bars(group_returns(120, lag=0, phi=phi, seed=2))
    frame, _ = collect(bars, {s: s[:2] for s in bars}, peer_days=20)
    d = _primary(frame)
    assert d["lo"] <= 0 <= d["hi"], d


def test_script_writes_a_verdict(tmp_path, monkeypatch):
    import yaml
    import peers as script

    (tmp_path / "bars" / "5Min").mkdir(parents=True)
    bars = to_bars(group_returns(40, groups=3, size=10, seed=3))
    for s, grid in bars.items():
        grid.drop(columns=["session_id", "bar_index"]).dropna().to_parquet(tmp_path / "bars" / "5Min" / f"{s}.parquet")
    pd.DataFrame({"symbol": list(bars), "sector": [s[:2] for s in bars]}).to_csv(tmp_path / "u.csv", index=False)
    cfg = yaml.safe_load((ROOT / "configs" / "sip_e12_peers.yaml").read_text())
    cfg["data"].update(universe=str(tmp_path / "u.csv"), cache_dir=str(tmp_path / "bars"))
    cfg["peers"].update(peer_days=10, min_peers=10, split="2024-02-15")
    cfg["evaluation"]["bootstrap"] = 100
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["peers.py", "--config", "c.yaml"])
    assert script.main() == 0
    text = (tmp_path / "outputs" / "sip_e12_peers" / "peers.md").read_text()
    assert "**Verdict:" in text and "## The rule" in text and "sector average (E5's peers)" in text
    assert (tmp_path / "outputs" / "sip_e12_peers" / "refreshes.csv").exists()
