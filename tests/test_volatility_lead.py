"""E1: the volatile-moment test must use only the past, and its rule must sort planted cases correctly."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.volatility_lead import LeadParams, analyse, block_mean, find_episodes, past_statistic, stress_cost, top_share


def _rows(effect="none", delta=40.0, seed=0, S=80, D=1000):
    """Stock-moments with volatility clustering in market-wide stress spells. `effect`:
    none; everywhere (the most volatile 2% earn +delta, in every period); episode (a large gain on the
    most volatile rows in one spell of the test period only); market (every stock earns +delta in stress)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-07-28", periods=D)
    m, i = np.ones(D), 0
    while i < D:
        if rng.random() < 0.02:
            n = rng.integers(4, 12)
            m[i:i + n] = rng.uniform(2.5, 4)
            i += n
        i += 1
    sig = rng.lognormal(0, 0.4, S)
    n = D * 3 * S
    d, b = np.repeat(np.arange(D), 3 * S), np.tile(np.repeat([11, 23, 35], S), D)
    s = np.tile(np.arange(S), 3 * D)
    vol = sig[s] * m[d] * np.exp(0.25 * rng.normal(size=n))
    ret = rng.normal(0, 25, D * 3)[d * 3 + b // 12] * m[d] + 50 * vol * rng.normal(size=n)
    hot = vol >= np.quantile(vol, 0.98)
    if effect == "everywhere":
        ret += delta * hot
    elif effect == "episode":
        stressed = np.flatnonzero(m > 1)
        start = stressed[np.argmax(stressed > 700)]
        ret += 40 * delta * hot * (np.abs(d - start) < 12)
    elif effect == "market":
        ret += delta * (m[d] > 1)
    return pd.DataFrame({"date": days[d], "bar_index": b, "symbol": s.astype(str), "ret_bps": ret, "vol": vol / 1e3})


P = LeadParams(n_boot=500, excluded=("2030-01-01", "2030-01-02"))


def test_cutoffs_and_medians_come_only_from_earlier_months():
    rows = _rows(D=200, S=20)
    dates, vol = rows["date"].to_numpy(), rows["vol"].to_numpy()
    base = top_share(vol, dates, 0.02, 20)
    later = rows["date"] >= pd.Timestamp("2021-02-01")
    changed = vol.copy()
    changed[later] *= 50                                   # the future explodes
    after = top_share(changed, dates, 0.02, 20)
    assert np.array_equal(base[~later.to_numpy()], after[~later.to_numpy()])
    med = past_statistic(vol, dates, np.median, 20)
    assert np.isnan(med[rows["date"] < pd.Timestamp("2020-09-01")]).all()      # burn-in: no history yet
    assert 0.002 < base[~np.isnan(med)].mean() < 0.08          # about 2%, moved by the volatility regime


def test_stress_cost_is_between_the_flat_cost_and_its_cap():
    rows = _rows(D=200, S=20)
    c = stress_cost(rows["vol"].to_numpy(), rows["date"].to_numpy(), 2.0, 4.0, 20)
    c = c[np.isfinite(c)]
    assert c.min() == pytest.approx(2.0) and c.max() == pytest.approx(8.0) and 2.0 < c.mean() < 8.0


def test_block_mean_pools_rows_and_counts_empty_days_in_the_calendar():
    cal = np.array(pd.bdate_range("2024-01-01", periods=100).values, dtype="datetime64[D]")
    dates = np.repeat(cal[::2], 3)
    values = np.tile([1.0, 2.0, 3.0], 50)
    r = block_mean(dates, values, cal, block_days=5, n_boot=300)
    assert r["mean"] == pytest.approx(2.0) and r["rows"] == 150 and r["days"] == 50 and r["calendar_days"] == 100
    assert r["lo"] <= 2.0 <= r["hi"]
    rng = np.random.default_rng(0)
    clustered = np.repeat(rng.normal(0, 1, 20), 5)        # five-day runs: blocks must widen the interval
    wide = block_mean(cal, clustered * 10, cal, block_days=5, n_boot=500)
    narrow = block_mean(cal, clustered * 10, cal, block_days=1, n_boot=500)
    assert (wide["hi"] - wide["lo"]) > 1.5 * (narrow["hi"] - narrow["lo"])


def test_episodes_merge_nearby_hot_days_and_split_distant_ones():
    counts = np.zeros(60)
    counts[[5, 7, 9]] = 10
    counts[[40, 41]] = 12
    counts[[20, 30, 50]] = 1
    assert find_episodes(counts, hot_quantile=0.5, merge_gap=5) == [(5, 9), (40, 41)]
    assert find_episodes(np.zeros(10)) == []


def test_no_effect_is_not_replicated():
    assert analyse(_rows("none"), P)["verdict"]["level"] == "not replicated"


@pytest.mark.parametrize("seed", [0, 3])
def test_a_real_stock_level_effect_replicates_in_both_periods(seed):
    v = analyse(_rows("everywhere", seed=seed), P)["verdict"]
    assert v["level"].startswith("replicated") and v["stock_level"]


@pytest.mark.parametrize("seed", [0, 1])
def test_a_gain_from_one_episode_of_the_test_period_is_not_replicated(seed):
    assert analyse(_rows("episode", seed=seed), P)["verdict"]["level"] == "not replicated"


def test_a_market_rebound_replicates_but_is_not_stock_level():
    v = analyse(_rows("market", seed=0), P)["verdict"]
    assert v["level"].startswith("replicated") and not v["stock_level"]


def test_the_report_carries_both_periods_and_every_check():
    res = analyse(_rows("everywhere"), P)
    for k in ("pre", "disc"):
        r = res[k]
        assert {"gross", "net_flat", "net_stress", "neutral_gross", "neutral_net_stress", "top_days",
                "without_excluded", "market_stress", "by_year", "other_shares"} <= set(r)
        assert r["net_flat"]["mean"] == pytest.approx(r["gross"]["mean"] - 2.0)
        assert r["net_stress"]["mean"] <= r["net_flat"]["mean"] + 1e-9
    assert res["periods"]["pre"][1] < "2022-10-27" <= res["periods"]["disc"][0]
