"""The selective strategy: cutoffs from the past, honest frequency accounting, and a placebo that
tells skill from volatility."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.selective import selective_table, top_share_mask, trailing_volatility
from src.synthetic import ar_returns, bars_from_returns

H = 30


def _pred(kind, n_folds=8, days_per_fold=40, per_day=200, seed=0):
    """'skill': confident calls are right more often, move sizes unrelated to confidence.
    'volatility': right 55% of the time whatever the confidence, but the model is most confident
    when the market is most volatile, so its confident trades are its biggest moves."""
    rng = np.random.default_rng(seed)
    n = n_folds * days_per_fold * per_day
    vol = rng.lognormal(0, 0.6, n) * 1e-3
    if kind == "skill":
        conf = rng.uniform(0, 0.4, n)
        right = rng.random(n) < 0.5 + 0.8 * conf
    else:
        rank = pd.Series(vol).rank(pct=True).to_numpy()
        conf = np.clip(0.35 * rank + rng.normal(0, 0.03, n), 0.001, 0.49)
        right = rng.random(n) < 0.55
    side = rng.choice([-1.0, 1.0], n)
    size = np.abs(rng.normal(0, vol * np.sqrt(H)))
    ret = np.where(right, side, -side) * size
    day = np.arange(n) // per_day
    return pd.DataFrame({
        "fold": day // days_per_fold,
        "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(day, unit="D"),
        "symbol": "S", "bar_index": 0, "prob": 0.5 + side * conf, "ret": ret,
        "y": (ret > 0).astype(int), "vol": vol,
    })


def _row(table, selector, share):
    return table[(table["selector"] == selector) & (table["share"] == share)].iloc[0]


def test_skill_shows_in_hit_rate_and_risk_scaled_return_and_beats_the_placebo_there():
    """In bps the placebo can win even here: a model right 66% of the time everywhere makes the
    most bps where moves are biggest. Skill shows per unit of each trade's own risk."""
    table = selective_table(_pred("skill"), cost=2.0, horizon=H, n_boot=200)
    everything, top = _row(table, "confidence", 1.0), _row(table, "confidence", 0.1)
    placebo = _row(table, "volatility (placebo)", 0.1)
    assert top["hit_rate"] > everything["hit_rate"] + 0.1
    assert top["risk_scaled"] > everything["risk_scaled"] + 10
    assert top["risk_scaled"] > placebo["risk_scaled"] + 10


def test_volatility_alone_raises_bps_per_trade_but_not_skill_and_the_placebo_matches_it():
    """The pattern the S&P 500 lead could be: more bps per trade as the filter tightens, from
    bigger moves, with the same hit rate."""
    table = selective_table(_pred("volatility"), cost=2.0, horizon=H, n_boot=200)
    everything, top = _row(table, "confidence", 1.0), _row(table, "confidence", 0.1)
    placebo = _row(table, "volatility (placebo)", 0.1)
    assert top["gross"] > 1.5 * everything["gross"]               # looks like an edge in bps
    assert abs(top["hit_rate"] - everything["hit_rate"]) < 0.03   # but no better at calling direction
    assert abs(top["risk_scaled"] - everything["risk_scaled"]) < 4
    assert top["vol_ratio"] > 1.5
    assert placebo["gross"] == pytest.approx(top["gross"], rel=0.35)


def test_frequency_and_daily_earnings_are_counted_over_every_eligible_day():
    p = _pred("skill")
    table = selective_table(p, cost=2.0, horizon=H, n_boot=50)
    row = _row(table, "confidence", 0.1)
    eligible_days = p[p["fold"] >= 1]["date"].nunique()      # fold 0 has no history to set a cutoff
    assert row["trades_per_day"] == pytest.approx(row["trades"] / eligible_days)
    assert row["net_per_day"] == pytest.approx(row["net"] * row["trades_per_day"], rel=1e-6)
    assert 0.05 < row["kept"] < 0.15 and row["folds"] == 7
    every = _row(table, "confidence", 1.0)
    assert every["kept"] == 1.0 and every["folds_positive"] <= every["folds"] == 7   # no scored quarter lost


def test_cutoffs_come_from_earlier_folds_only():
    rng = np.random.default_rng(1)
    score, folds = rng.random(2000), np.repeat(np.arange(5), 400)
    changed = score.copy()
    changed[folds >= 4] *= 10                                       # rewrite the last fold
    a, b = top_share_mask(score, folds, 0.1), top_share_mask(changed, folds, 0.1)
    np.testing.assert_array_equal(a[folds <= 3], b[folds <= 3])
    assert not a[folds == 0].any()                                  # nothing earlier to set a cutoff from
    assert top_share_mask(score, folds, 1.0)[folds >= 1].all()
    gappy = np.where(score > 0.5, np.nan, score)
    assert not top_share_mask(gappy, folds, 0.5)[np.isnan(gappy)].any()   # a missing score is never kept


def test_trailing_volatility_uses_the_signal_bar_and_nothing_after_it():
    r = ar_returns(10, 0.0, seed=2)
    df = bars_from_returns("S", r)
    ts = df.index.get_level_values("timestamp")
    keys = pd.DataFrame({"symbol": "S", "date": ts[[500, 700]].normalize().tz_localize(None),
                         "bar_index": df["bar_index"].to_numpy()[[500, 700]]})
    vol = trailing_volatility({"S": df}, keys)
    logret = np.log(df["close"] / df["close"].shift(1)).to_numpy()
    assert vol.iloc[0] == pytest.approx(np.nanstd(logret[500 - 77 : 501], ddof=1))

    changed = r.copy()
    changed[600:] *= 5
    later = trailing_volatility({"S": bars_from_returns("S", changed)}, keys)
    assert later.iloc[0] == pytest.approx(vol.iloc[0]) and later.iloc[1] != pytest.approx(vol.iloc[1])
