"""Cross-sectional tests: skill that directional scoring cannot see, no skill invented, and honest breadth."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crosssection import (
    breakeven_ic, cross_sectional_ic, cross_sectional_sigma, directional_book, effective_bets, ensemble_scores,
    quantile_book, score_book, wide_moments, with_sector,
)

SLOTS = (0, 12, 24)


def _panel(ic=0.06, days=150, n_symbols=60, n_sectors=6, long_bias=2.0, model_knows_sector=0.0,
           market_sd=40.0, sector_sd=25.0, idio_sd=80.0, noise=0.8, seed=0):
    """Returns in bps = a market move shared by everyone + a sector move + a stock-specific part correlated
    `ic` with a hidden signal z. The market move is mostly shared across a day's moments (64% of its
    variance), because holds starting at 0, 12 and 24 bars overlap. The model's score is z plus noise, a
    long bias (so it is almost always long, as ours are) and optionally the sector move (a model that
    only knows the sector)."""
    rng = np.random.default_rng(seed)
    d, b, s = (a.ravel() for a in np.meshgrid(np.arange(days), SLOTS, np.arange(n_symbols), indexing="ij"))
    moment = d * len(SLOTS) + np.searchsorted(SLOTS, b)
    sector = s % n_sectors
    market = market_sd * (0.8 * rng.normal(size=days)[d] + 0.6 * rng.normal(size=days * len(SLOTS))[moment])
    sector_move = sector_sd * rng.normal(size=(days * len(SLOTS), n_sectors))[moment, sector]
    z = rng.normal(size=len(d))
    idio = idio_sd * (ic * z + np.sqrt(1 - ic**2) * rng.normal(size=len(d)))
    score = (z + noise * rng.normal(size=len(d))) * 0.8 + long_bias + model_knows_sector * sector_move / sector_sd
    return pd.DataFrame({
        "fold": d // 30, "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(d, unit="D"),
        "symbol": [f"S{i:03d}" for i in s], "sector": [f"G{g}" for g in sector], "bar_index": b,
        "ret": (market + sector_move + idio) / 1e4, "prob": 1 / (1 + np.exp(-score / 2)),
    })


def test_ic_finds_stock_level_skill_that_directional_scoring_cannot_see():
    p = _panel(ic=0.08)
    directional = score_book(directional_book(p), cost_bps=0.0, n_boot=300)
    ic = cross_sectional_ic(p, n_boot=300)
    assert directional["net_exposure"] > 0.9                                   # one long bet on the market
    assert directional["gross_lo"] < 0 < directional["gross_hi"]               # skill invisible under the market's noise
    assert ic["ic_lo"] > 0.02 and ic["t"] > 4


def test_ic_invents_nothing_when_there_is_no_signal():
    flagged, worst = 0, 0.0
    for seed in range(10):
        ic = cross_sectional_ic(_panel(ic=0.0, days=100, seed=seed), n_boot=200)
        flagged += ic["ic_lo"] > 0 or ic["ic_hi"] < 0
        worst = max(worst, abs(ic["ic"]))
    assert flagged <= 2 and worst < 0.03


def test_sector_neutral_ic_removes_a_signal_that_is_only_knowing_the_sector():
    p = _panel(ic=0.0, model_knows_sector=1.5)
    raw, neutral = cross_sectional_ic(p, n_boot=300), cross_sectional_ic(p, sector_neutral=True, n_boot=300)
    assert raw["ic_lo"] > 0.05
    assert abs(neutral["ic"]) < 0.02 and neutral["ic_lo"] < 0 < neutral["ic_hi"]


def test_sector_neutral_ic_keeps_real_stock_level_skill():
    neutral = cross_sectional_ic(_panel(ic=0.06, model_knows_sector=1.5), sector_neutral=True, n_boot=300)
    assert neutral["ic_lo"] > 0.015 and neutral["t"] > 4


def test_quantile_book_is_market_neutral_and_earns_the_planted_edge_only_when_it_exists():
    skilled = score_book(quantile_book(_panel(ic=0.08)), cost_bps=2.0, n_boot=300)
    assert abs(skilled["net_exposure"]) < 0.01
    assert skilled["gross_lo"] > 3 and skilled["long_leg"] > 0 > skilled["short_leg"] * -1 - 1e-9  # both legs contribute
    assert skilled["net"] == pytest.approx(skilled["gross"] - 2.0)
    flat = score_book(quantile_book(_panel(ic=0.0, seed=3)), cost_bps=2.0, n_boot=300)
    assert flat["gross_lo"] < 0 < flat["gross_hi"]


def test_sector_balanced_book_holds_equal_longs_and_shorts_in_every_sector():
    p = _panel(ic=0.05, days=20)
    book = quantile_book(p, sector_neutral=True)
    counts = book.groupby(["date", "bar_index", "sector", "direction"]).size().unstack("direction", fill_value=0)
    assert (counts[1.0] == counts[-1.0]).all() and (counts[1.0] > 0).all()
    ordinary = quantile_book(p)
    by_sector = ordinary.groupby(["date", "bar_index", "sector"])["direction"].sum()
    assert by_sector.abs().max() > 0                                       # the plain book is not sector balanced


def test_ties_do_not_let_symbol_order_pick_the_book():
    """Half the symbols tie at the top and half at the bottom. Broken by position, the first few symbols
    would win every moment (120 picks each); broken at random each of the 30 is picked about 24 times."""
    p = _panel(ic=0.0, days=40)
    p["prob"] = 0.5 + 0.1 * (p.groupby(["date", "bar_index"]).cumcount() % 2)     # two values only
    longs = quantile_book(p, q=0.1).query("direction > 0").groupby("symbol").size()
    assert len(longs) == 30 and longs.max() < 50 and longs.min() > 5


def test_effective_bets_tell_one_market_bet_from_many_independent_ones():
    p = _panel(ic=0.0, market_sd=60.0)
    long_only = effective_bets(directional_book(p))
    neutral = effective_bets(quantile_book(p))
    assert long_only["positions_per_day"] == 180 and long_only["bets_per_day"] < 5      # 180 trades, a few bets
    assert neutral["bets_per_day"] > 15
    assert neutral["bets_per_day"] <= neutral["positions_per_day"] + 1e-9


def test_effective_bets_recovers_a_known_common_component():
    """n positions a day with pairwise correlation rho amount to n / (1 + (n - 1) rho) bets."""
    rng = np.random.default_rng(0)
    n, days = 20, 4000
    for rho in (0.0, 0.1, 0.3, 0.7):
        common = np.repeat(rng.normal(size=days), n)
        frame = pd.DataFrame({"date": np.repeat(np.arange(days), n),
                              "gross_bps": np.sqrt(rho) * common + np.sqrt(1 - rho) * rng.normal(size=days * n)})
        got = effective_bets(frame)
        assert got["rho"] == pytest.approx(rho, abs=0.03)
        assert got["bets_per_day"] == pytest.approx(n / (1 + (n - 1) * rho), rel=0.2)


def test_breakeven_ic_matches_the_closed_form():
    assert breakeven_ic(2.0, 80.0, q=0.1) == pytest.approx(2.0 / (80.0 * 1.7550), rel=1e-3)
    assert breakeven_ic(4.0, 80.0) == pytest.approx(2 * breakeven_ic(2.0, 80.0))
    assert breakeven_ic(2.0, 80.0, q=0.02) < breakeven_ic(2.0, 80.0, q=0.1)    # thinner tails, bigger picks


def test_cross_sectional_sigma_is_the_spread_after_the_market_and_optionally_the_sector():
    p = _panel(ic=0.0, days=80)
    assert cross_sectional_sigma(p) == pytest.approx(np.hypot(80, 25), rel=0.06)
    assert cross_sectional_sigma(p, sector_neutral=True) == pytest.approx(80, rel=0.06)


def test_ensemble_of_independent_noisy_models_beats_each_and_reports_how_alike_they_are():
    base = _panel(ic=0.12, days=200, noise=0.0, long_bias=0.0)
    frames = {}
    for i, noise in enumerate((2.0, 2.0, 2.0)):
        rng = np.random.default_rng(100 + i)
        f = base.copy()
        logit = np.log(base["prob"] / (1 - base["prob"])) * 2
        f["prob"] = 1 / (1 + np.exp(-(logit + noise * rng.normal(size=len(f))) / 2))
        frames[f"m{i}"] = f
    merged, summary = ensemble_scores(frames)
    singles = [cross_sectional_ic(merged.assign(prob=merged[f"p_{m}"]), n_boot=50)["ic"] for m in frames]
    ensemble = cross_sectional_ic(merged.assign(prob=merged["ens"]), n_boot=50)["ic"]
    assert ensemble > max(singles) + 0.01
    assert 0.1 < summary["mean_corr"] < 0.8 and 1.0 < summary["independent_models"] < 3.0


def test_thin_moments_are_ignored_and_reported_as_no_data():
    p = _panel(n_symbols=12, days=30)
    assert cross_sectional_ic(p, min_peers=20) == {"days": 0}
    assert quantile_book(p, min_peers=20).empty and len(wide_moments(p, 12)) == len(p)


def test_sectors_come_from_a_map_and_unmapped_symbols_stay_visible():
    p = _panel(days=5).drop(columns="sector")
    mapped = with_sector(p, {"S000": "Energy", "S001": "Tech"})
    assert set(mapped["sector"]) == {"Energy", "Tech", "unknown"}
    assert set(with_sector(p, None)["sector"]) == {"all"}


def test_fixed_count_book_holds_exactly_n_names_a_side_and_skips_thin_moments():
    p = _panel(ic=0.05, days=20)                              # 60 symbols a moment
    book = quantile_book(p, n_per_side=12)
    sides = book.groupby(["date", "bar_index", "direction"]).size().unstack("direction")
    assert (sides[1.0] == 12).all() and (sides[-1.0] == 12).all()
    assert quantile_book(p, n_per_side=31, min_peers=20).empty    # 60 names cannot fill 31 a side


def test_fixed_count_sector_book_is_balanced_in_every_sector_and_close_to_n_overall():
    p = _panel(ic=0.05, days=20)                              # 6 sectors of 10
    book = quantile_book(p, n_per_side=12, sector_neutral=True)
    counts = book.groupby(["date", "bar_index", "sector", "direction"]).size().unstack("direction", fill_value=0)
    assert (counts[1.0] == counts[-1.0]).all() and (counts[1.0] == 2).all()   # 12 x 10/60 = 2 per sector a side
    assert (book.groupby(["date", "bar_index", "direction"]).size() == 12).all()
