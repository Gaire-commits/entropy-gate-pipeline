"""The volatility panel on synthetic stocks with every case planted: a clean one, a placeholder bar, a split seen only in
the vendor's adjustment, a spin-off seen only in a record, an identity break, a market-wide truncated day, a range the
5-minute cache disagrees with, a stock the vendor has nothing for, and one that lists late. Causality is tested by
changing the future and checking that nothing dated earlier moves."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import nyse, volpanel
from src.volpanel import PARKINSON, Rules

DAYS = nyse.sessions("2024-10-01", "2024-12-31")
SYMBOLS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
SMALL = Rules(weekly=5, weekly_min=4, monthly=10, monthly_min=8, coverage=0.9, market_gap_min_symbols=5)
TRUNC_DAY, SPLIT_DAY, SPIN_DAY, BREAK_DAY, HOLE_DAY, CONFLICT_DAY, LATE_START = 50, 30, 40, 25, 20, 35, 10


def daily(symbol, rng):
    n = len(DAYS)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.002, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.002, 0.01, n))
    return pd.DataFrame({"symbol": symbol, "date": DAYS, "open": open_, "high": high, "low": low, "close": close,
                         "volume": 1e6, "trade_count": 20000.0})


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(1)
    frames = {s: daily(s, rng) for s in SYMBOLS if s != "S7"}                 # S7: the vendor returns nothing
    frames["S8"] = frames["S8"].iloc[LATE_START:].reset_index(drop=True)      # lists late
    adj = pd.concat(frames.values(), ignore_index=True)
    adj.loc[(adj["symbol"] == "S1") & (adj["date"] == DAYS[HOLE_DAY]), ["volume", "trade_count"]] = 0    # a placeholder bar
    adj.loc[adj["date"] == DAYS[TRUNC_DAY], "trade_count"] = 2.0              # the vendor's day holds two prints
    adj.loc[(adj["symbol"] == "S6") & (adj["date"] == DAYS[CONFLICT_DAY]), "low"] *= 0.7               # a bad print
    raw, split = adj.copy(), adj.copy()
    before = (split["symbol"] == "S2") & (split["date"] < DAYS[SPLIT_DAY])
    split.loc[before, ["open", "high", "low", "close"]] /= 2                  # a 2-for-1 the vendor adjusts for
    adj = split.copy()
    sessions = adj[["symbol", "date", "high", "low"]].copy()
    sessions["bars"], sessions["slots"], sessions["dv_trailing63"] = 78.0, 78, 5e7
    sessions.loc[(sessions["symbol"] == "S6") & (sessions["date"] == DAYS[CONFLICT_DAY]), "low"] /= 0.7    # the cache is right
    part = (sessions["symbol"] == "S0") & (sessions["date"] == DAYS[45])                                    # a partial 5-minute day
    sessions.loc[part, "bars"] = 30.0
    sessions.loc[part, "high"] = sessions.loc[part, "low"] + 0.5 * (sessions.loc[part, "high"] - sessions.loc[part, "low"])
    actions = pd.DataFrame([{"kind": "spin_offs", "source_symbol": "S3", "new_symbol": "S3N", "when": DAYS[SPIN_DAY],
                             "symbol": None, "old_symbol": None},
                            {"kind": "cash_dividends", "symbol": "S0", "when": DAYS[55], "source_symbol": None,
                             "new_symbol": None, "old_symbol": None}])
    registry = pd.DataFrame([{"symbol": "S4", "date": DAYS[BREAK_DAY], "effect": "segment_break", "kind": "x"}])
    return raw, split, adj, sessions, actions, registry


def build(inputs, rules=SMALL, symbols=SYMBOLS, adj_change=None):
    raw, split, adj, sessions, actions, registry = inputs
    if adj_change is not None:
        adj = adj_change(adj.copy())
    v = volpanel.vendor_frame(raw, split, adj)
    return volpanel.build_panel(sessions, v, actions, registry, symbols, rules, start=DAYS[0], end=DAYS[-1])


@pytest.fixture(scope="module")
def panel(inputs):
    return build(inputs)


def row(p, symbol, day):
    return p[(p["symbol"] == symbol) & (p["date"] == DAYS[day])].iloc[0]


def test_the_target_is_overnight_plus_parkinson_from_the_previous_session(panel, inputs):
    adj = inputs[2]
    a, b = (adj[(adj["symbol"] == "S0") & (adj["date"] == DAYS[d])].iloc[0] for d in (60, 61))
    r = row(panel, "S0", 61)
    assert r["reason"] == ""
    assert r["r_on"] == pytest.approx(np.log(b["open"] / a["close"]))
    assert r["rho"] == pytest.approx(PARKINSON * np.log(b["high"] / b["low"]) ** 2)
    assert r["V"] == pytest.approx(r["r_on"] ** 2 + r["rho"])
    assert r["ret"] == pytest.approx(np.log(b["close"] / a["close"]))


def test_a_placeholder_bar_is_missing_with_its_reason_and_so_is_the_next_overnight(panel):
    assert row(panel, "S1", HOLE_DAY)["reason"] == "no trades (placeholder bar)"
    nxt = row(panel, "S1", HOLE_DAY + 1)
    assert nxt["reason"] == "previous session has no valid bar" and np.isnan(nxt["V"])
    assert not np.isnan(nxt["rho"])                                             # the intraday part is still valid
    assert row(panel, "S1", HOLE_DAY + 2)["reason"] == ""


def test_the_timeline_is_never_compressed(panel):
    assert (panel.groupby("symbol").size() == len(DAYS)).all()
    assert panel[["symbol", "date"]].equals(panel.sort_values(["symbol", "date"])[["symbol", "date"]])
    assert panel.loc[panel["symbol"] == "S0", "half_day"].sum() == nyse.is_half_day(DAYS).sum()


def test_a_split_seen_only_in_the_vendors_adjustment_is_an_event(panel):
    r = row(panel, "S2", SPLIT_DAY)
    assert r["reason"] == "corporate action or adjustment step" and "vendor split-type" in r["event_source"]
    assert row(panel, "S2", SPLIT_DAY + 1)["reason"] == ""
    assert row(panel, "S2", SPLIT_DAY - 1)["reason"] == ""


def test_a_spin_off_seen_only_in_a_record_is_an_event(panel):
    r = row(panel, "S3", SPIN_DAY)
    assert r["reason"] == "corporate action or adjustment step" and r["event_source"] == "record: spin off"
    assert row(panel, "S0", 55)["reason"] == ""                                 # a cash dividend is not an event


def test_an_identity_break_starts_a_new_segment(panel):
    r = row(panel, "S4", BREAK_DAY)
    assert r["reason"] == "first session of a segment (identity change)" and r["segment"] == 1
    assert row(panel, "S4", BREAK_DAY - 1)["segment"] == 0 and np.isnan(r["ret"])
    q = volpanel.add_eligibility(panel, SMALL, warmup=10)
    after = q[(q["symbol"] == "S4") & (q["date"] > DAYS[BREAK_DAY])].reset_index(drop=True)
    assert not after.loc[:4, "warm_ok"].any()                                   # nothing carries over from the old segment
    assert after.loc[12, "warm_ok"]


def test_a_truncated_market_wide_day_is_found_and_the_day_after_is_not_trusted(panel):
    r = row(panel, "S0", TRUNC_DAY)
    assert r["trunc"] and r["reason"] == "market-wide truncated day" and np.isnan(r["V"])
    nxt = row(panel, "S0", TRUNC_DAY + 1)
    assert nxt["reason"] == "previous session is a truncated day"
    assert panel.groupby("date")["trunc"].any().sum() == 1


def test_the_range_must_agree_with_a_complete_five_minute_day_but_a_partial_one_only_binds_one_way(panel):
    r = row(panel, "S6", CONFLICT_DAY)
    assert r["reason"] == "range disagrees with the 5-minute cache" and r["range_d"] > np.log(1.25)
    partial = row(panel, "S0", 45)                                              # the cache saw less: the vendor is wider
    assert partial["reason"] == "" and partial["range_d"] > 0


def test_symbols_without_bars_or_before_listing_say_so(panel):
    assert (panel.loc[panel["symbol"] == "S7", "reason"] == "symbol has no vendor bars").all()
    assert row(panel, "S8", 3)["reason"] == "outside the security's trading span"
    assert row(panel, "S8", LATE_START)["reason"] == "previous session has no valid bar"     # its first session
    assert row(panel, "S8", LATE_START + 1)["reason"] == ""


def test_eligibility_is_causal_changing_the_future_moves_nothing_earlier(inputs):
    base = volpanel.add_eligibility(build(inputs), SMALL, warmup=10)
    cut = DAYS[52]

    def perturb(df):
        df = df.copy()
        late = df["date"] >= cut
        df.loc[late, ["open", "high", "low", "close"]] *= 3
        df.loc[late, ["volume", "trade_count"]] = 7
        return df

    other = volpanel.add_eligibility(build(inputs, adj_change=perturb), SMALL, warmup=10)
    early = base["date"] < cut
    for col in ("V", "ret", "lag_ok", "week_ok", "month_ok", "warm_ok", "liq_ok", "trunc"):
        a, b = base.loc[early, col], other.loc[early, col]
        assert ((a == b) | (a.isna() & b.isna())).all(), col


def test_the_target_is_the_next_session_and_eligibility_needs_every_requirement(inputs):
    q = volpanel.add_eligibility(build(inputs), SMALL, warmup=10)
    r = q[(q["symbol"] == "S0") & (q["date"] == DAYS[61])].iloc[0]
    nxt = q[(q["symbol"] == "S0") & (q["date"] == DAYS[62])].iloc[0]
    assert r["target_date"] == DAYS[62] and r["V_next"] == pytest.approx(nxt["V"]) and r["eligible"]
    low_price = volpanel.add_eligibility(build(inputs), Rules(**{**SMALL.__dict__, "min_raw_close": 1e6}), warmup=10)
    assert not low_price["eligible"].any()
    assert not q.loc[q["symbol"] == "S7", "eligible"].any()


def test_sample_sizes_are_a_waterfall_by_window_of_the_target_date(inputs):
    windows = {"a": (str(DAYS[0].date()), str(DAYS[39].date())), "b": (str(DAYS[41].date()), str(DAYS[-1].date()))}
    p = build(inputs)
    table = volpanel.sample_sizes(p, SMALL, warmups=(10,), windows=windows)
    assert list(table["window"]) == ["a", "b"] and (table["sessions"] > 0).all()
    steps = ["listed stock-days", "target has a value", "+ price and liquidity", "+ daily lag", "+ weekly window",
             "+ monthly window", "+ warm-up"]
    for _, r in table.iterrows():
        values = [r[s] for s in steps]
        assert values == sorted(values, reverse=True) and r["+ warm-up"] == r["eligible"] > 0
    reasons = volpanel.reason_counts(p, windows)
    assert "(valid)" in reasons.index and "no trades (placeholder bar)" in reasons.index


def test_the_registry_is_well_formed():
    registry = volpanel.load_registry(Path(__file__).resolve().parents[1] / "registry" / "vol_registry.csv")
    assert set(registry["effect"]) <= {"segment_break", "overnight_unreliable"}
    assert registry["evidence"].str.len().gt(40).all() and registry["status"].notna().all()
    assert not registry.duplicated(["symbol", "date", "effect"]).any()


def test_the_sample_script_runs_on_the_audit_fixture(tmp_path):
    """End to end on the synthetic audit: the panel, the tables and the report are written, nothing needs the real data."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import vol_sample
    from src import nyse as cal
    days = cal.sessions("2024-10-01", "2024-12-31")
    rng = np.random.default_rng(3)
    symbols = [f"T{i}" for i in range(6)]
    frames = {s: daily(s, rng) for s in symbols}
    adj = pd.concat(frames.values(), ignore_index=True)
    ref, audit_dir = tmp_path / "ref", tmp_path / "audit"
    (ref / "daily").mkdir(parents=True)
    audit_dir.mkdir()
    for tag in ("raw", "split", "all"):
        adj.to_parquet(ref / "daily" / f"daily_sip_{tag}.parquet")
    pd.DataFrame(columns=["kind", "id", "symbol", "ex_date"]).to_parquet(ref / "corporate_actions.parquet")
    sessions = adj[["symbol", "date", "high", "low"]].assign(bars=78.0, slots=78, dv_trailing63=5e7, nyse_session=True)
    sessions.to_parquet(audit_dir / "sessions.parquet")
    pd.DataFrame({"symbol": ["T0"], "date": [days[40]], "gap_sessions": 1, "kind": "overnight", "move": 0.3,
                  "class": "market move (no corporate action found)"}).to_csv(audit_dir / "large_moves.csv", index=False)
    pd.DataFrame({"symbol": symbols, "name": "x", "sector": "x"}).to_csv(tmp_path / "u.csv", index=False)
    vol_sample.WINDOWS = {"a": (str(days[0].date()), str(days[39].date())), "b": (str(days[41].date()), str(days[-1].date()))}
    old, sys.argv = sys.argv, ["vol_sample.py", "--audit", str(audit_dir), "--reference", str(ref), "--universe",
                                str(tmp_path / "u.csv"), "--out", str(tmp_path / "out"), "--end", str(days[-1].date())]
    try:
        assert vol_sample.main() == 0
    finally:
        sys.argv = old
        vol_sample.WINDOWS = volpanel.WINDOWS
    out = tmp_path / "out"
    for name in ("report.md", "sample_sizes.csv", "reasons.csv", "events.csv", "panel.parquet", "large_moves_check.csv"):
        assert (out / name).exists(), name
    assert pd.read_csv(out / "sample_sizes.csv")["eligible"].sum() > 0
