"""The extended audit on synthetic bars with every case planted: half-days with prints after 13:00, an
impossible bar, tickers reused (shown by an exit record, by a when-issued rename, and by Alpaca's issuer
mapping alone), the same issuer not trading for a while (vendor placeholder bars), a special dividend recorded
under an older ticker, a distribution with no record that the dividend adjustment shows, a spin-off, a genuine
crash next to an ordinary dividend, a symbol-mapping splice, a symbol the mapping returns nothing for, and
vendor daily bars that count extended-hours volume."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import audit, nyse

DAYS = nyse.sessions("2024-10-01", "2024-12-31")
SYMBOLS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
EVENTS = {"S3": (30, np.log(0.65)), "S2": (35, np.log(0.5)), "S4": (40, -0.30), "S9": (25, np.log(0.5))}
GAP = range(20, 45)                                   # these symbols have no bars here
GAPPED = {"S1": 3.0, "S6": 1.0, "S7": 3.0, "S8": 3.0}   # price factor when trading resumes (1.0: same issuer)


def build(symbol, rng):
    """Returns (cache bars as the cache stores them, every bar of each day incl. extended hours)."""
    rows, price = [], 100.0
    for i, day in enumerate(DAYS):
        if symbol in GAPPED and i in GAP:
            continue
        if symbol in GAPPED and i == GAP.stop:
            price *= GAPPED[symbol]
        if symbol in EVENTS and EVENTS[symbol][0] == i:
            price *= np.exp(EVENTS[symbol][1])
        close_h = 13 if nyse.is_half_day(day) else 16
        times = [day + pd.Timedelta(hours=8) + k * audit.BAR for k in range(18)]          # pre-market 08:00-09:25
        times += list(pd.date_range(day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(hours=close_h) - audit.BAR,
                                    freq="5min"))
        times += [day + pd.Timedelta(hours=close_h) + k * audit.BAR for k in range(6)]    # after the close
        for t in times:
            o = price
            price *= np.exp(rng.normal(0, 0.001))
            after = (t - day) >= pd.Timedelta(hours=close_h)
            pre = (t - day) < pd.Timedelta(hours=9, minutes=30)
            h = max(o, price) * (1.05 if after else 1.0005)                 # thin after-close prints: wide range
            lo = min(o, price) * 0.9995
            rows.append((t, o, h, lo, price, 50.0 if (after or pre) else 1000.0))
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"])
    df["trade_count"], df["vwap"] = 10.0, (df["high"] + df["low"] + df["close"]) / 3
    tod = df["t"] - df["t"].dt.normalize()
    cache = df[(tod >= pd.Timedelta(hours=9, minutes=30)) & (tod < pd.Timedelta(hours=16))].copy()
    if symbol == "S0":                                                       # one impossible bar
        cache.loc[cache.index[10 * 78 + 5], "low"] = 0.0
    idx = pd.MultiIndex.from_arrays([np.full(len(cache), symbol), pd.DatetimeIndex(cache["t"]).tz_localize(audit.EASTERN)],
                                    names=["symbol", "timestamp"])
    return cache.drop(columns="t").set_index(idx), df.assign(symbol=symbol)


def vendor_daily(full: pd.DataFrame) -> pd.DataFrame:
    """Alpaca's rules: open/high/low/close from the regular session only, volume from the whole day."""
    full = full.copy()
    day = full["t"].dt.normalize()
    tod = full["t"] - day
    close = np.where(day.isin(nyse.HALF_DAYS), pd.Timedelta(hours=13).to_timedelta64(), pd.Timedelta(hours=16).to_timedelta64())
    rth = (tod >= pd.Timedelta(hours=9, minutes=30)) & (tod.to_numpy() < close)
    g, gv = full[rth].groupby(["symbol", day[rth]]), full.groupby(["symbol", day])
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": gv["volume"].sum(), "trade_count": 1.0, "vwap": 1.0})
    out.index.names = ["symbol", "date"]
    return out.reset_index()


def with_placeholders(df, symbol):
    """Alpaca-style placeholder bars for the sessions a symbol did not trade: zero volume, last close."""
    last = df[(df["symbol"] == symbol) & (df["date"] < DAYS[GAP.start])]["close"].iloc[-1]
    fill = pd.DataFrame({"symbol": symbol, "date": DAYS[list(GAP)], "open": last, "high": last, "low": last,
                         "close": last, "volume": 0.0, "trade_count": 0.0, "vwap": 0.0})
    return pd.concat([df, fill], ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)


def scale_before(df, symbol, day, factor):
    df = df.copy()
    sel = (df["symbol"] == symbol) & (df["date"] < day)
    df.loc[sel, ["open", "high", "low", "close"]] *= factor
    return df


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("audit")
    rng = np.random.default_rng(0)
    fulls = []
    for sym in SYMBOLS:
        cache, full = build(sym, rng)
        fulls.append(full)
        for feed in ("sip", "iex"):
            path = tmp / feed / "5Min" / f"{sym}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            c = cache.copy()
            if feed == "iex" and sym == "S1":
                c.loc[c.index[3 * 78 + 2], "low"] = 0.0                     # an IEX zero price
            c.to_parquet(path)
            path.with_suffix(".meta.json").write_text(json.dumps({"start": "2024-10-01", "end": "2024-12-31",
                                                                   "feed": feed, "adjustment": "split"}))
    full = pd.concat(fulls, ignore_index=True)
    split = with_placeholders(vendor_daily(full), "S6")
    ex2, spin4, ex9 = DAYS[EVENTS["S2"][0]], DAYS[EVENTS["S4"][0]], DAYS[EVENTS["S9"][0]]
    prev2 = split[(split["symbol"] == "S2") & (split["date"] < ex2)]["close"].iloc[-1]
    divs = scale_before(scale_before(split, "S2", ex2, 0.5), "S9", ex9, 0.5)
    every = scale_before(divs, "S4", spin4, np.exp(EVENTS["S4"][1]))
    nomap = split[~((split["symbol"] == "S5") & (split["date"] < DAYS[10]))]
    # Alpaca's issuer mapping: S8's issuer starts after its gap; S10 gets nothing at all
    mapped = lambda df: df[~((df["symbol"] == "S8") & (df["date"] < DAYS[GAP.start])) & (df["symbol"] != "S10")]  # noqa: E731
    ref = tmp / "ref"
    (ref / "daily").mkdir(parents=True)
    for tag, df in {"raw": mapped(split), "split": mapped(split), "split+dividend": mapped(divs), "all": mapped(every),
                    "split_nomap": nomap}.items():
        df.to_parquet(ref / "daily" / f"daily_sip_{tag}.parquet")
    ca = pd.DataFrame([
        {"kind": "name_changes", "id": "1", "old_symbol": "S1", "new_symbol": "S1X", "process_date": str(DAYS[GAP.start].date())},
        # S2 traded as OLD2 at the time: its dividend is recorded under the old ticker
        {"kind": "cash_dividends", "id": "2", "symbol": "OLD2", "rate": prev2 * 0.5, "special": True, "ex_date": str(ex2.date())},
        {"kind": "name_changes", "id": "6", "old_symbol": "OLD2", "new_symbol": "S2", "process_date": str(DAYS[50].date())},
        {"kind": "spin_offs", "id": "3", "source_symbol": "S4", "new_symbol": "S4N", "ex_date": str(spin4.date())},
        {"kind": "cash_dividends", "id": "4", "symbol": "S3", "rate": 0.1, "special": False,
         "ex_date": str(DAYS[EVENTS["S3"][0] - 3].date())},
        {"kind": "name_changes", "id": "5", "old_symbol": "OLD5", "new_symbol": "S5", "process_date": str(DAYS[10].date())},
        {"kind": "name_changes", "id": "7", "old_symbol": "S7.WI", "new_symbol": "S7", "process_date": str(DAYS[GAP.stop].date())},
    ])
    ca.to_parquet(ref / "corporate_actions.parquet")
    sample_days = [pd.Timestamp("2024-11-29"), DAYS[5]]
    ext = full[(full["symbol"] == "S5") & full["t"].dt.normalize().isin(sample_days)].copy()
    ext["t"] = pd.DatetimeIndex(ext["t"]).tz_localize(audit.EASTERN).tz_convert("UTC")
    (ref / "sample").mkdir()
    ext[["symbol", "t", "open", "high", "low", "close", "volume", "trade_count", "vwap"]].to_parquet(ref / "sample" / "ext_5min.parquet")
    s5 = split[split["symbol"] == "S5"]
    auctions = pd.concat([s5.assign(side="open", condition="Q", price=s5["open"]),
                          s5.assign(side="close", condition="M", price=s5["close"])])
    auctions = auctions.assign(size=1, exchange="N", t=pd.Timestamp("2024-01-01", tz="UTC"))[
        ["symbol", "date", "side", "condition", "price", "size", "t", "exchange"]]
    auctions.to_parquet(ref / "sample" / "auctions.parquet")
    (ref / "manifest.json").write_text(json.dumps({"files": {}}))
    pd.DataFrame({"symbol": SYMBOLS, "name": "x", "sector": "x"}).to_csv(tmp / "u.csv", index=False)
    return tmp


@pytest.fixture(scope="module")
def outputs(made):
    import data_audit_v2 as script
    out = made / "out"
    argv = sys.argv
    sys.argv = ["data_audit_v2.py", "--sip", str(made / "sip"), "--iex", str(made / "iex"), "--universe",
                str(made / "u.csv"), "--reference", str(made / "ref"), "--out", str(out), "--start", "2024-10-01"]
    try:
        assert script.main() == 0
    finally:
        sys.argv = argv
    return out


def test_every_planted_move_gets_its_class(outputs):
    moves = pd.read_csv(outputs / "large_moves.csv", parse_dates=["date"])
    cls = {(r.symbol, r.kind): r["class"] for _, r in moves.iterrows()}
    assert cls[("S2", "overnight")] == "cash distribution"                 # found under the older ticker
    assert cls[("S9", "overnight")] == "cash distribution (shown by Alpaca's dividend adjustment; no record)"
    assert cls[("S4", "overnight")] == "spin-off"
    assert cls[("S3", "overnight")] == "market move (no corporate action found)"   # kept, despite the dividend 3 days before
    for sym in ("S1", "S7", "S8"):
        assert cls[(sym, "overnight")] == "identity change (after a long gap)"
    assert "removed by split+dividend: True" in moves[moves["symbol"] == "S2"].iloc[0]["reason"]
    assert len(moves) == 7                                                   # S6 resumes at its old price: no move


def test_gaps_are_classified_on_evidence(outputs):
    gaps = pd.read_csv(outputs / "gaps.csv").set_index("symbol")
    assert (gaps.loc[["S1", "S6", "S7", "S8"], "sessions"] == len(GAP)).all()
    assert gaps.loc["S1", "class"] == audit.REUSED                           # an exit record
    assert gaps.loc["S7", "class"] == audit.REUSED                           # a when-issued rename at the gap's end
    assert gaps.loc["S8", "class"] == audit.REUSED                           # the issuer mapping starts after the gap
    assert gaps.loc["S6", "class"] == audit.SAME_ISSUER_NO_TRADING
    assert "placeholder bars during the gap: mapped 25" in gaps.loc["S6", "evidence"]
    splices = pd.read_csv(outputs / "symbol_mapping_splices.csv").set_index("symbol")
    assert splices.loc["S5", "mapped_only_days"] == 10
    ph = pd.read_csv(outputs / "vendor_placeholder_bars.csv")
    assert ph["symbol"].tolist() == ["S6"] and ph["bars"].iloc[0] == len(GAP)


def test_unresolved_cases_are_listed_apart(outputs):
    unresolved = pd.read_csv(outputs / "unresolved.csv")
    assert dict(zip(unresolved["kind"], unresolved["symbol"])) == {
        "impossible bar values (cause unknown)": "S0",
        "gap cause not recorded (identity: same issuer)": "S6",
        "Alpaca's mapping returns no bars": "S10"}
    identity = pd.read_csv(outputs / "identity_vs_mapping.csv").set_index("symbol")
    assert identity.loc["S8", "cache_only_days"] == GAP.start                 # explained by the reuse, so not unresolved


def test_impossible_values_are_located(outputs):
    bad = pd.read_csv(outputs / "invalid_bars_sip.csv")
    assert len(bad) == 1 and bad["symbol"].iloc[0] == "S0" and "zero or negative price" in bad["reason"].iloc[0]
    ratios = pd.read_csv(outputs / "first_audit_nonfinite_ratios.csv")
    reasons = dict(zip(ratios["symbol"], ratios["reason"]))
    assert "sip: zero or negative price" in reasons["S0"] and reasons["S1"] == "iex: zero or negative price"
    assert len(pd.read_csv(outputs / "nonfinite_sessions_sip.csv")) == 0   # the rebuild leaves the bad bar out


def test_half_days_end_at_13_and_the_first_audits_rule_missed_them(outputs):
    cov = pd.read_csv(outputs / "coverage_by_year.csv")
    half = cov[cov["day_type"] == "half"].iloc[0]
    in_gap = sum(1 for d in nyse.HALF_DAYS if d in DAYS and DAYS.get_loc(d) in GAP) * len(GAPPED)
    assert half["sessions"] == 2 and half["missing_close"] == 0
    assert half["first_audit_missing_close"] == 2 * len(SYMBOLS) - in_gap   # every half-day the first audit saw
    days = pd.read_csv(outputs / "half_days.csv")
    assert (days["with_bars_after_13"] == days["symbols"]).all()


def test_the_vendor_session_definition_is_found(outputs):
    d = pd.read_csv(outputs / "session_definitions.csv").set_index(["day_type", "definition"])
    assert d.loc[("half", "calendar"), ["open_match", "high_match", "low_match", "close_match"]].eq(1).all()
    assert d.loc[("half", "clock"), "high_match"] < 1                        # the after-13:00 prints widen it
    assert d.loc[("full", "whole_day"), "volume_exact_share"] == 1           # volume counts extended hours
    assert d.loc[("full", "calendar"), "volume_exact_share"] == 0
    basis = pd.read_csv(outputs / "vendor_basis.csv")
    assert (basis["basis"] == "same").all()
    overall = pd.read_csv(outputs / "vendor_overall.csv").set_index("day_type")
    assert overall.loc["half", "high_match"] == 1 and overall.loc["half", "high_match_clock_rebuild"] < 1


def test_sessions_table_covers_every_symbol_and_session(outputs):
    s = pd.read_parquet(outputs / "sessions.parquet")
    assert len(s) == len(SYMBOLS) * len(DAYS)
    assert set(s["status"]) <= {"ok", "sparse", "no_bars", "before_first_bar", "after_last_bar"}
    s1 = s[s["symbol"] == "S1"].reset_index(drop=True)
    assert (s1.loc[list(GAP), "status"] == "no_bars").all()
    assert s1.loc[GAP.stop, "gap_sessions"] == len(GAP) + 1                   # the return spans the gap, and says so
    assert s1.loc[GAP.stop, "segment"] == 1 and s1.loc[0, "segment"] == 0
    s6 = s[s["symbol"] == "S6"].reset_index(drop=True)
    assert (s6["segment"] == 0).all()                                         # same issuer: no segment break
    assert s["dv_trailing63"].notna().any()
    report = (outputs / "report.md").read_text()
    for section in ("## 1. Calendar", "## 2. Half-day", "## 3. Dollar volume", "## 4. Impossible", "## 5. Vendor",
                    "## 6. Security identity", "**Placeholder bars.**", "## Unresolved"):
        assert section in report


def test_a_broken_drive_link_stops_with_a_message(tmp_path, monkeypatch, capsys):
    import data_audit_v2 as script
    (tmp_path / "outputs").symlink_to(tmp_path / "drive_not_mounted" / "outputs")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["data_audit_v2.py", "--out", "outputs/data_audit_v2"])
    assert script.main() == 1
    assert "Drive is not mounted" in capsys.readouterr().out


def test_the_vendor_is_compared_unadjusted_and_the_cache_only_where_nothing_adjusts_it():
    """Prints arrive unadjusted. A later 2-for-1 split must not make the vendor look unequal to them, and the cache (split-adjusted
    and rounded to the cent, so not comparable to half a basis point) is compared only on the day nothing adjusts."""
    d1, d2 = pd.Timestamp("2024-11-04"), pd.Timestamp("2024-11-05")
    raw = pd.DataFrame({"symbol": "AAA", "date": [d1, d2], "open": [100.0, 100.0], "close": [102.0, 102.0]})
    split = pd.DataFrame({"symbol": "AAA", "date": [d1, d2], "open": [50.0, 100.0], "close": [51.0, 102.0]})   # 2-for-1 after d1
    table = split.assign(bars=78)
    factor = audit.split_factor(raw, split)
    assert factor["factor"].tolist() == [0.5, 1.0]
    auctions = pd.DataFrame({"symbol": "AAA", "date": [d1, d1, d2, d2], "side": ["open", "close"] * 2, "condition": ["Q", "M"] * 2,
                             "price": [100.0, 102.0, 100.0, 102.0], "size": 1, "t": pd.Timestamp("2024-11-04T14:30:00Z"),
                             "exchange": "N"})
    res = audit.auction_check(auctions, raw, table, factor=factor)
    summary = res["summary"].set_index("side")
    assert summary["vendor_any"].eq(1).all() and summary["symbol_days"].eq(2).all()
    assert summary["rebuilt_any"].eq(1).all() and summary["rebuilt_days"].eq(1).all()      # d1 is adjusted: not compared
    assert res["by_print"]["rebuilt_prints"].eq(1).all()
    wrong_basis = audit.auction_check(auctions, split, table, factor=factor)["summary"].set_index("side")
    assert wrong_basis["vendor_any"].eq(0.5).all()                                          # what an adjusted vendor would say
    trades = pd.DataFrame({"symbol": "AAA", "t": [pd.Timestamp("2024-11-04T14:30:00.2Z"), pd.Timestamp("2024-11-04T20:59:59Z"),
                                                  pd.Timestamp("2024-11-05T14:30:00.2Z"), pd.Timestamp("2024-11-05T20:59:59Z")],
                           "price": [100.0, 102.0, 100.0, 102.0], "size": 1, "exchange": "N", "conditions": ["O", "6", "O", "6"],
                           "tape": "A", "id": [1, 2, 3, 4], "update": None, "window": ["open", "close", "open", "close"]})
    t = audit.trades_check(trades, raw, table, factor=factor)
    assert t["vendor_equal"].eq(1).all()
    assert t.loc[t["date"] == d1, "rebuilt_equal"].isna().all()
    assert t.loc[t["date"] == d2, "rebuilt_equal"].eq(1).all()


def test_the_prints_script_runs_from_the_audit_outputs(outputs, monkeypatch, capsys):
    import audit_prints as script
    monkeypatch.setattr(sys, "argv", ["audit_prints.py", "--audit", str(outputs), "--reference", str(outputs.parent / "ref")])
    assert script.main() == 0
    summary = pd.read_csv(outputs / "auction_summary.csv").set_index("side")
    assert summary["vendor_any"].eq(1).all()
    assert "trades_open_close.parquet` is missing" in (outputs / "prints.md").read_text()


def test_the_prints_script_stops_when_the_audit_has_not_run(tmp_path, monkeypatch, capsys):
    import audit_prints as script
    monkeypatch.setattr(sys, "argv", ["audit_prints.py", "--audit", str(tmp_path), "--reference", str(tmp_path)])
    assert script.main() == 1
    assert "STOPPED" in capsys.readouterr().out


def test_the_followup_and_sample_scripts_run_from_the_audit_outputs(outputs, monkeypatch, tmp_path):
    import audit_followup as followup
    monkeypatch.setattr(sys, "argv", ["audit_followup.py", "--audit", str(outputs), "--reference", str(outputs.parent / "ref")])
    assert followup.main() == 0
    text = (outputs / "followup" / "followup.md").read_text()
    for part in ("## 1. The vendor's daily range", "## 2. Dates on which", "## 3. The vendor's adjustment steps", "## 4. The vendor's close", "## 5. Evidence"):
        assert part in text
    assert (outputs / "followup" / "range_buckets.csv").exists()
