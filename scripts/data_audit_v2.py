#!/usr/bin/env python3
"""The extended data audit, read-only: the checks the first audit (scripts/data_audit.py) left open.

    python scripts/data_audit_v2.py --sip data/bars_sip --iex data/bars --universe universe/sp500.csv \
        --reference data/reference --out outputs/data_audit_v2

1. Calendar and coverage on the NYSE calendar (a half-day ends at 13:00), by year and month, and what
   changed in 2026.
2. Half-day timestamps: bars after the close, and whether the first audit's rule caught them.
3. Dollar volume per session, and each symbol's yearly median.
4. Every impossible bar and every non-finite ratio, located.
5. Vendor daily bars against the rebuilt sessions, after checking their adjustment basis; which session
   definition the vendor uses (extended-hours sample); auction prints and trades at the open and close.
6. Security identity and corporate actions for gaps and large moves. Unresolved cases are listed apart.

Writes report.md, the tables behind it (CSV), summary.json, and sessions.parquet: one row per symbol and
NYSE session with its status and why it is not usable, the base for the volatility project's loader.
Parts that need the reference data (scripts/fetch_reference.py) are skipped with a note when it is missing.
Nothing is fitted, nothing is excluded, and no setting is chosen here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import reachable

from src import audit, nyse
from src.data import cache_path, read_meta, resolve_universe

SETTINGS = {"raw": "raw", "split": "split", "split+dividend": "split+dividend", "all": "all"}


def raw_extremes(bars: pd.DataFrame) -> pd.DataFrame:
    """What the first audit saw for IEX: high and low over every 09:30-16:00 bar, impossible ones included."""
    ts = bars.index
    tod = ts - ts.normalize()
    clock = (tod >= audit.OPEN) & (tod < audit.FULL_CLOSE)
    g = bars[clock].groupby(ts[clock].normalize())
    out = pd.DataFrame({"high_raw": g["high"].max(), "low_raw": g["low"].min(), "bars_clock": g.size()})
    out.index.name = "date"
    return out.reset_index()


def full_grid(table: pd.DataFrame, calendar: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """Every symbol on every session, so a session outside a symbol's data says why instead of vanishing."""
    grid = pd.MultiIndex.from_product([symbols, calendar], names=["symbol", "date"]).to_frame(index=False)
    out = grid.merge(table, on=["symbol", "date"], how="outer")
    first = table.groupby("symbol")["date"].min()
    last = table.groupby("symbol")["date"].max()
    missing = out["status"].isna()
    before = missing & (out["date"] < out["symbol"].map(first))
    out.loc[before, "status"] = "before_first_bar"
    out.loc[missing & ~before, "status"] = np.where(out.loc[missing & ~before, "symbol"].isin(first.index),
                                                     "after_last_bar", "no_data_for_symbol")
    out["half_day"] = out["date"].isin(nyse.HALF_DAYS)
    out["nyse_session"] = out["nyse_session"].astype("boolean").fillna(out["date"].isin(calendar)).astype(bool)
    out["slots"] = np.where(out["half_day"], 42, 78)
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_settings(ref: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for name, tag in SETTINGS.items():
        p = ref / "daily" / f"daily_sip_{tag}.parquet"
        if p.exists():
            out[name] = pd.read_parquet(p)
    p = ref / "daily" / "daily_sip_split_nomap.parquet"
    if p.exists():
        out["split_nomap"] = pd.read_parquet(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sip", default="data/bars_sip")
    ap.add_argument("--iex", default="data/bars")
    ap.add_argument("--universe", default="universe/sp500.csv")
    ap.add_argument("--reference", default="data/reference")
    ap.add_argument("--out", default="outputs/data_audit_v2")
    ap.add_argument("--start", default="2016-01-04")
    args = ap.parse_args()
    out, ref = Path(args.out), Path(args.reference)
    if not reachable(args.out, args.sip, args.iex, args.reference):
        return 1
    out.mkdir(parents=True, exist_ok=True)
    if not (ref / "manifest.json").exists():
        print(f"note: no reference data in {ref} (run section 35 first); sections 5 and 6 will be skipped")
    universe = resolve_universe(args.universe)
    calendar = nyse.sessions(args.start, "2026-12-31")
    t0 = time.time()

    tables, problems, metas = [], [], []
    for i, sym in enumerate(universe):
        bars = audit.read_bars(cache_path(args.sip, sym, "5Min"))
        if bars is None or bars.empty:
            continue
        metas.append({"symbol": sym, **{k: (read_meta(args.sip, sym, "5Min") or {}).get(k)
                                        for k in ("feed", "adjustment", "start", "end")}})
        problems.append(audit.bar_problems(bars, sym, calendar))
        tables.append(audit.session_table(bars, sym, calendar))
        if (i + 1) % 50 == 0:
            print(f"SIP {i + 1}/{len(universe)} ({time.time() - t0:.0f}s)")
    if not tables:
        print(f"no cached SIP bars under {args.sip}")
        return 1
    sip = audit.add_sequence(pd.concat(tables, ignore_index=True), calendar)
    sip_problems = pd.concat([p for p in problems if len(p)] or problems[:1], ignore_index=True)
    meta = pd.DataFrame(metas)
    print(f"SIP: {sip['symbol'].nunique()} symbols, {len(sip):,} symbol-sessions ({time.time() - t0:.0f}s)")

    iex_frames, iex_problems = [], []
    if Path(args.iex).exists():
        for sym in universe:
            bars = audit.read_bars(cache_path(args.iex, sym, "5Min"))
            if bars is None or bars.empty:
                continue
            iex_problems.append(audit.bar_problems(bars, sym, calendar))
            iex_frames.append(raw_extremes(bars).assign(symbol=sym))
        print(f"IEX: {len(iex_frames)} symbols ({time.time() - t0:.0f}s)")
    iex = pd.concat(iex_frames, ignore_index=True) if iex_frames else pd.DataFrame()
    iex_problems = pd.concat([p for p in iex_problems if len(p)] or iex_problems[:1], ignore_index=True) \
        if iex_problems else pd.DataFrame()

    # 1-2. calendar, half-days, coverage
    cal = audit.calendar_check(sip, calendar)
    common_end = cal["last"]
    halfd = audit.half_day_check(sip)
    cov_year = audit.coverage(sip[sip["date"] <= common_end], "year")
    cov_month = audit.coverage(sip[sip["date"] <= common_end], "month")
    change = audit.coverage_change(cov_year, 2026)

    # 3. dollar volume
    sip["dv_trailing63"], dv_year = audit.dollar_volume(sip)

    # 4. impossible values
    nonfinite_first = audit.first_audit_nonfinite(sip, iex) if len(iex) else pd.DataFrame()
    nonfinite_sip = audit.nonfinite_sessions(sip, "sip")

    # 5-6. against the reference data
    settings = load_settings(ref)
    have_ref = "split" in settings
    results = {}
    unresolved = []
    if have_ref:
        split = settings["split"]
        traded = audit.real(split)          # placeholder bars (no trades) are not trading days
        results["vendor"] = audit.vendor_comparison(sip, traded)
        b = results["vendor"]["basis"]
        for _, r in b[b["basis"] == "varies over time"].iterrows():
            unresolved.append({"kind": "adjustment basis varies over time", "symbol": r["symbol"],
                               "detail": f"median log ratio {r['median']:+.4f}, IQR {r['iqr']:.4f}; steps in basis_breaks.csv"})
        sample = ref / "sample"
        if (sample / "ext_5min.parquet").exists():
            ext = pd.read_parquet(sample / "ext_5min.parquet")
            results["definitions"] = audit.session_definition_check(ext, traded, sip)
        if (sample / "auctions.parquet").exists():
            results["auctions"] = audit.auction_check(pd.read_parquet(sample / "auctions.parquet"), traded, sip)
        if (sample / "trades_open_close.parquet").exists():
            results["trades"] = audit.trades_check(pd.read_parquet(sample / "trades_open_close.parquet"), traded, sip)
        ca = audit.normalize_actions(pd.read_parquet(ref / "corporate_actions.parquet")) \
            if (ref / "corporate_actions.parquet").exists() else audit.normalize_actions(pd.DataFrame())
        gap_rows = audit.gaps(sip[sip["date"] <= common_end])
        nomap = settings.get("split_nomap", pd.DataFrame(columns=["symbol", "date", "volume", "trade_count", "close"]))
        splices = audit.mapping_splices(split, nomap) if len(nomap) else pd.DataFrame()
        gap_cls = audit.classify_gaps(gap_rows, ca, split, nomap)
        moves = audit.large_moves(sip)
        move_cls = audit.classify_moves(moves, {k: settings[k] for k in SETTINGS if k in settings}, ca, gap_rows, splices)
        identity = audit.identity_vs_mapping(sip[sip["date"] <= common_end], split, universe)
        results.update(gaps=gap_cls, splices=splices, moves=move_cls, actions=ca, identity=identity,
                       placeholders=audit.placeholder_runs(split), placeholders_nomap=audit.placeholder_runs(nomap)
                       if len(nomap) else pd.DataFrame(), coverage=audit.action_coverage(ca),
                       fractional=audit.fractional_splits(ca))
        for _, g in gap_cls[gap_cls["class"].isin(["unresolved", audit.SAME_ISSUER_NO_TRADING])].iterrows():
            kind = "gap" if g["class"] == "unresolved" else "gap cause not recorded (identity: same issuer)"
            unresolved.append({"kind": kind, "symbol": g["symbol"],
                               "detail": f"{g['first_missing'].date()} to {g['last_missing'].date()}, "
                                         f"{g['sessions']} sessions; {g['evidence']}"})
        reused = set(gap_cls.loc[gap_cls["class"] == audit.REUSED, "symbol"])
        for _, r in identity.iterrows():
            if r["note"]:
                unresolved.append({"kind": "Alpaca's mapping returns no bars", "symbol": r["symbol"],
                                   "detail": f"the cache has {r['cache_days']} trading days; vendor comparison and "
                                             "issuer mapping unavailable for this symbol"})
            elif r["cache_only_days"] > 0 and r["symbol"] not in reused:
                unresolved.append({"kind": "cache has days outside today's issuer (Alpaca's mapping)", "symbol": r["symbol"],
                                   "detail": f"{r['cache_only_days']} days, {r['cache_only_first']} to {r['cache_only_last']}"})
        for _, m in move_cls[move_cls["class"] == "unresolved"].iterrows():
            unresolved.append({"kind": f"large {m['kind']} move", "symbol": m["symbol"],
                               "detail": f"{m['date'].date()} log {m['move']:+.3f}: {m['reason']}"})
        if len(splices):
            renames = ca[ca["kind"].astype(str).str.contains("name_change")]
            records_begin = renames["when"].min() if len(renames) else pd.NaT
            notes = []
            for _, s in splices.iterrows():
                if s["mapped_only_days"] == 0:
                    notes.append("")
                elif len(renames[renames["new_symbol"] == s["symbol"]]):
                    notes.append("name-change record")
                elif pd.notna(records_begin) and s["mapped_only_last"] < records_begin:
                    notes.append(f"rename before the name-change records begin ({records_begin.date()}): identity "
                                 "from Alpaca's mapping alone")
                else:
                    notes.append("no name-change record")
                    unresolved.append({"kind": "symbol mapping adds history without a name-change record",
                                       "symbol": s["symbol"],
                                       "detail": f"{s['mapped_only_days']} days, {s['mapped_only_first']} to {s['mapped_only_last']}"})
            splices["note"] = notes
        # issuer segments, only where a gap was classified as reuse on the evidence
        sip["segment"] = 0
        for _, g in gap_cls[gap_cls["class"] == audit.REUSED].iterrows():
            sip.loc[(sip["symbol"] == g["symbol"]) & (sip["date"] > g["last_missing"]), "segment"] += 1
    for _, p in sip_problems.groupby("symbol"):
        unresolved.append({"kind": "impossible bar values (cause unknown)", "symbol": p["symbol"].iloc[0],
                           "detail": f"{len(p)} bars, first {p['timestamp'].min()}; reasons: "
                                     f"{', '.join(sorted(set(';'.join(p['reason']).split(';'))))}"})
    unresolved = pd.DataFrame(unresolved, columns=["kind", "symbol", "detail"])

    # outputs
    sessions = full_grid(sip[sip["date"] <= common_end], calendar[calendar <= common_end], universe)
    sessions.to_parquet(out / "sessions.parquet", index=False)
    cov_year.to_csv(out / "coverage_by_year.csv", index=False)
    cov_month.to_csv(out / "coverage_by_month.csv", index=False)
    halfd["half_days"].to_csv(out / "half_days.csv", index=False)
    dv_year.to_csv(out / "dollar_volume_by_year.csv", index=False)
    sip_problems.to_csv(out / "invalid_bars_sip.csv", index=False)
    if len(iex_problems):
        iex_problems.to_csv(out / "invalid_bars_iex.csv", index=False)
    if len(nonfinite_first):
        nonfinite_first.to_csv(out / "first_audit_nonfinite_ratios.csv", index=False)
    nonfinite_sip.to_csv(out / "nonfinite_sessions_sip.csv", index=False)
    meta.to_csv(out / "cache_meta.csv", index=False)
    unresolved.to_csv(out / "unresolved.csv", index=False)
    if have_ref:
        v = results["vendor"]
        for name in ("basis", "basis_breaks", "by_year", "overall", "range_effect", "overnight_effect", "worst"):
            v[name].to_csv(out / f"vendor_{name}.csv", index=False)
        if "definitions" in results:
            results["definitions"]["definitions"].to_csv(out / "session_definitions.csv", index=False)
        if "auctions" in results:
            results["auctions"]["by_print"].to_csv(out / "auction_prints.csv", index=False)
        if "trades" in results:
            results["trades"].to_csv(out / "trades_open_close_check.csv", index=False)
        results["gaps"].to_csv(out / "gaps.csv", index=False)
        results["splices"].to_csv(out / "symbol_mapping_splices.csv", index=False)
        results["moves"].to_csv(out / "large_moves.csv", index=False)
        results["identity"].to_csv(out / "identity_vs_mapping.csv", index=False)
        results["placeholders"].to_csv(out / "vendor_placeholder_bars.csv", index=False)
        if len(results["placeholders_nomap"]):
            results["placeholders_nomap"].to_csv(out / "vendor_placeholder_bars_unmapped.csv", index=False)
        results["coverage"].to_csv(out / "corporate_action_coverage.csv", index=False)
        results["fractional"].to_csv(out / "fractional_split_records.csv", index=False)

    lines = report(args, sip, meta, cal, halfd, cov_year, change, dv_year, sip_problems, iex_problems,
                   nonfinite_first, nonfinite_sip, results, unresolved, ref, have_ref, common_end)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    summary = {"common_end": str(common_end.date()), "symbol_sessions": int(len(sip)),
               "coverage_change": change, "invalid_bars_sip": int(len(sip_problems)),
               "invalid_bars_iex": int(len(iex_problems)), "first_audit_nonfinite": int(len(nonfinite_first)),
               "unresolved": int(len(unresolved)), "reference": have_ref}
    if have_ref:
        counts = lambda s: {str(k): int(v) for k, v in s.value_counts().items()}  # noqa: E731
        summary["moves_by_class"] = counts(results["moves"]["class"])
        summary["gaps_by_class"] = counts(results["gaps"]["class"])
        summary["basis"] = counts(results["vendor"]["basis"]["basis"])
    audit.save_json(summary, out / "summary.json")
    print("\n".join(lines))
    print(f"\ndone in {time.time() - t0:.0f}s; outputs in {out}")
    return 0


def report(args, sip, meta, cal, halfd, cov_year, change, dv_year, sip_problems, iex_problems, nonfinite_first,
           nonfinite_sip, results, unresolved, ref, have_ref, common_end) -> list[str]:
    md = audit.md
    manifest = json.loads((ref / "manifest.json").read_text()) if (ref / "manifest.json").exists() else {}
    lines = ["# Data audit, part 2 (read-only)", "",
             "Measured values first in each section; the rules that classify are stated with them. Nothing is "
             "excluded here: the per-session table keeps every row with its status.", "",
             "## 0. Inputs", "",
             f"SIP 5-minute cache `{args.sip}`: {sip['symbol'].nunique()} symbols. Its own metadata:", "",
             md(meta.groupby(["feed", "adjustment", "start", "end"], dropna=False).size().rename("symbols").reset_index()),
             ""]
    if manifest:
        rows = [{"file": k, "endpoint": v.get("endpoint"), "settings": json.dumps({x: y for x, y in
                 (v.get("settings") or {}).items() if x in ("adjustment", "asof", "timeframe", "start", "end")}),
                 "retrieved": (v.get("retrieved_at") or {}).get("end"), "rows": v.get("rows"),
                 "missing symbols": len(v.get("missing_symbols") or [])} for k, v in manifest.get("files", {}).items()]
        lines += ["Reference data (`scripts/fetch_reference.py`; full settings and SHA-256 in its manifest.json):", "",
                  md(pd.DataFrame(rows)), ""]
    else:
        lines += ["No reference data found: sections 5 and 6 are skipped.", ""]

    lines += ["## 1. Calendar and coverage", "",
              f"Sessions with bars for at least half the symbols: {cal['sessions_data']:,}, from {cal['first'].date()} "
              f"to {cal['last'].date()} (the common end). NYSE sessions in that span: {cal['sessions_nyse']:,}. "
              f"NYSE sessions missing from the data: {cal['nyse_not_in_data'] or 'none'}. Data days that are not NYSE "
              f"sessions: {cal['data_not_nyse'] or 'none'}. Bars on non-session days: "
              f"{len(cal['bars_on_non_sessions'])} symbol-days.", "",
              "By year and day type (half = NYSE 1:00 pm close). `first_audit_*` reproduce the first audit's rule (a "
              "day was early only if no bar printed at or after 13:00):", "",
              md(cov_year[["year", "day_type", "sessions", "symbol_sessions", "ok_share", "slot_share", "missing_open",
                           "missing_close", "first_audit_early", "first_audit_missing_close"]], ".4f"), "",
              "**The 2026 change.** Full days only: slot share "
              f"{change['full_day_slot_share']:.4f} in 2026 against {change['full_day_slot_share_before']:.4f} on average "
              f"before; missing close bars {change['full_day_missing_close_rate']:.4%} of sessions against "
              f"{change['full_day_missing_close_rate_before']:.4%}. Half-day sessions by year: "
              f"{change['half_days_by_year']}. The first audit's missing close bars on half-days by year: "
              f"{change['first_audit_missing_close_on_half_days']}; on full days: "
              f"{change['first_audit_missing_close_on_full_days']}.", "",
              "## 2. Half-day timestamps", "",
              "Per NYSE half-day: symbols with bars, how many had bars at or after 13:00 (extended-hours trades kept by "
              "the cache's clock filter), the median share of the day's volume after 13:00, the median last bar inside "
              "the session (minutes after midnight), and how many the first audit called early:", "",
              md(halfd["half_days"], ".4f"), "",
              f"Full days where at least half the symbols' last bar starts before 15:00 (an unlisted early close?): "
              f"{len(halfd['unlisted_early_close_candidates'])}. Sessions where under half the symbols have a 09:30 bar "
              f"(a clock or time-zone shift?): {len(halfd['sessions_missing_0930_bar'])}.", "",
              md(halfd["unlisted_early_close_candidates"], ".3f", 20), "", md(halfd["sessions_missing_0930_bar"], ".3f", 20), "",
              "## 3. Dollar volume", "",
              "Each symbol's median daily dollar volume over its usable sessions that year (volume times each bar's "
              "VWAP, inside the session; the trailing 63-session past-only median per session is in sessions.parquet). "
              "Quantiles across symbols, and how many symbols fall below each level:", "",
              md(dv_year, ".3g"), "",
              "## 4. Impossible bars and non-finite ratios", "",
              f"SIP bars with an impossible or misplaced value: {len(sip_problems):,} "
              f"({sip_problems['symbol'].nunique() if len(sip_problems) else 0} symbols). IEX: {len(iex_problems):,}. "
              "Every one is in invalid_bars_sip.csv / invalid_bars_iex.csv; reasons:", "",
              md(sip_problems["reason"].str.split(";").explode().value_counts().rename("SIP bars").reset_index()
                 if len(sip_problems) else pd.DataFrame()), "",
              f"The first audit's IEX/SIP Parkinson ratio was non-finite on {len(nonfinite_first):,} symbol-days "
              "(first_audit_nonfinite_ratios.csv), by reason:", "",
              md(nonfinite_first["reason"].value_counts().rename("symbol-days").reset_index() if len(nonfinite_first)
                 else pd.DataFrame()), "",
              f"Rebuilt SIP sessions (impossible bars left out) with a non-finite log ratio: {len(nonfinite_sip):,}.", ""]

    if have_ref:
        v = results["vendor"]
        lines += ["## 5. Vendor daily bars against the rebuilt sessions", "",
                  "Both split-adjusted. Basis check per symbol (vendor close over rebuilt close): same within 0.1%, a "
                  "constant offset, or varying over time (not compared, listed as unresolved):", "",
                  md(v["basis"]["basis"].value_counts().rename("symbols").reset_index()), "",
                  f"Compared: {v['aligned_rows']:,} symbol-sessions. Share within {audit.TOL_BP} bp, median and 99th "
                  "percentile absolute difference, how often the vendor's range is wider, and the vendor/rebuilt volume "
                  "ratio. The `*_clock_rebuild` columns repeat the match against the first audit's 09:30-16:00 rebuild:",
                  "", md(v["overall"], ".4f"), "",
                  "What the definitions do to a range-based variance estimate (Parkinson, vendor against rebuilt):", "",
                  md(v["range_effect"], ".4g"), "",
                  "Mean squared overnight gap, consecutive sessions only, vendor against rebuilt:", "",
                  md(v["overnight_effect"], ".4g"), ""]
        if "definitions" in results:
            d = results["definitions"]
            lines += ["**Which session the vendor's daily bar uses** (sample of symbols and days, 5-minute bars fetched "
                      "with extended hours). Share of symbol-days where the vendor's field equals the bar rebuilt under "
                      "each definition, and the vendor/rebuilt volume ratio:", "",
                      md(d["definitions"], ".4f"), "",
                      f"The same 5-minute bars fetched again against the cache, inside the session: {d['cache_vs_refetch']}.", ""]
        if "auctions" in results and len(results["auctions"]["summary"]):
            lines += ["**Auction prints** (sample symbols, every day): share of symbol-days where the vendor's (rebuilt) "
                      "open or close equals an auction print, and which prints:", "",
                      md(results["auctions"]["summary"], ".4f"), "", md(results["auctions"]["by_print"], ".4f", 30), ""]
        if "trades" in results and len(results["trades"]):
            t = results["trades"]
            lines += ["**Trades at the open and close** (subsample): for each candidate print, how often the vendor's "
                      "and the rebuilt value equal it (rows in trades_open_close_check.csv):", "",
                      md(t.groupby(["window", "candidate"])[["vendor_equal", "rebuilt_equal"]].mean().reset_index(), ".3f"), ""]
        g, s, m = results["gaps"], results["splices"], results["moves"]
        ph, idt = results["placeholders"], results["identity"]
        lines += ["## 6. Security identity and corporate actions", "",
                  "**What the records cover.** Alpaca's corporate-action records by kind; where a kind starts late, "
                  "older events of that kind can only be seen in the adjustment factors or not at all:", "",
                  md(results["coverage"]), "",
                  "Split records whose ratio is not a whole number either way (several match spin-offs, which "
                  "Alpaca's split adjustment then covers):", "", md(results["fractional"], ".4f", 40), "",
                  f"**Placeholder bars.** Vendor daily bars with zero volume or zero trades (no trading under the "
                  f"ticker; prices repeat the last close): {int(ph['bars'].sum()) if len(ph) else 0} bars in "
                  f"{ph['symbol'].nunique() if len(ph) else 0} symbols with symbol mapping on"
                  + (f", {int(results['placeholders_nomap']['bars'].sum())} with it off"
                     if len(results["placeholders_nomap"]) else "") + ". They are left out of every comparison.", "",
                  md(ph.sort_values("bars", ascending=False) if len(ph) else ph, ".2f", 20), "",
                  "**The cache against today's issuer.** Trading days the cache has that Alpaca's issuer mapping "
                  "(as of the reference retrieval) does not give today's issuer, and the reverse:", "",
                  md(idt.sort_values("cache_only_days", ascending=False) if len(idt) else idt, ".0f", 30), "",
                  f"Gaps of 5+ sessions with no bar: {len(g)} ({(g['sessions'] >= audit.LONG_GAP).sum()} of "
                  f"{audit.LONG_GAP}+). Rule for long gaps: an exit record for an earlier issuer under the ticker near the "
                  "gap's start means reuse; bars under the bare ticker during the gap mean the cache is missing them; "
                  "otherwise unresolved.", "",
                  md(g[g["sessions"] >= audit.LONG_GAP][["symbol", "first_missing", "last_missing", "sessions",
                                                         "log_move_across", "class", "evidence"]]), "",
                  f"Symbols whose history Alpaca's symbol mapping extends with an earlier ticker: "
                  f"{int((s['mapped_only_days'] > 0).sum()) if len(s) else 0} (symbol_mapping_splices.csv).", "",
                  md(s[s["mapped_only_days"] > 0] if len(s) else s, ".0f", 40), "",
                  f"Large moves (|log| > {audit.JUMP}) on the calendar-aware sessions: {len(m)}. Classes (rules in "
                  "src/audit.py classify_moves; market moves are genuine events and stay in the data):", "",
                  md(m["class"].value_counts().rename("moves").reset_index()), "",
                  md(m[m["class"] != "market move (no corporate action found)"][
                      ["symbol", "date", "kind", "move", "class", "reason", "records"]], ".3f", 60), ""]
    lines += ["## Unresolved", "",
              f"{len(unresolved)} cases (unresolved.csv), by kind:", "",
              md(unresolved["kind"].value_counts().rename("cases").reset_index()), "",
              md(unresolved, ".3f", 80), ""]
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
