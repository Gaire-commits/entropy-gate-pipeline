#!/usr/bin/env python3
"""Follow-up tables for the extended audit, read-only (about a minute): see src/audit_followup.py.

    python scripts/audit_followup.py --audit outputs/data_audit_v2 --reference data/reference

Reads the audit's sessions.parquet and large_moves.csv and the reference data; writes followup.md and the tables behind it
into <audit>/followup/. Nothing is fitted and nothing is excluded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import reachable

from src import audit, audit_followup as fu, volpanel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="outputs/data_audit_v2")
    ap.add_argument("--reference", default="data/reference")
    args = ap.parse_args()
    audit_dir, ref = Path(args.audit), Path(args.reference)
    out = audit_dir / "followup"
    if not reachable(audit_dir, ref):
        return 1
    need = [audit_dir / "sessions.parquet", audit_dir / "large_moves.csv", ref / "corporate_actions.parquet"] + \
           [ref / "daily" / f"daily_sip_{t}.parquet" for t in ("raw", "split", "all")]
    if [p for p in need if not p.exists()]:
        print("STOPPED: missing", ", ".join(str(p) for p in need if not p.exists()), "- run sections 35 and 36 first.")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    sessions = pd.read_parquet(audit_dir / "sessions.parquet")
    daily = {t: pd.read_parquet(ref / "daily" / f"daily_sip_{t}.parquet") for t in ("raw", "split", "all")}
    vendor = volpanel.vendor_frame(daily["raw"], daily["split"], daily["all"])
    actions = audit.normalize_actions(pd.read_parquet(ref / "corporate_actions.parquet"))
    moves = pd.read_csv(audit_dir / "large_moves.csv", parse_dates=["date"])

    rc = fu.range_conflicts(sessions, vendor)
    tr = fu.truncation(sessions, vendor)
    st = fu.adjustment_steps(vendor, actions)
    ev = fu.move_evidence(moves, vendor, sessions)
    cg = fu.close_gaps(sessions, vendor)
    mv = fu.market_move_volume(ev)
    for name, table in {"range_buckets": rc["buckets"], "range_conflicts_by_year": rc["by_year"], "range_worst": rc["worst"],
                        "truncated_days": tr["flagged"], "adjustment_steps_by_size": st["by_size"],
                        "split_steps_without_record": st["unmatched"], "records_without_vendor_step": st["records_without_a_vendor_step"],
                        "large_cash_steps": st["large_cash_steps"], "move_evidence": ev, "close_gaps_by_year": cg["by_year"],
                        "market_move_volume": mv}.items():
        table.to_csv(out / f"{name}.csv", index=False)

    lines = ["# Follow-up to the extended audit (read-only)", "",
             "## 1. The vendor's daily range against the range rebuilt from the 5-minute cache",
             "", "log(vendor range / rebuilt range), by bucket and by whether the cache holds every slot of the day:", "",
             audit.md(rc["buckets"], ",.4g"), "", "Sessions beyond the threshold, by year and direction:", "",
             audit.md(rc["by_year"], ",.0f", 60), "", "Means: " + "; ".join(f"{k} {v:,.6g}" for k, v in rc["means"].items()), "",
             "Largest disagreements on complete cache days:", "", audit.md(rc["worst"], ".4g"), "",
             "## 2. Dates on which a source holds a fraction of the session", "",
             "The cache's missing share of slots and the vendor's trade count over its trailing median, full days only "
             "(a date is the vendor's truncated day when the cross-section's median is under 5%):", "",
             audit.md(tr["flagged"], ".4g", 40), "", "The three lowest vendor ratios among the dates not flagged:", "",
             audit.md(tr["lowest_unflagged_trade_count_ratio"], ".4g"), "",
             f"Stock-sessions on the other dates with a trade count under 5% of their own trailing median: "
             f"{tr['stock_sessions_under_ratio_elsewhere']:,} of {tr['stock_sessions_with_a_ratio_elsewhere']:,}.", "",
             "## 3. The vendor's adjustment steps against the corporate-action records", "", audit.md(st["by_size"], ",.0f"), "",
             f"Split-type steps above 0.001: {st['split_steps']}; on the same session as a split or spin-off record: {st['matched']}. "
             "The rest:", "", audit.md(st["unmatched"], ".4f"), "",
             f"Split, stock-dividend and spin-off records: {st['records']}. Those with no vendor step above 1% on the session "
             "(the vendor did not adjust, or the records name an older ticker):", "",
             audit.md(st["records_without_a_vendor_step"], ".4f", 40), "",
             "Dividend or spin-off adjustment steps above 3%:", "", audit.md(st["large_cash_steps"], ".4f", 40), "",
             "## 4. The vendor's close against the cache's last bar", "",
             f"Compared {cg['compared']:,} sessions (both split-adjusted). The close is equal within 0.5 bp on {cg['within_half_bp']:.1%}; "
             f"{cg['beyond']:,} differ by more than 1%, of which {cg['share_without_last_bar']:.1%} lack the cache's last slot "
             f"and {cg['share_on_complete_days']:.1%} are on complete cache days. {cg['beyond_10pct']} differ by more than 10%, "
             f"{cg['beyond_10pct_with_last_bar']} of them with the cache's last slot present (by year {cg['beyond_10pct_by_year']}). "
             "By year, beyond 1%:", "", audit.md(cg["by_year"], ",.0f"), "",
             "## 5. Evidence behind each large move", "", "The audit's 'market move (no record found)' days by volume:", "",
             audit.md(mv, ",.0f"), "",
             audit.md(ev[["symbol", "date", "kind", "move", "class", "vol_ratio", "trade_count", "close_gap_pct", "bars"]]
                      .sort_values("date"), ".4g", 200), ""]
    (out / "followup.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
