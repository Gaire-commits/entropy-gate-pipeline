#!/usr/bin/env python3
"""The volatility project's candidate sample, read-only: the panel (src/volpanel.py), why each value is missing,
eligible stock-days by window and warm-up, and the regimes each window contains.

    python scripts/vol_sample.py --audit outputs/data_audit_v2 --reference data/reference --out outputs/vol_sample

Nothing is fitted and no forecast is looked at. Writes report.md, the tables behind it, events.csv (every session whose
overnight return is not a clean overnight, with the source that named it), and panel.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import reachable

from src import audit, nyse, volpanel
from src.volpanel import WINDOWS, Rules

ROOT = Path(__file__).resolve().parents[1]
KEEP = ["symbol", "date", "half_day", "session_index", "segment", "open", "high", "low", "close", "raw_close", "volume",
        "trade_count", "dv63", "bars5", "slots5", "range_d", "trunc", "event_source", "reason_id", "reason", "rho", "r_on", "V",
        "ret"]


def regimes(p: pd.DataFrame, windows: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Market-wide variance per session (the cross-section's mean V over valid, liquid stock-sessions), by window and by year,
    and the highest sessions. A description of what each window contains, nothing fitted."""
    q = p[p["V"].notna() & (p["raw_close"] >= Rules().min_raw_close) & (p["dv63"] >= Rules().min_dollar_volume)]
    m = q.groupby("date")["V"].agg(["mean", "median", "size"]).rename(columns={"mean": "market_V", "median": "median_V", "size": "stocks"})
    base = m["market_V"].median()
    m["window"] = volpanel.window_of(pd.Series(m.index, index=m.index), windows)
    m["year"] = m.index.year
    rows = []
    for name in list(windows) + ["all sessions"]:
        x = m if name == "all sessions" else m[m["window"] == name]
        if x.empty:
            continue
        top = x["market_V"].idxmax()
        rows.append({"window": name, "sessions": len(x), "mean market_V": x["market_V"].mean(),
                     "typical daily vol, % (sqrt of mean)": 100 * np.sqrt(x["market_V"].mean()),
                     "median market_V": x["market_V"].median(), "p95 market_V": x["market_V"].quantile(0.95),
                     "sessions above 3x the all-session median": int((x["market_V"] > 3 * base).sum()),
                     "highest session": top.date(), "its market_V": x.loc[top, "market_V"],
                     "stocks per session, median": int(x["stocks"].median())})
    by_year = m.groupby("year").agg(sessions=("market_V", "size"), mean_market_V=("market_V", "mean"),
                                    median_stocks=("stocks", "median")).reset_index()
    by_year["typical daily vol, %"] = 100 * np.sqrt(by_year["mean_market_V"])
    return pd.DataFrame(rows), by_year, m.sort_values("market_V", ascending=False).head(15).reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="outputs/data_audit_v2")
    ap.add_argument("--reference", default="data/reference")
    ap.add_argument("--universe", default=str(ROOT / "universe" / "sp500.csv"))
    ap.add_argument("--registry", default=str(ROOT / "registry" / "vol_registry.csv"))
    ap.add_argument("--out", default="outputs/vol_sample")
    ap.add_argument("--end", default="2026-10-12")
    args = ap.parse_args()
    audit_dir, ref, out = Path(args.audit), Path(args.reference), Path(args.out)
    if not reachable(audit_dir, ref, out):
        return 1
    need = [audit_dir / "sessions.parquet", audit_dir / "large_moves.csv", ref / "corporate_actions.parquet"] + \
           [ref / "daily" / f"daily_sip_{t}.parquet" for t in ("raw", "split", "all")]
    if [p for p in need if not p.exists()]:
        print("STOPPED: missing", ", ".join(str(p) for p in need if not p.exists()), "- run sections 35 and 36 first.")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    rules = Rules()
    sessions = pd.read_parquet(audit_dir / "sessions.parquet")
    daily = {t: pd.read_parquet(ref / "daily" / f"daily_sip_{t}.parquet") for t in ("raw", "split", "all")}
    vendor = volpanel.vendor_frame(daily["raw"], daily["split"], daily["all"])
    actions = audit.normalize_actions(pd.read_parquet(ref / "corporate_actions.parquet"))
    registry = volpanel.load_registry(args.registry)
    symbols = sorted(pd.read_csv(args.universe)["symbol"])
    p = volpanel.build_panel(sessions, vendor, actions, registry, symbols, rules, end=args.end)
    cal = nyse.sessions("2016-01-04", args.end)
    events = volpanel.event_sessions(actions, vendor, registry, cal, rules)
    events = events[events["symbol"].isin(symbols)].sort_values(["date", "symbol"])
    events.to_csv(out / "events.csv", index=False)
    p[KEEP].to_parquet(out / "panel.parquet")

    sizes = volpanel.sample_sizes(p, rules, windows=WINDOWS)
    sizes.to_csv(out / "sample_sizes.csv", index=False)
    reasons = volpanel.reason_counts(p, WINDOWS)
    reasons.to_csv(out / "reasons.csv")
    win, year, top = regimes(p, WINDOWS)
    win.to_csv(out / "regimes_by_window.csv", index=False)
    year.to_csv(out / "regimes_by_year.csv", index=False)

    lm = pd.read_csv(audit_dir / "large_moves.csv", parse_dates=["date"])
    keyed = p.set_index(["symbol", "date"])
    lm["reason"] = [keyed["reason"].get((s, d), "(not in panel)") if k == "overnight" else keyed["reason_id"].get((s, d), "(not in panel)")
                    for s, d, k in zip(lm["symbol"], lm["date"], lm["kind"])]
    lm["kept"] = lm["reason"] == ""
    lm[["symbol", "date", "kind", "move", "class", "kept", "reason"]].to_csv(out / "large_moves_check.csv", index=False)
    lm["group"] = np.where(lm["class"].str.startswith("market move"), "market move (no record found)", "corporate action, identity or unresolved")
    kept = lm.groupby("group")["kept"].agg(["size", "sum"]).rename(columns={"size": "moves", "sum": "kept in the target"})
    why = lm[~lm["kept"]].groupby(["group", "reason"]).size().rename("moves").reset_index()

    lines = ["# The volatility project's candidate sample (read-only)", "",
             f"Panel: {len(symbols)} symbols x {p['date'].nunique():,} NYSE sessions ({p['date'].min().date()} to {p['date'].max().date()}), "
             f"{len(p):,} rows. Rules: {rules}.", "",
             "## Eligible stock-days by window of the target date and warm-up", "", audit.md(sizes, ",.0f"), "",
             "## Why the target is missing (stock-sessions in the window, inside the security's span)", "", audit.md(reasons.reset_index(), ",.0f"), "",
             "## Regimes (market-wide variance per session: the cross-section's mean V)", "", audit.md(win, ".4g"), "",
             audit.md(year, ".4g"), "", "Highest sessions:", "", audit.md(top, ".4g"), "",
             "## Do the rules keep genuine extreme events? The audit's large moves", "", audit.md(kept.reset_index(), ",.0f"), "",
             audit.md(why, ",.0f"), "",
             f"## Event sessions ({len(events):,}; events.csv)", "", audit.md(events["source"].value_counts().rename("sessions").reset_index(), ",.0f"), ""]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
