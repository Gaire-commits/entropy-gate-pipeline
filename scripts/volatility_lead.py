#!/usr/bin/env python3
"""E1: test the +15.21 bps volatile-moment lead once, without a model, against a rule fixed in advance.

    python scripts/volatility_lead.py --config configs/e1_volatility.yaml

Reads the cached S&P 500 bars (no training). Writes outputs/e1_volatility/e1.md and e1.json. The rule
and its order are in the config; the method is in src/volatility_lead.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import output_dir

from src.config import load_config
from src.data import load_universe, resolve_universe
from src.dataset import build_dataset
from src.experiment import ensemble, load_predictions
from src.selective import selective_table, trailing_volatility
from src.volatility_lead import LeadParams, analyse


def fmt(x, spec="+.2f") -> str:
    return "—" if x is None or not np.isfinite(x) else format(x, spec)


def ci(r: dict) -> str:
    return f"{fmt(r['mean'])} [{fmt(r['lo'])}, {fmt(r['hi'])}]"


def params(e) -> LeadParams:
    return LeadParams(share=e.share, burn_in_days=e.burn_in_days, block_days=e.block_days, base_cost_bps=e.base_cost_bps,
                      cost_cap=e.cost_cap, hedge_cost_bps=e.hedge_cost_bps, discovery_start=e.discovery_start,
                      claim_bps=e.claim_bps, n_boot=e.n_boot, min_peers=e.min_peers, hot_quantile=e.hot_quantile,
                      merge_gap_days=e.merge_gap_days, top_days=e.top_days, excluded=tuple(e.excluded),
                      stress_share=e.stress_share, other_shares=tuple(e.other_shares),
                      labels=tuple(getattr(e, "labels", ("2020-2022", "the test quarters"))))


def reproduce(pred_dir: Path, bars, horizon: int, n_boot: int) -> dict | None:
    """The original cell: logistic regression's direction at its top 2% by volatility (selective.py's placebo)."""
    pred = load_predictions(pred_dir)
    if pred.empty or not (pred["arch"] == "logreg").any():
        return None
    frame = ensemble(pred[pred["arch"] == "logreg"])
    keys = frame[["symbol", "date", "bar_index"]].drop_duplicates(ignore_index=True)
    keys["vol"] = trailing_volatility(bars, keys).to_numpy()
    frame = frame.merge(keys, on=["symbol", "date", "bar_index"], how="left")
    table = selective_table(frame, cost=2.0, horizon=horizon, shares=[0.02], n_boot=n_boot)
    row = table[table["selector"] != "confidence"]
    return None if row.empty else row.iloc[0].to_dict()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e1_volatility.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out, p = output_dir(cfg), params(cfg.e1)

    t0 = time.time()
    bars = load_universe(cfg.data.cache_dir, resolve_universe(cfg.data.universe), cfg.data.timeframe)
    if not bars:
        print(f"no cached bars under {cfg.data.cache_dir} — run scripts/fetch_data.py first")
        return 1
    data = build_dataset(bars, None, cfg)
    rows = pd.DataFrame({"date": pd.to_datetime(data["date"]), "bar_index": data["bar_index"], "symbol": data["symbol"],
                         "ret_bps": data["ret"] * 1e4})
    del data
    rows["vol"] = trailing_volatility(bars, rows[["symbol", "date", "bar_index"]]).to_numpy()
    print(f"{len(bars)} symbols, {len(rows):,} stock-moments, volatility for {rows['vol'].notna().mean():.1%} "
          f"({time.time() - t0:.0f}s)")
    rows = rows[rows["vol"].notna()]

    res = analyse(rows, p)
    rep = None
    src = Path(getattr(cfg.e1, "reproduce_from", ""))
    if src and (src / "predictions").exists():
        rep = reproduce(src, bars, int(cfg.features.horizon), p.n_boot)
    write_report(cfg, out, p, res, rep)
    return 0


def write_report(cfg, out: Path, p: LeadParams, res: dict, rep: dict | None) -> None:
    v = res["verdict"]
    pre, disc = res["pre"], res["disc"]
    early, late = p.labels
    names = {"pre": f"untouched ({res['periods']['pre'][0]} to {res['periods']['pre'][1]})",
             "disc": f"{late} ({res['periods']['disc'][0]} to {res['periods']['disc'][1]})"}
    lines = [f"# {cfg.experiment}: the volatile-moment lead, tested once", "",
             f"**Verdict: {v['level'].upper()}** — {v['reason']}. "
             + ("The gain survives removing the market's move (stock-level)." if v["stock_level"] else
                "Market-neutral, the gain does not hold in both periods: it is a bet on the market, not on stocks."),
             "", f"Rule (no model): long every stock-moment in the top {p.share:.0%} by trailing volatility, cutoff from "
             f"earlier months only; {cfg.features.horizon}-bar hold. bps per trade, 95% intervals from blocks of "
             f"{p.block_days} trading days. Stress cost: {p.base_cost_bps:g} bps x volatility / earlier median, "
             f"{1:g}x to {p.cost_cap:g}x. The protocol is in the config.", "",
             "## Primary results", "",
             "| period | trades (days with any) | gross | net, flat 2 bps | net, stress cost (mean) | market-neutral gross | "
             "market-neutral net | detectable |", "|---|---|---|---|---|---|---|---:|"]
    for k in ("pre", "disc"):
        r = res[k]
        lines.append(f"| {names[k]} | {r['gross']['rows']:,} ({r['gross']['days']}) | {ci(r['gross'])} | "
                     f"{ci(r['net_flat'])} | {ci(r['net_stress'])} ({fmt(r['mean_cost_stress'], '.1f')}) | "
                     f"{ci(r['neutral_gross'])} | {ci(r['neutral_net_stress'])} | {fmt(r['gross']['mde'], '.1f')} |")
    lines += ["", "**detectable**: the gross effect this period would detect 80% of the time (2.8 standard errors).", "",
              f"## Does it rest on a few days? ({late})", "",
              "| left out | gross without it | its share of the gross |", "|---|---|---:|"]
    le = disc.get("largest_episode")
    if le:
        share = next((e["pnl_share"] for e in disc["episodes"] if e["start"] == le["start"]), np.nan)
        lines.append(f"| largest volatile episode, {le['start']} to {le['end']} | {ci(le['without'])} | {fmt(share, '.0%')} |")
    lines.append(f"| its {p.top_days} biggest days | {ci(disc['top_days']['without'])} | {fmt(disc['top_days']['pnl_share'], '.0%')} |")
    lines.append(f"| {p.excluded[0]} to {p.excluded[1]} | {ci(disc['without_excluded'])} | |")

    lines += ["", "## Described, not decisive", "", "**Other shares** (gross):", "",
              f"| share | untouched | {late} |", "|---|---|---|"]
    for sh in p.other_shares:
        lines.append(f"| top {sh:.0%} | {ci(pre['other_shares'][sh])} | {ci(disc['other_shares'][sh])} |")
    lines += ["", f"**Market stress** (moments whose average volatility is in the top {p.stress_share:.0%}), gross:", "",
              "| period | selected trades in stress | in stress | outside stress | every stock-moment |", "|---|---:|---|---|---|"]
    for k in ("pre", "disc"):
        r = res[k]
        lines.append(f"| {names[k]} | {fmt(r['market_stress_rows_share'], '.0%')} | {ci(r['market_stress'])} | "
                     f"{ci(r['market_calm'])} | {ci(r['all_rows_gross'])} |")
    lines += ["", "**By year** (gross, trades):", "", "| year | gross | trades |", "|---|---|---:|"]
    for k in ("pre", "disc"):
        for y, r in res[k]["by_year"].items():
            lines.append(f"| {y} ({'untouched' if k == 'pre' else 'test'}) | {ci(r)} | {r['rows']:,} |")
    lines += ["", "**By signal time** (gross):", "", f"| bar | untouched | {late} |", "|---|---|---|"]
    for b in sorted(set(pre["by_bar"]) | set(disc["by_bar"])):
        cells = [ci(res[k]["by_bar"][b]) if b in res[k]["by_bar"] else "—" for k in ("pre", "disc")]
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    lines += ["", f"**Volatile episodes in {late}**:", "", "| start | end | selected trades | share of gross |",
              "|---|---|---:|---:|"]
    for e in sorted(disc["episodes"], key=lambda e: -abs(e["pnl_share"]) if np.isfinite(e["pnl_share"]) else 0)[:10]:
        lines.append(f"| {e['start']} | {e['end']} | {e['rows']:,} | {fmt(e['pnl_share'], '.0%')} |")
    lines += ["", "## Reproduction of the original cell", ""]
    if rep:
        lines.append(f"Logistic regression's direction on its top 2% by volatility (scripts/selective.py's placebo, saved "
                     f"predictions): {fmt(rep.get('gross'))} bps gross [{fmt(rep.get('gross_lo'))}, {fmt(rep.get('gross_hi'))}], "
                     f"{rep.get('long_share', np.nan):.0%} long, {int(rep.get('trades', 0)):,} trades (RESULTS.md: +15.21 [+1.57, +29.18]).")
    else:
        lines.append("Saved sp500_multihour predictions not found; not reproduced.")
    text = "\n".join(lines) + "\n"
    (out / "e1.md").write_text(text)
    (out / "e1.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("\n" + text)


if __name__ == "__main__":
    raise SystemExit(main())
