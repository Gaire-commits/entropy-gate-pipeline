#!/usr/bin/env python3
"""E5: does a stock catch up with its sector? Judged by the rule in the config.

    python scripts/leadlag.py --config configs/sip_e5_leadlag.yaml

Reads the cached SIP bars (no training). Writes outputs/sip_e5_leadlag/leadlag.md and leadlag.csv.
The method is in src/leadlag.py.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import output_dir

from src.capacity import paired_ic_difference
from src.config import load_config
from src.crosssection import breakeven_ic, cross_sectional_ic, cross_sectional_sigma, quantile_book, score_book
from src.data import load_universe, resolve_universe
from src.leadlag import collect, verdict

NAMES = {"gap": "gap to sector peers (the signal)", "gap_shuffled": "gap with shuffled sectors (control)",
         "gap_leaders": "gap to the 3 most traded peers", "peers": "peers' move only", "own": "own move reversed only"}


def fmt(x, spec="+.4f") -> str:
    return "—" if x is None or not np.isfinite(x) else format(x, spec)


def evaluate(frame: pd.DataFrame, e, n_boot: int) -> dict:
    """ICs of every signal, the paired control, the halves, the book and what the data can detect."""
    def scored(col):
        f = frame.dropna(subset=[col])
        return f.assign(prob=f[col])

    gap = scored("gap")
    out = {"ic": {k: cross_sectional_ic(scored(k), min_peers=e.min_peers, n_boot=n_boot) for k in NAMES}}
    out["paired"] = paired_ic_difference(gap, scored("gap_shuffled"), e.min_peers, n_boot, level=0.95)
    split = pd.Timestamp(e.split)
    out["early"] = cross_sectional_ic(gap[gap["date"] < split], min_peers=e.min_peers, n_boot=n_boot)
    out["late"] = cross_sectional_ic(gap[gap["date"] >= split], min_peers=e.min_peers, n_boot=n_boot)
    out["book"] = score_book(quantile_book(gap, q=e.q, min_peers=e.min_peers), e.cost_bps, n_boot)
    out["sigma"] = cross_sectional_sigma(gap, e.min_peers)
    out["needed"] = breakeven_ic(e.cost_bps, out["sigma"], e.q)
    out["by_year"] = {int(y): cross_sectional_ic(g, min_peers=e.min_peers, n_boot=200)
                      for y, g in gap.groupby(gap["date"].dt.year)}
    out["rows"], out["symbols"] = len(frame), frame["symbol"].nunique()
    return out


def table(res: dict) -> list[str]:
    lines = ["| signal | IC [95% CI] | t | days IC > 0 |", "|---|---|---:|---:|"]
    for k, name in NAMES.items():
        s = res["ic"][k]
        lines.append(f"| {name} | {fmt(s.get('ic'))} [{fmt(s.get('ic_lo'))}, {fmt(s.get('ic_hi'))}] | "
                     f"{fmt(s.get('t'), '+.1f')} | {fmt(s.get('share_days_positive'), '.0%')} |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sip_e5_leadlag.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out, e = output_dir(cfg), cfg.leadlag
    n_boot = int(cfg.evaluation.bootstrap)
    universe = pd.read_csv(cfg.data.universe)
    sectors = dict(zip(universe["symbol"], universe["sector"]))
    t0 = time.time()
    bars = load_universe(cfg.data.cache_dir, resolve_universe(cfg.data.universe), cfg.data.timeframe)
    if not bars:
        print(f"no cached bars under {cfg.data.cache_dir} — fetch the SIP bars first (Colab section 21)")
        return 1
    print(f"{len(bars)} symbols loaded ({time.time() - t0:.0f}s)")

    common = dict(embargo=e.embargo, min_sector=e.min_sector, n_leaders=e.n_leaders, leader_days=e.leader_days,
                  seed=e.shuffle_seed)
    frame = collect(bars, sectors, lookback=e.lookback, hold=e.hold, stride=e.stride, **common)
    print(f"primary: {len(frame):,} stock-moments ({time.time() - t0:.0f}s)")
    res = evaluate(frame, e, n_boot)
    del frame
    s2 = e.secondary
    frame2 = collect(bars, sectors, lookback=s2.lookback, hold=s2.hold, stride=s2.stride, **common)
    del bars
    res2 = evaluate(frame2, e, n_boot)
    del frame2

    v = verdict(res["ic"]["gap"], res["paired"], res["early"], res["late"], res["book"], e.cost_bps, e.t_bar)
    b, g = res["book"], res["ic"]["gap"]
    lines = [f"# {cfg.experiment}: does a stock catch up with its sector?", "",
             f"**Verdict: {v['level'].upper()}** — {v['reason']}.", "",
             f"{res['symbols']} symbols, {res['rows']:,} stock-moments. Gap over the last {e.lookback} bars, entry "
             f"{e.embargo} bar later, {e.hold}-bar hold; no stale prices. IC is the rank correlation across stocks at "
             "each moment, with 95% day-block intervals.", "", "## Primary", ""] + table(res) + [
             "", f"- **Real sectors against shuffled ones** (same moments): IC difference {fmt(res['paired']['diff'])} "
             f"[{fmt(res['paired']['lo'])}, {fmt(res['paired']['hi'])}].",
             f"- **Halves:** before {e.split} {fmt(res['early'].get('ic'))} (t {fmt(res['early'].get('t'), '+.1f')}); "
             f"from it {fmt(res['late'].get('ic'))} (t {fmt(res['late'].get('t'), '+.1f')}).",
             f"- **Long-short book** (top and bottom {e.q:.0%} by gap): gross {fmt(b.get('gross'), '+.2f')} bps "
             f"[{fmt(b.get('gross_lo'), '+.2f')}, {fmt(b.get('gross_hi'), '+.2f')}] per position, net "
             f"{fmt(b.get('net'), '+.2f')} after {e.cost_bps:g} bps; long leg {fmt(b.get('long_leg'), '+.2f')}, short leg "
             f"{fmt(b.get('short_leg'), '+.2f')}; {fmt(b.get('positions_per_day'), '.0f')} positions a day.",
             f"- **What it would take:** stocks differ by about {res['sigma']:.0f} bps over the hold, so the book needs an "
             f"IC of {res['needed']:.4f} to pay; this data detects about {fmt(g.get('ic_min_detectable'), '.4f')}.",
             "", "**By year** (the gap's IC):", "", "| year | IC | t |", "|---|---:|---:|"]
    lines += [f"| {y} | {fmt(s.get('ic'))} | {fmt(s.get('t'), '+.1f')} |" for y, s in res["by_year"].items()]
    lines += ["", f"## Reported: {s2.lookback}-bar lookback, {s2.hold}-bar hold (the pipeline's usual geometry)", ""] + table(res2)
    b2 = res2["book"]
    lines += ["", f"Book: gross {fmt(b2.get('gross'), '+.2f')} bps [{fmt(b2.get('gross_lo'), '+.2f')}, "
              f"{fmt(b2.get('gross_hi'), '+.2f')}], net {fmt(b2.get('net'), '+.2f')}; real against shuffled sectors "
              f"{fmt(res2['paired']['diff'])} [{fmt(res2['paired']['lo'])}, {fmt(res2['paired']['hi'])}]."]
    text = "\n".join(lines) + "\n"
    (out / "leadlag.md").write_text(text)
    pd.DataFrame([{"geometry": name, "signal": k, **r["ic"][k]} for name, r in (("primary", res), ("secondary", res2))
                  for k in NAMES]).to_csv(out / "leadlag.csv", index=False)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
