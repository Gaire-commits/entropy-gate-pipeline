#!/usr/bin/env python3
"""E11: does the reversal appear in a second sample of stocks, and is it stronger in less liquid ones?

    python scripts/build_universe.py --index sp400
    python scripts/fetch_data.py --config configs/sip_e11_midcap.yaml --workers 3
    python scripts/second_sample.py --config configs/sip_e11_midcap.yaml

Reads the cached SIP bars of both universes (no training). Writes outputs/<experiment>/second_sample.md
and .csv. The signal and holds are E10's; the rule is in the config (src/horizons.replication_verdict).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import enough_symbols, output_dir

from src.capacity import paired_ic_difference
from src.config import load_config
from src.crosssection import cross_sectional_ic, quantile_book, score_book
from src.data import load_universe, resolve_universe
from src.horizons import day_ic, hold_frames, paired_day_difference, replication_verdict

NAME = {6: "30 minutes", 12: "1 hour", 24: "2 hours", 48: "4 hours", "day": "a full day"}


def fmt(x, spec="+.4f") -> str:
    return "—" if x is None or not np.isfinite(x) else format(x, spec)


def load(spec, cache_dir, timeframe, label):
    symbols = resolve_universe(spec)
    bars = load_universe(cache_dir, symbols, timeframe)
    if not bars:
        print(f"no cached bars for the {label} universe under {cache_dir} — fetch it first")
        return None, None
    if not enough_symbols(bars, symbols, cache_dir):
        return None, None
    table = pd.read_csv(spec)
    return bars, dict(zip(table["symbol"], table["sector"]))


def profile(frames: dict, e, n_boot: int) -> dict:
    out = {}
    for h, f in frames.items():
        out[h] = {"ic": cross_sectional_ic(f, min_peers=e.min_peers, n_boot=n_boot),
                  "book": score_book(quantile_book(f, q=e.q, min_peers=e.min_peers), e.cost_bps, n_boot),
                  "days": day_ic(f, min_peers=e.min_peers), "rows": len(f)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sip_e11_midcap.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out, e, s = output_dir(cfg), cfg.holds, cfg.second
    n_boot = int(cfg.evaluation.bootstrap)
    t0 = time.time()
    log = lambda m: print(f"  {m} ({time.time() - t0:.0f}s)")

    results, frames_by = {}, {}
    shared = sorted(set(resolve_universe(cfg.data.universe)) & set(resolve_universe(s.compare_universe)))
    for label, spec in (("mid-cap", cfg.data.universe), ("S&P 500", s.compare_universe)):
        bars, sectors = load(spec, cfg.data.cache_dir, cfg.data.timeframe, label)
        if bars is None:
            return 1
        if label == "mid-cap":                     # a stock in both lists belongs to the S&P 500 sample only
            bars = {k: v for k, v in bars.items() if k not in shared}
        print(f"{label}: {len(bars)} symbols loaded ({time.time() - t0:.0f}s)")
        frames_by[label] = hold_frames(bars, sectors, e.lookback, e.embargo, tuple(e.offsets), tuple(e.holds),
                                       e.min_peers, log=log)
        del bars
        results[label] = profile(frames_by[label], e, n_boot)

    mid, big = results["mid-cap"], results["S&P 500"]
    fm, kw = frames_by["mid-cap"], dict(min_peers=e.min_peers, n_boot=n_boot, level=e.level)
    hour_vs_30 = paired_ic_difference(fm[12], fm[6], **kw)
    day_vs_4h = paired_ic_difference(fm["day"], fm[48], **kw)
    size = paired_day_difference(mid[6]["days"], big[6]["days"], n_boot, level=e.level)
    split = pd.Timestamp(e.split)
    halves = {k: paired_day_difference(mid[6]["days"][mid[6]["days"].index < split] if k == "early" else mid[6]["days"][mid[6]["days"].index >= split],
                                       big[6]["days"][big[6]["days"].index < split] if k == "early" else big[6]["days"][big[6]["days"].index >= split],
                                       n_boot, level=e.half_level) for k in ("early", "late")}
    v = replication_verdict(mid[6]["ic"], hour_vs_30, day_vs_4h, size, e.t_bar)

    lines = [f"# {cfg.experiment}: the reversal in a second sample of stocks", "",
             f"**Replication: {v['level'].upper()}** — {v['reason']}.", "",
             f"**Size: {v['size'].upper()}** — mid-cap 30-minute IC minus the S&P 500's, from each day's average IC: "
             f"{fmt(size['diff'])} [{fmt(size['lo'])}, {fmt(size['hi'])}] at {e.level:.0%} over {size['days']} days.", "",
             f"Mid caps: {mid[6]['rows']:,} stock-moments at the 30-minute hold; "
             f"{len(shared)} stock(s) on both lists ({', '.join(shared) or 'none'}) kept in the S&P 500 sample only, so the "
             "two samples share no stocks. The signal is a stock's move relative to the "
             f"market over the last {e.lookback} bars, reversed, at bars {', '.join(map(str, e.offsets))}; **the same moments for "
             "every hold**. IC is the rank correlation across stocks; book is the top and bottom fifth by the signal. A "
             "second sample of stocks over the same years, not of time.", "",
             "## The three claims, on mid caps", "",
             f"| claim | S&P 500 (E10) | mid caps now |", "|---|---|---|",
             f"| a. 30-minute IC with t >= {e.t_bar:g} | +0.0063, t 4.2 | {fmt(mid[6]['ic'].get('ic'))}, t {fmt(mid[6]['ic'].get('t'), '+.1f')} |",
             f"| b. 1 hour minus 30 minutes, IC [{e.level:.0%}] | +0.0035 [+0.0013, +0.0062] | "
             f"{fmt(hour_vs_30['diff'])} [{fmt(hour_vs_30['lo'])}, {fmt(hour_vs_30['hi'])}] |",
             f"| c. a full day minus 4 hours, IC [{e.level:.0%}] | −0.0047 [−0.0087, −0.0005] | "
             f"{fmt(day_vs_4h['diff'])} [{fmt(day_vs_4h['lo'])}, {fmt(day_vs_4h['hi'])}] |", "",
             "## Every hold, both samples", "",
             "| hold | mid caps: IC [95%] | t | gross bps | S&P 500 (this run): IC [95%] | t | gross bps |",
             "|---|---|---:|---:|---|---:|---:|"]
    for h in list(e.holds) + ["day"]:
        a, b = mid[h], big[h]
        lines.append(f"| {NAME[h]} | {fmt(a['ic'].get('ic'))} [{fmt(a['ic'].get('ic_lo'))}, {fmt(a['ic'].get('ic_hi'))}] | "
                     f"{fmt(a['ic'].get('t'), '+.1f')} | {fmt(a['book'].get('gross'), '+.2f')} | "
                     f"{fmt(b['ic'].get('ic'))} [{fmt(b['ic'].get('ic_lo'))}, {fmt(b['ic'].get('ic_hi'))}] | "
                     f"{fmt(b['ic'].get('t'), '+.1f')} | {fmt(b['book'].get('gross'), '+.2f')} |")
    lines += ["", "## Reported: mid caps minus S&P 500 at each hold (IC, day level, 99%)", "", "| hold | difference |", "|---|---|"]
    for h in list(e.holds) + ["day"]:
        d = paired_day_difference(mid[h]["days"], big[h]["days"], n_boot, level=e.level)
        lines.append(f"| {NAME[h]} | {fmt(d['diff'])} [{fmt(d['lo'])}, {fmt(d['hi'])}] |")
    lines += ["", f"- **Halves of the 30-minute difference** ({e.half_level:.0%}): before {e.split} {fmt(halves['early']['diff'])} "
              f"[{fmt(halves['early']['lo'])}, {fmt(halves['early']['hi'])}]; from it {fmt(halves['late']['diff'])} "
              f"[{fmt(halves['late']['lo'])}, {fmt(halves['late']['hi'])}]."]
    text = "\n".join(lines) + "\n"
    (out / "second_sample.md").write_text(text)
    pd.DataFrame([{"universe": u, "hold": str(h), "rows": r["rows"], **{f"ic_{k}": x for k, x in r["ic"].items()},
                   **{f"book_{k}": x for k, x in r["book"].items() if isinstance(x, (int, float))}}
                  for u, res in results.items() for h, r in res.items()]).to_csv(out / "second_sample.csv", index=False)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
