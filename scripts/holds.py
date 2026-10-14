#!/usr/bin/env python3
"""E10: does the reversal keep growing the longer the position is held?

    python scripts/holds.py --config configs/sip_e10_holds.yaml

Reads the cached SIP bars (no training). Writes outputs/sip_e10_holds/holds.md and holds.csv.
The signal is E5's; every hold uses the same moments. The method is src/horizons.py; the rule is in the config.
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
from src.horizons import collect_overnight, growth_verdict
from src.leadlag import collect

NAME = {6: "30 minutes", 12: "1 hour", 24: "2 hours", 48: "4 hours", "day": "a full day"}


def fmt(x, spec="+.4f") -> str:
    return "—" if x is None or not np.isfinite(x) else format(x, spec)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sip_e10_holds.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out, e = output_dir(cfg), cfg.holds
    n_boot = int(cfg.evaluation.bootstrap)
    universe = pd.read_csv(cfg.data.universe)
    sectors = dict(zip(universe["symbol"], universe["sector"]))
    t0 = time.time()
    bars = load_universe(cfg.data.cache_dir, resolve_universe(cfg.data.universe), cfg.data.timeframe)
    if not bars:
        print(f"no cached bars under {cfg.data.cache_dir} — fetch the SIP bars first (Colab section 21)")
        return 1
    if not enough_symbols(bars, resolve_universe(cfg.data.universe), cfg.data.cache_dir):
        return 1
    print(f"{len(bars)} symbols loaded ({time.time() - t0:.0f}s)")

    frames = {}
    for h in e.holds:
        f = collect(bars, sectors, lookback=e.lookback, hold=h, embargo=e.embargo, stride=e.lookback, min_sector=1)
        frames[h] = f[f["bar_index"].isin(list(e.offsets))].assign(prob=lambda x: x["own"]).reset_index(drop=True)
        print(f"  hold {NAME[h]}: {len(frames[h]):,} stock-moments ({time.time() - t0:.0f}s)")
    frames["day"] = collect_overnight(bars, e.lookback, e.embargo, tuple(e.offsets), e.min_peers).assign(
        prob=lambda x: x["own"]).reset_index(drop=True)
    print(f"  hold {NAME['day']}: {len(frames['day']):,} stock-moments ({time.time() - t0:.0f}s)")
    del bars

    rows, short = {}, min(e.holds)
    for h, f in frames.items():
        rows[h] = {"ic": cross_sectional_ic(f, min_peers=e.min_peers, n_boot=n_boot),
                   "book": score_book(quantile_book(f, q=e.q, min_peers=e.min_peers), e.cost_bps, n_boot)}
    split = pd.Timestamp(e.split)
    base = frames[short]
    verdicts, diffs = {}, {}
    for h in e.decisive:
        f = frames[h]
        kw = dict(min_peers=e.min_peers, n_boot=n_boot)
        paired = paired_ic_difference(f, base, level=e.level, **kw)
        early = paired_ic_difference(f[f["date"] < split], base[base["date"] < split], level=e.half_level, **kw)
        late = paired_ic_difference(f[f["date"] >= split], base[base["date"] >= split], level=e.half_level, **kw)
        diffs[h] = (paired, early, late)
        verdicts[h] = growth_verdict(rows[short]["ic"], rows[h]["ic"], paired, early, late, e.t_bar)

    base_gross = rows[short]["book"].get("gross", np.nan)
    lines = [f"# {cfg.experiment}: does the reversal keep growing the longer it is held?", ""]
    for h in e.decisive:
        lines.append(f"- **{NAME[h].capitalize()} against {NAME[short]}: {verdicts[h]['level'].upper()}** — {verdicts[h]['reason']}.")
    lines += ["", f"The same signal as E5 (a stock's move relative to the market over the last {e.lookback} bars, reversed), at "
              f"signal bars {', '.join(str(o) for o in e.offsets)} (10:00 to 11:30), entry {e.embargo} bar later. **Every hold "
              "uses the same moments**, so only the time held changes. The IC (skill per unit of risk) carries the verdicts; "
              "the amount in bps grows with the horizon on its own. Book: top and bottom fifth by the signal.", "",
              "## Every hold", "",
              "| hold | stock-moments | IC [95%] | t | book gross bps [95%] | net after 2 bps | expected from the 30-minute book "
              "by size of moves alone |", "|---|---:|---|---:|---|---:|---:|"]
    for h in list(e.holds) + ["day"]:
        c, bk = rows[h]["ic"], rows[h]["book"]
        bars_held = 78 if h == "day" else h
        lines.append(f"| {NAME[h]} | {len(frames[h]):,} | {fmt(c.get('ic'))} [{fmt(c.get('ic_lo'))}, {fmt(c.get('ic_hi'))}] | "
                     f"{fmt(c.get('t'), '+.1f')} | {fmt(bk.get('gross'), '+.2f')} [{fmt(bk.get('gross_lo'), '+.2f')}, "
                     f"{fmt(bk.get('gross_hi'), '+.2f')}] | {fmt(bk.get('net'), '+.2f')} | "
                     f"{fmt(base_gross * np.sqrt(bars_held / short), '+.2f')} |")
    lines += ["", "## The decisive differences (paired on the same moments)", "",
              f"| comparison | IC difference [{e.level:.0%}] | before {e.split} [{e.half_level:.0%}] | from it [{e.half_level:.0%}] |",
              "|---|---|---|---|"]
    for h, (p, a, b) in diffs.items():
        lines.append(f"| {NAME[h]} minus {NAME[short]} | {fmt(p['diff'])} [{fmt(p['lo'])}, {fmt(p['hi'])}] | "
                     f"{fmt(a['diff'])} [{fmt(a['lo'])}, {fmt(a['hi'])}] | {fmt(b['diff'])} [{fmt(b['lo'])}, {fmt(b['hi'])}] |")
    lines += ["", "## Reported: each step to the next hold (99%)", "", "| step | IC difference |", "|---|---|"]
    order = list(e.holds) + ["day"]
    for a, b in zip(order, order[1:]):
        d = paired_ic_difference(frames[b], frames[a], min_peers=e.min_peers, n_boot=n_boot, level=e.level)
        lines.append(f"| {NAME[b]} minus {NAME[a]} | {fmt(d['diff'])} [{fmt(d['lo'])}, {fmt(d['hi'])}] |")
    lines += ["", "## Reported: each signal time", "", "| hold | " + " | ".join(f"bar {o}" for o in e.offsets) + " |",
              "|---|" + "---:|" * len(e.offsets)]
    for h in order:
        cells = []
        for o in e.offsets:
            r = cross_sectional_ic(frames[h][frames[h]["bar_index"] == o], min_peers=e.min_peers, n_boot=200)
            cells.append(f"{fmt(r.get('ic'))} (t {fmt(r.get('t'), '+.1f')})")
        lines.append(f"| {NAME[h]} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (out / "holds.md").write_text(text)
    pd.DataFrame([{"hold": str(h), "rows": len(frames[h]), **{f"ic_{k}": v for k, v in r["ic"].items()},
                   **{f"book_{k}": v for k, v in r["book"].items() if isinstance(v, (int, float))}}
                  for h, r in rows.items()]).to_csv(out / "holds.csv", index=False)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
