#!/usr/bin/env python3
"""E9: how does the reversal change with the horizon, from 30 minutes to 2 hours?

    python scripts/horizons.py --config configs/sip_e9_horizons.yaml
    python scripts/horizons.py --config configs/sip_e9_horizons.yaml --diagonal-only    # three cells, faster

Reads the cached SIP bars (no training). Writes outputs/sip_e9_horizons/horizons.md and horizons.csv.
The method is src/horizons.py; the rule is in the config.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import enough_symbols, output_dir

from src.config import load_config
from src.crosssection import cross_sectional_ic, quantile_book, score_book
from src.data import load_universe, resolve_universe
from src.horizons import day_ic, paired_day_difference, verdict
from src.leadlag import collect

MIN = {6: "30 min", 12: "1 hour", 24: "2 hours"}


def fmt(x, spec="+.4f") -> str:
    return "—" if x is None or not np.isfinite(x) else format(x, spec)


def evaluate(frame: pd.DataFrame, e, n_boot: int) -> dict:
    f = frame.assign(prob=frame["own"])
    ic = cross_sectional_ic(f, min_peers=e.min_peers, n_boot=n_boot)
    book = score_book(quantile_book(f, q=e.q, min_peers=e.min_peers), e.cost_bps, n_boot)
    days = day_ic(f, min_peers=e.min_peers)
    return {"ic": ic, "book": book, "days": days, "rows": len(frame),
            "per_day": float(frame.groupby("date")["bar_index"].nunique().mean())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sip_e9_horizons.yaml")
    parser.add_argument("--diagonal-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out, e = output_dir(cfg), cfg.horizons
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

    cells = [(h, h) for h in e.diagonal]
    if not args.diagonal_only:
        cells += [(lb, h) for lb in e.grid for h in e.grid if lb != h]
    res = {}
    for lb, h in cells:
        frame = collect(bars, sectors, lookback=lb, hold=h, embargo=e.embargo, stride=h, min_sector=e.min_sector)
        res[(lb, h)] = evaluate(frame, e, n_boot)
        r = res[(lb, h)]
        print(f"  lookback {lb:2d} hold {h:2d}: {r['rows']:,} stock-moments, IC {fmt(r['ic'].get('ic'))} "
              f"(t {fmt(r['ic'].get('t'), '+.1f')}) ({time.time() - t0:.0f}s)")
    del bars

    short, long_ = min(e.diagonal), max(e.diagonal)
    a, b = res[(long_, long_)]["days"], res[(short, short)]["days"]
    split = pd.Timestamp(e.split)
    diff = paired_day_difference(a, b, n_boot, level=e.level)
    early = paired_day_difference(a[a.index < split], b[b.index < split], n_boot, level=0.95)
    late = paired_day_difference(a[a.index >= split], b[b.index >= split], n_boot, level=0.95)
    v = verdict({h: res[(h, h)]["ic"] for h in e.diagonal}, diff, early, late, e.t_bar)

    lines = [f"# {cfg.experiment}: how does the reversal change from {MIN[short]} to {MIN[long_]}?", "",
             f"**Verdict: {v['level'].upper()}** — {v['reason']}.", "",
             "A stock's move relative to the market over the last *lookback* bars, reversed; entry one bar later; held "
             "for *hold* bars. IC is the rank correlation across stocks at each moment. **Skill per unit of risk (IC) is "
             "the quantity compared across horizons**: the amount in bps grows with the horizon on its own, as moves are "
             "bigger. Book: long the top and short the bottom fifth by the signal.", "",
             "## Diagonal (lookback = hold): the cells that carry the verdict", "",
             "| horizon | moments a day | IC [95%] | t | book gross bps [95%] | net after 2 bps | amount expected from "
             "the 30-minute book by size of moves alone |", "|---|---:|---|---:|---|---:|---:|"]
    base_gross = res[(short, short)]["book"].get("gross", np.nan)
    for h in e.diagonal:
        r = res[(h, h)]
        c, bk = r["ic"], r["book"]
        lines.append(f"| {MIN[h]} | {r['per_day']:.1f} | "
                     f"{fmt(c.get('ic'))} [{fmt(c.get('ic_lo'))}, {fmt(c.get('ic_hi'))}] | {fmt(c.get('t'), '+.1f')} | "
                     f"{fmt(bk.get('gross'), '+.2f')} [{fmt(bk.get('gross_lo'), '+.2f')}, {fmt(bk.get('gross_hi'), '+.2f')}] | "
                     f"{fmt(bk.get('net'), '+.2f')} | {fmt(base_gross * np.sqrt(h / short), '+.2f')} |")
    lines += ["", f"- **{MIN[long_]} minus {MIN[short]}, IC** (99%): {fmt(diff['diff'])} "
              f"[{fmt(diff['lo'])}, {fmt(diff['hi'])}] over {diff['days']} days.",
              f"- **Halves** (95%): before {e.split} {fmt(early['diff'])} [{fmt(early['lo'])}, {fmt(early['hi'])}]; "
              f"from it {fmt(late['diff'])} [{fmt(late['lo'])}, {fmt(late['hi'])}].",
              ]
    if (12, 12) in res and (6, 6) in res:
        mid = paired_day_difference(res[(12, 12)]["days"], res[(6, 6)]["days"], n_boot, level=e.level)
        lines.append(f"- **1 hour minus 30 minutes, IC** (99%, reported): {fmt(mid['diff'])} "
                     f"[{fmt(mid['lo'])}, {fmt(mid['hi'])}].")
    if not args.diagonal_only:
        lines += ["", "## Reported: every lookback and hold", "",
                  "| lookback \\ hold | " + " | ".join(MIN[h] for h in e.grid) + " |", "|---|" + "---:|" * len(e.grid)]
        for lb in e.grid:
            cells_txt = []
            for h in e.grid:
                r = res.get((lb, h))
                cells_txt.append("—" if r is None else f"IC {fmt(r['ic'].get('ic'))} (t {fmt(r['ic'].get('t'), '+.1f')}), "
                                 f"gross {fmt(r['book'].get('gross'), '+.2f')} bps")
            lines.append(f"| {MIN[lb]} | " + " | ".join(cells_txt) + " |")
    text = "\n".join(lines) + "\n"
    (out / "horizons.md").write_text(text)
    pd.DataFrame([{"lookback": lb, "hold": h, **{f"ic_{k}": v for k, v in r["ic"].items()},
                   **{f"book_{k}": v for k, v in r["book"].items() if isinstance(v, (int, float))}}
                  for (lb, h), r in res.items()]).to_csv(out / "horizons.csv", index=False)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
