#!/usr/bin/env python3
"""How long must the forward test run? (LIVE_PLAN.md, step 1)

    python scripts/forward_power.py --config configs/sip_e11_midcap.yaml

A design calculation on the cached SIP bars of both universes (the S&P 500 and the S&P MidCap 400; the
config names both). Not a test, so no rule: it fixes the forward test's length and geometry before
any forward day exists. Writes outputs/forward_power/power.md and days.csv (each day's IC per universe
and geometry, which the forward report also uses as its historical band). Method: src/forward_power.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import enough_symbols

from src.config import load_config
from src.data import load_universe, resolve_universe
from src.forward_power import combine, days_needed, window_pass_rates
from src.horizons import day_ic
from src.leadlag import collect

LENGTHS = (63, 126, 189, 252, 378, 504, 756)
GEOMETRIES = {"30-minute hold, a moment every 30 minutes (E5)": (6, None),
              "1-hour hold, a moment every 30 minutes": (12, None),
              "1-hour hold, 4 moments a day (E10, E11)": (12, (6, 12, 18, 24))}


def months(days: float) -> str:
    return "never" if not np.isfinite(days) else f"{days:,.0f} days (about {days / 21:.0f} months)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sip_e11_midcap.yaml")
    parser.add_argument("--min-peers", type=int, default=20)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path("outputs") / "forward_power"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    days: dict[tuple[str, str], pd.Series] = {}
    shared = set(resolve_universe(cfg.data.universe)) & set(resolve_universe(cfg.second.compare_universe))
    for label, spec in (("S&P 500", cfg.second.compare_universe), ("mid caps", cfg.data.universe)):
        symbols = resolve_universe(spec)
        bars = load_universe(cfg.data.cache_dir, symbols, cfg.data.timeframe)
        if not bars:
            print(f"no cached bars for {label} under {cfg.data.cache_dir} — fetch them first")
            return 1
        if not enough_symbols(bars, symbols, cfg.data.cache_dir):
            return 1
        if label == "mid caps":
            bars = {k: v for k, v in bars.items() if k not in shared}
        table = pd.read_csv(spec)
        sectors = dict(zip(table["symbol"], table["sector"]))
        print(f"{label}: {len(bars)} symbols ({time.time() - t0:.0f}s)")
        for hold in sorted({h for h, _ in GEOMETRIES.values()}):
            f = collect(bars, sectors, lookback=6, hold=hold, embargo=1, stride=6, min_sector=1).assign(prob=lambda x: x["own"])
            for name, (h, offsets) in GEOMETRIES.items():
                if h == hold:
                    g = f if offsets is None else f[f["bar_index"].isin(list(offsets))]
                    days[(label, name)] = day_ic(g, min_peers=args.min_peers)
            print(f"  hold {hold}: {len(f):,} stock-moments ({time.time() - t0:.0f}s)")
            del f
        del bars
    for name in GEOMETRIES:
        days[("both, averaged", name)] = combine(days[("S&P 500", name)], days[("mid caps", name)])

    lines = ["# How long must the forward test run?", "",
             "A design calculation on 2016–2026, not a test. **Days needed**: trading days for a one-sided 5% test of "
             "mean daily IC > 0 to succeed 80% of the time. **Historical windows**: every stretch of that many "
             "consecutive days in the history (starting every 5 days) treated as the forward test; *above zero* is "
             "the share with mean IC above zero at one-sided 5%, *full rule* the share that is also at least half the "
             "history's IC. \"Both, averaged\" is the mean of the two universes' daily ICs, each ranked within itself.",
             ""]
    rows = []
    for name in GEOMETRIES:
        lines += [f"## {name}", "", "| universe | IC | per-day sd | days needed | if half as strong |", "|---|---:|---:|---|---|"]
        for label in ("S&P 500", "mid caps", "both, averaged"):
            d = days[(label, name)].dropna()
            ic, sd = float(d.mean()), float(d.std(ddof=1))
            lines.append(f"| {label} | {ic:+.4f} | {sd:.4f} | {months(days_needed(ic, sd))} | "
                         f"{months(days_needed(ic / 2, sd))} |")
            for _, r in window_pass_rates(d, LENGTHS).iterrows():
                rows.append({"geometry": name, "universe": label, **r.to_dict()})
        lines += ["", "**Historical windows** (both, averaged):", "",
                  "| forward days | windows | above zero | full rule |", "|---:|---:|---:|---:|"]
        for r in rows:
            if r["geometry"] == name and r["universe"] == "both, averaged":
                lines.append(f"| {int(r['days'])} (about {r['days'] / 21:.0f} months) | {int(r['windows'])} | "
                             f"{r['share_above']:.0%} | {r['share_rule']:.0%} |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    (out / "power.md").write_text(text)
    pd.DataFrame(rows).to_csv(out / "windows.csv", index=False)
    pd.concat([s.rename("ic").rename_axis("date").reset_index().assign(universe=u, geometry=g)
               for (u, g), s in days.items()]).to_csv(out / "days.csv", index=False)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
