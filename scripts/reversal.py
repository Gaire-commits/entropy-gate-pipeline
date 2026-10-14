#!/usr/bin/env python3
"""E6: does permutation entropy tell when the short-term reversal is stronger?

    python scripts/reversal.py --config configs/sip_e6_reversal.yaml

Reads the cached SIP bars (no training). Writes outputs/sip_e6_reversal/reversal.md and reversal.csv.
The method is in src/reversal.py; the rule is in the config.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import enough_symbols, output_dir

from src.config import load_config
from src.crosssection import cross_sectional_ic
from src.data import load_universe, resolve_universe
from src.reversal import STATES, collect_states, conditional_ic, verdict

NAMES = {"pe3": "permutation entropy (registered)", "trend3": "share of monotone patterns",
         "ac1": "lag-1 autocorrelation", "vol": "volatility", "size": "size of the move"}


def fmt(x, spec="+.4f") -> str:
    return "—" if x is None or not np.isfinite(x) else format(x, spec)


def evaluate(frame: pd.DataFrame, e, n_boot: int) -> dict:
    kw = dict(groups=e.groups, n_boot=n_boot, level=e.level)
    out = {"overall": cross_sectional_ic(frame.assign(prob=frame["signal"]), min_peers=e.min_peers, n_boot=n_boot),
           "single": {k: conditional_ic(frame, k, min_group=e.min_group, **kw) for k in STATES},
           "matched": {k: conditional_ic(frame, k, control=e.control, min_group=e.min_group_matched, **kw)
                       for k in STATES if k != e.control}}
    split = pd.Timestamp(e.split)
    out["early"] = conditional_ic(frame[frame["date"] < split], e.state, min_group=e.min_group, **kw)
    out["late"] = conditional_ic(frame[frame["date"] >= split], e.state, min_group=e.min_group, **kw)
    out["rows"], out["symbols"] = len(frame), frame["symbol"].nunique()
    out["with_state"] = float(frame[e.state].notna().mean())
    return out


def table(res: dict, e) -> list[str]:
    lines = ["| state | reversal IC: bottom third | middle | top third | top − bottom [99%] | t | "
             f"same, within thirds of {NAMES[e.control]} [99%] |", "|---|---:|---:|---:|---|---:|---|"]
    for k in STATES:
        s, m = res["single"][k], res["matched"].get(k)
        matched = "—" if m is None else f"{fmt(m.get('diff'))} [{fmt(m.get('lo'))}, {fmt(m.get('hi'))}]"
        lines.append(f"| {NAMES[k]} | {fmt(s.get('low'))} | {fmt(s.get('mid'))} | {fmt(s.get('high'))} | "
                     f"{fmt(s.get('diff'))} [{fmt(s.get('lo'))}, {fmt(s.get('hi'))}] | {fmt(s.get('t'), '+.1f')} | {matched} |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sip_e6_reversal.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out, e = output_dir(cfg), cfg.reversal
    n_boot = int(cfg.evaluation.bootstrap)
    universe = resolve_universe(cfg.data.universe)
    t0 = time.time()
    bars = load_universe(cfg.data.cache_dir, universe, cfg.data.timeframe)
    if not bars:
        print(f"no cached bars under {cfg.data.cache_dir} — fetch the SIP bars first (Colab section 21)")
        return 1
    if not enough_symbols(bars, universe, cfg.data.cache_dir):
        return 1
    print(f"{len(bars)} symbols loaded ({time.time() - t0:.0f}s)")
    frame = collect_states(bars, e.lookback, e.hold, e.embargo, e.stride, e.window)
    print(f"primary: {len(frame):,} stock-moments ({time.time() - t0:.0f}s)")
    res = evaluate(frame, e, n_boot)
    del frame
    s2 = e.secondary
    frame2 = collect_states(bars, s2.lookback, s2.hold, e.embargo, s2.stride, e.window)
    del bars
    res2 = evaluate(frame2, e, n_boot)
    del frame2

    v = verdict(res["single"][e.state], res["matched"][e.state], res["early"], res["late"])
    o = res["overall"]
    lines = [f"# {cfg.experiment}: does permutation entropy tell when the reversal is stronger?", "",
             f"**Verdict: {v['level'].upper()}** — {v['reason']}.", "",
             f"{res['symbols']} symbols, {res['rows']:,} stock-moments; {res['with_state']:.0%} have {e.window} returns of "
             f"history for the states. Reversal over the last {e.lookback} bars, entry {e.embargo} bar later, {e.hold}-bar "
             f"hold. **The reversal itself:** IC {fmt(o.get('ic'))} [{fmt(o.get('ic_lo'))}, {fmt(o.get('ic_hi'))}], "
             f"t {fmt(o.get('t'), '+.1f')} (E5 found +0.0064, t 7.5).", "",
             "At each moment stocks are split into thirds by the state; the reversal's rank IC is measured in each third. "
             "Differences are top third minus bottom third, with 99% day-block intervals.", "", "## Primary", ""]
    lines += table(res, e)
    lines += ["", f"- **Halves** ({NAMES[e.state]}, top − bottom): before {e.split} {fmt(res['early'].get('diff'))} "
              f"[{fmt(res['early'].get('lo'))}, {fmt(res['early'].get('hi'))}]; from it {fmt(res['late'].get('diff'))} "
              f"[{fmt(res['late'].get('lo'))}, {fmt(res['late'].get('hi'))}].",
              "", f"## Reported: {s2.lookback}-bar lookback, {s2.hold}-bar hold", "",
              f"The reversal itself: IC {fmt(res2['overall'].get('ic'))}, t {fmt(res2['overall'].get('t'), '+.1f')}.", ""]
    lines += table(res2, e)
    text = "\n".join(lines) + "\n"
    (out / "reversal.md").write_text(text)
    pd.DataFrame([{"geometry": g, "state": k, "sort": kind, **r[kind][k]} for g, r in (("primary", res), ("secondary", res2))
                  for kind in ("single", "matched") for k in r[kind]]).to_csv(out / "reversal.csv", index=False)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
