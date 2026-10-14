#!/usr/bin/env python3
"""E7: does a stock recover a bigger share of a large lag than of a small one?

    python scripts/reversal_slope.py --config configs/sip_e7_slope.yaml

Reads the cached SIP bars (no training). Writes outputs/sip_e7_slope/slope.md and slope.csv.
The method is src/reversal.slope_by_group; the rule is in the config.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import enough_symbols, output_dir

from src.config import load_config
from src.data import load_universe, resolve_universe
from src.reversal import collect_states, slope_by_group, slope_verdict


def pct(x) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:+.2f}%"


def evaluate(frame: pd.DataFrame, e, n_boot: int) -> dict:
    kw = dict(min_peers=e.min_peers, n_boot=n_boot, level=e.level)
    split = pd.Timestamp(e.split)
    top = frame.assign(decile=(frame.groupby(["date", "bar_index"])["size"].rank(pct=True) > 0.9).astype(float))
    return {"overall": slope_by_group(frame, by=None, **kw), "size": slope_by_group(frame, e.by, e.groups, **kw),
            "fifths": slope_by_group(frame, e.by, 5, **kw), "decile": slope_by_group(top, "decile", 2, **kw),
            "vol": slope_by_group(frame, "vol", e.groups, **kw),
            "early": slope_by_group(frame[frame["date"] < split], e.by, e.groups, **kw),
            "late": slope_by_group(frame[frame["date"] >= split], e.by, e.groups, **kw),
            "rows": len(frame), "symbols": frame["symbol"].nunique()}


def rows(r: dict, labels: list[str]) -> list[str]:
    out = ["| group | typical move (bps) | share recovered [99%] |", "|---|---:|---|"]
    for i, label in enumerate(labels):
        out.append(f"| {label} | {r['typical_move_bps'][i]:.1f} | {pct(r['slope'][i])} [{pct(r['lo'][i])}, {pct(r['hi'][i])}] |")
    return out


def diff(r: dict) -> str:
    return f"{pct(r.get('diff'))} [{pct(r.get('diff_lo'))}, {pct(r.get('diff_hi'))}]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sip_e7_slope.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out, e = output_dir(cfg), cfg.slope
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

    v = slope_verdict(res["overall"], res["size"], res["early"], res["late"])
    o, s = res["overall"], res["size"]
    thirds = ["smallest third of moves", "middle third", "largest third"]
    lines = [f"# {cfg.experiment}: is a bigger share of a large lag recovered?", "",
             f"**Verdict: {v['level'].upper()}** — {v['reason']}.", "",
             f"{res['symbols']} symbols, {res['rows']:,} stock-moments. **Share recovered**: of the amount a stock lagged "
             f"the market over the last {e.lookback} bars, the part it makes back over the next {e.hold} (entry "
             f"{e.embargo} bar later). 99% day-block intervals.", "",
             f"**Every stock-moment:** {pct(o['slope'][0])} [{pct(o['lo'][0])}, {pct(o['hi'][0])}] of a typical "
             f"{o['typical_move_bps'][0]:.1f} bps move.", "", "## Primary: by size of the move", ""]
    lines += rows(s, thirds)
    lines += ["", f"- **Largest minus smallest third:** {diff(s)}. This data could detect a difference of about "
              f"{pct(s.get('detectable'))}.",
              f"- **Halves:** before {e.split} {diff(res['early'])}; from it {diff(res['late'])}.",
              "", "## Reported: the shape", "", "**By fifths of size:**", ""]
    lines += rows(res["fifths"], ["smallest fifth", "second", "third", "fourth", "largest fifth"])
    lines += ["", "**The largest tenth against the rest:**", ""] + rows(res["decile"], ["the other nine tenths", "largest tenth"])
    lines += ["", "**By thirds of volatility:**", ""] + rows(res["vol"], ["calmest third", "middle third", "most volatile third"])
    lines += [f"", f"Most volatile minus calmest: {diff(res['vol'])}.",
              "", f"## Reported: {s2.lookback}-bar lookback, {s2.hold}-bar hold", "",
              f"Every stock-moment: {pct(res2['overall']['slope'][0])} [{pct(res2['overall']['lo'][0])}, "
              f"{pct(res2['overall']['hi'][0])}].", ""]
    lines += rows(res2["size"], thirds) + ["", f"Largest minus smallest third: {diff(res2['size'])}."]
    text = "\n".join(lines) + "\n"
    (out / "slope.md").write_text(text)
    table = [{"geometry": g, "split": k, "group": i, "slope": r[k]["slope"][i], "lo": r[k]["lo"][i], "hi": r[k]["hi"][i],
              "typical_move_bps": r[k]["typical_move_bps"][i]}
             for g, r in (("primary", res), ("secondary", res2)) for k in ("overall", "size", "fifths", "decile", "vol")
             for i in range(len(r[k]["slope"]))]
    pd.DataFrame(table).to_csv(out / "slope.csv", index=False)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
