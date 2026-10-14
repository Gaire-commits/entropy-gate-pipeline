#!/usr/bin/env python3
"""E7 and E8: does a stock recover a bigger share of a large lag than of a small one?

    python scripts/reversal_slope.py --config configs/sip_e7_slope.yaml          # E7: the plain slope
    python scripts/reversal_slope.py --config configs/sip_e8_scaled_slope.yaml   # E8: scaled by volatility, winsorized

Reads the cached SIP bars (no training). Writes outputs/<experiment>/slope.md and slope.csv.
The method is src/reversal.slope_by_group; each rule is in its config.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import enough_symbols, output_dir

from src.config import load_config
from src.data import load_universe, resolve_universe
from src.reversal import collect_states, confirm_verdict, slope_by_group, slope_verdict


def pct(x) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:+.2f}%"


def evaluate(frame: pd.DataFrame, e, n_boot: int) -> dict:
    scale = getattr(e, "scale", None)
    kw = dict(min_peers=e.min_peers, n_boot=n_boot, level=e.level, scale=scale, winsor=getattr(e, "winsor", None))
    halves = dict(kw, level=getattr(e, "half_level", e.level))
    split = pd.Timestamp(e.split)
    frame = frame.assign(decile=(frame.groupby(["date", "bar_index"])["size"].rank(pct=True) > 0.9).astype(int),
                         unusual=frame["size"] / frame["vol"])
    return {"overall": slope_by_group(frame, by=None, **kw), "size": slope_by_group(frame, e.by, e.groups, **kw),
            "fifths": slope_by_group(frame, e.by, 5, **kw), "decile": slope_by_group(frame, "decile", 2, precut=True, **kw),
            "vol": slope_by_group(frame, "vol", e.groups, **kw), "unusual": slope_by_group(frame, "unusual", e.groups, **kw),
            "early": slope_by_group(frame[frame["date"] < split], e.by, e.groups, **halves),
            "late": slope_by_group(frame[frame["date"] >= split], e.by, e.groups, **halves),
            "rows": len(frame), "symbols": frame["symbol"].nunique()}


def rows(r: dict, labels: list[str], level: float) -> list[str]:
    out = [f"| group | stock-moments | average move (bps) | share recovered [{level:.0%}] |", "|---|---:|---:|---|"]
    for i, label in enumerate(labels):
        out.append(f"| {label} | {r['n'][i]:,} | {r['mean_move_bps'][i]:.1f} | {pct(r['slope'][i])} "
                   f"[{pct(r['lo'][i])}, {pct(r['hi'][i])}] |")
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
    scale, winsor, confirm = getattr(e, "scale", None), getattr(e, "winsor", None), bool(getattr(e, "confirm", False))
    half_level = getattr(e, "half_level", e.level)
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

    judge = confirm_verdict if confirm else slope_verdict
    v = judge(res["overall"], res["size"], res["early"], res["late"])
    o, s = res["overall"], res["size"]
    thirds = ["smallest third of moves", "middle third", "largest third"]
    how = "a slope through the origin, pooled by day"
    if scale:
        how += (f"; each move and its later return are first divided by the stock's own volatility (its last {e.window} "
                "five-minute returns), so every stock-moment carries the same noise")
    if winsor:
        how += f"; the later return is limited to its {winsor:.0%}–{1 - winsor:.0%} range within each moment"
    title = "does a sharper slope confirm that large moves reverse more?" if confirm else "is a bigger share of a large lag recovered?"
    lines = [f"# {cfg.experiment}: {title}", "",
             f"**Verdict: {v['level'].upper()}** — {v['reason']}.", "",
             f"{res['symbols']} symbols, {o['rows']:,} stock-moments used. **Share recovered**: of the amount a stock lagged "
             f"the market over the last {e.lookback} bars, the part it makes back over the next {e.hold} (entry "
             f"{e.embargo} bar later). Measured as {how}. Day-block intervals.", ""]
    if confirm:
        lines += ["Same bars and stock-moments as E7: this is a sharper instrument on the same data, not a second sample.", ""]
    lines += [f"**Every stock-moment:** {pct(o['slope'][0])} [{pct(o['lo'][0])}, {pct(o['hi'][0])}] of an average "
              f"{o['mean_move_bps'][0]:.1f} bps move.", "", "## Primary: by size of the move", ""]
    lines += rows(s, thirds, e.level)
    lines += ["", f"- **Largest minus smallest third:** {diff(s)} at {e.level:.0%}. This data could detect a difference of "
              f"about {pct(s.get('detectable'))}.",
              f"- **Halves** ({half_level:.0%}): before {e.split} {diff(res['early'])}; from it {diff(res['late'])}.",
              "", "## Reported: the shape", "", "**By fifths of size:**", ""]
    lines += rows(res["fifths"], ["smallest fifth", "second", "third", "fourth", "largest fifth"], e.level)
    lines += ["", "**The largest tenth against the rest:**", ""]
    lines += rows(res["decile"], ["the other nine tenths", "largest tenth"], e.level)
    lines += ["", "**By how unusual the move is for that stock** (size divided by the stock's volatility):", ""]
    lines += rows(res["unusual"], ["least unusual third", "middle third", "most unusual third"], e.level)
    lines += ["", f"Most minus least unusual: {diff(res['unusual'])}.", "", "**By thirds of volatility:**", ""]
    lines += rows(res["vol"], ["calmest third", "middle third", "most volatile third"], e.level)
    lines += ["", f"Most volatile minus calmest: {diff(res['vol'])}.",
              "", f"## Reported: {s2.lookback}-bar lookback, {s2.hold}-bar hold", "",
              f"Every stock-moment: {pct(res2['overall']['slope'][0])} [{pct(res2['overall']['lo'][0])}, "
              f"{pct(res2['overall']['hi'][0])}].", ""]
    lines += rows(res2["size"], thirds, e.level) + ["", f"Largest minus smallest third: {diff(res2['size'])}."]
    text = "\n".join(lines) + "\n"
    (out / "slope.md").write_text(text)
    table = [{"geometry": g, "split": k, "group": i, "n": r[k]["n"][i], "slope": r[k]["slope"][i], "lo": r[k]["lo"][i],
              "hi": r[k]["hi"][i], "mean_move_bps": r[k]["mean_move_bps"][i]}
             for g, r in (("primary", res), ("secondary", res2))
             for k in ("overall", "size", "fifths", "decile", "unusual", "vol") for i in range(len(r[k]["slope"]))]
    pd.DataFrame(table).to_csv(out / "slope.csv", index=False)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
