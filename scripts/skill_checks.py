#!/usr/bin/env python3
"""Before believing a cross-sectional IC: is it timing, or a fixed tilt? Does it hold over time?

    python scripts/skill_checks.py --config configs/ndx_holdout.yaml
    python scripts/skill_checks.py --config configs/ndx_replication.yaml --only universe/sp500_ex_nasdaq100.csv

Reads the saved predictions (no retraining). Writes outputs/<experiment>/skill_checks.md (with --only:
skill_checks_<list>.md), restricted to the symbols in --only when given.

1. Timing or tilt (src/crosssection.skill_decomposition): a model that always prefers the same stocks
   looks skilled on today's index members, which got there by rising (survivorship). Its IC then lives in
   the 'static' row and vanishes once each stock's own average return is removed. Skill at timing
   survives that.
2. Quarter by quarter: how many of the test quarters had a positive IC, the sign-test p-value, and the
   largest quarter's share, so one good quarter cannot pass for a steady effect.
3. By time of day: the IC for each of the day's signal times.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import DEFAULT_CONFIG, output_dir

from src.baselines import is_rule
from src.config import load_config
from src.crosssection import consistency, ic_by_period, skill_decomposition, with_sector
from src.data import resolve_universe
from src.experiment import ensemble, load_predictions


def fmt(x, spec="+.4f") -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="*", default=None, help="default: every trained model")
    parser.add_argument("--only", default=None, help="universe CSV (or symbols) to restrict the scored symbols to")
    parser.add_argument("--min-peers", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = output_dir(cfg)
    n_boot = int(getattr(cfg.evaluation, "bootstrap", 2000))
    pred = load_predictions(out)
    if pred.empty:
        print(f"no predictions under {out / 'predictions'} — run scripts/sweep.py first")
        return 1
    tag = ""
    if args.only:
        keep = set(resolve_universe(args.only))
        pred = pred[pred["symbol"].isin(keep)]
        tag = "_" + Path(args.only).stem
    models = [m for m in (args.models or sorted(pred["arch"].unique())) if not is_rule(m) and (pred["arch"] == m).any()]
    if not models:
        print("no trained model's predictions to check")
        return 1
    print(f"{cfg.experiment}{tag}: {pred['symbol'].nunique()} symbols, models {models}")

    lines = [f"# {cfg.experiment}: is the skill real?" + (f" ({Path(args.only).stem})" if args.only else ""), "",
             f"{pred['symbol'].nunique()} symbols; moments with fewer than {args.min_peers} are left out. ICs are rank "
             "correlations across stocks at each moment, averaged over days, with 95% day-block intervals.", "",
             "## 1. Timing, or a fixed preference for some stocks?", "",
             "**static**: each stock ranked by the model's average score for it in earlier quarters only (one fixed "
             "ranking). **timing**: the score minus that average, against returns minus each stock's own average return. "
             "The middle column removes each stock's average return from the plain model. A fixed tilt, which a list of "
             "today's index members can reward through survivorship, shows up in *static* and vanishes in the last two "
             "columns; skill at timing survives them. The first quarter has no earlier average and is left out of all four.",
             "", "| model | model IC [95% CI] · t | returns net of stock averages | static | timing |", "|---|---|---|---|---|"]
    by_quarter, by_slot = {}, {}
    for m in models:
        frame = ensemble(pred[pred["arch"] == m])
        d = skill_decomposition(frame, min_peers=args.min_peers, n_boot=n_boot)
        cells = [f"{fmt(r.get('ic'))} [{fmt(r.get('ic_lo'))}, {fmt(r.get('ic_hi'))}] · {fmt(r.get('t'), '+.1f')}"
                 for r in (d["model"], d["model, returns net of stock averages"], d["static"], d["timing"])]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
        by_quarter[m] = ic_by_period(frame, "fold", min_peers=args.min_peers)
        by_slot[m] = ic_by_period(frame, "bar_index", min_peers=args.min_peers)
        print(f"  {m}: model {fmt(d['model'].get('ic'))}, static {fmt(d['static'].get('ic'))}, timing {fmt(d['timing'].get('ic'))}")

    lines += ["", "## 2. Quarter by quarter", "",
              "Mean IC per test quarter. **Sign test**: the chance of at least this many positive quarters if the true IC "
              "were zero. **Largest share**: the best quarter's share of the summed positive IC (near 1 means one "
              "quarter carries everything).", ""]
    folds = sorted(set().union(*(set(q["fold"]) for q in by_quarter.values() if len(q))))
    lines += ["| quarter | test days | " + " | ".join(models) + " |", "|---|---|" + "---:|" * len(models)]
    for fold in folds:
        ref = next(q for q in by_quarter.values() if len(q) and fold in set(q["fold"]))
        row = ref[ref["fold"] == fold].iloc[0]
        cells = []
        for m in models:
            q = by_quarter[m]
            hit = q[q["fold"] == fold]
            cells.append(fmt(float(hit["ic"].iloc[0])) if len(hit) else "—")
        lines.append(f"| {fold} | {str(row['start'])[:10]} – {str(row['end'])[:10]} | " + " | ".join(cells) + " |")
    summary = {m: consistency(q) for m, q in by_quarter.items() if len(q)}
    lines.append("| **positive quarters** | | " + " | ".join(
        f"**{summary[m]['positive']}/{summary[m]['periods']}**" if m in summary else "—" for m in models) + " |")
    lines.append("| sign test p | | " + " | ".join(fmt(summary[m]['sign_p'], '.3f') if m in summary else "—" for m in models) + " |")
    lines.append("| largest share | | " + " | ".join(fmt(summary[m]['largest_share'], '.0%') if m in summary else "—" for m in models) + " |")

    lines += ["", "## 3. By time of day", "", "Mean IC [t] for each of the day's signal times (bar 11 = 10:25, the first "
              "signal of the session).", "", "| signal bar | " + " | ".join(models) + " |", "|---|" + "---:|" * len(models)]
    bars = sorted(set().union(*(set(s["bar_index"]) for s in by_slot.values() if len(s))))
    for bar in bars:
        cells = []
        for m in models:
            s = by_slot[m]
            hit = s[s["bar_index"] == bar]
            cells.append(f"{fmt(float(hit['ic'].iloc[0]))} [{fmt(float(hit['t'].iloc[0]), '+.1f')}]" if len(hit) else "—")
        lines.append(f"| {bar} | " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    (out / f"skill_checks{tag}.md").write_text(text)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
