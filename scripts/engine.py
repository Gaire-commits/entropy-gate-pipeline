#!/usr/bin/env python3
"""Train the decision engine on saved walk-forward predictions and compare it to simple rules.

    python scripts/engine.py --config configs/etf_intraday.yaml
    python scripts/engine.py --config configs/sp500_multihour.yaml --archs logreg always_up

Needs predictions from scripts/sweep.py. Writes outputs/<experiment>/engine.md and
engine_folds.csv. Learning happens online, fold by fold, from earlier feedback only.

The reference is staying flat, which earns exactly 0: a policy that trades only
loses money to costs unless it has found something. Three rows answer the research
question directly: the engine given the model outputs only, the engine also given
the entropy gate's verdict, and the best fixed rule chosen from the past.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import DEFAULT_CONFIG, output_dir

from src.config import load_config
from src.engine import (
    EngineConfig, baseline_positions, build_state, run_engine, score_positions, threshold_rule,
)
from src.experiment import load_predictions


def with_pnl(rows: pd.DataFrame, pos: np.ndarray, cost: float) -> pd.DataFrame:
    out = rows[["fold", "date", "symbol", "ret_bps", "y"]].copy()
    out["pos"] = pos
    out["pnl_bps"] = pos * out["ret_bps"].to_numpy() - cost * np.abs(pos)
    return out


def averaged_over_seeds(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, float]:
    """Same rows every seed, so average the P&L per row; trade rate is averaged separately."""
    base = frames[0].copy()
    base["pnl_bps"] = np.mean([f["pnl_bps"].to_numpy() for f in frames], axis=0)
    rate = float(np.mean([(f["pos"] != 0).mean() for f in frames]))
    base["pos"] = 1.0
    return base, rate


def row(name: str, frame: pd.DataFrame, rate: float | None = None, n_boot: int = 2000) -> dict:
    s = score_positions(frame, n_boot=n_boot)
    return {"policy": name, "net": s["net_bps"], "lo": s["net_lo"], "hi": s["net_hi"],
            "trade_rate": s["trade_rate"] if rate is None else rate}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--archs", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--cost", type=float, default=None, help="round-trip cost in bps (default: the config's headline cost)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = output_dir(cfg)
    cost = args.cost if args.cost is not None else float(cfg.evaluation.headline_cost_bps)
    pred = load_predictions(out)
    if pred.empty:
        print(f"no predictions under {out / 'predictions'} — run scripts/sweep.py first")
        return 1

    state = build_state(pred, args.archs)
    gated = "pass_trend" in state.columns
    folds = np.sort(state["fold"].unique())
    scored_from = folds[1]
    print(f"{cfg.experiment}: {len(state):,} samples, {len(folds)} folds, models "
          f"{[c[5:] for c in state.columns if c.startswith('conf_')]}, cost {cost} bps, gate features: {gated}")

    def run(use_gate: bool):
        frames = []
        for seed in args.seeds:
            print(f"  engine seed {seed} gate={'on' if use_gate else 'off'}")
            frames.append(run_engine(state, EngineConfig(cost_bps=cost, seed=seed), use_gate=use_gate, log=lambda _: None))
        return frames

    runs = {"engine, models only": run(False)}
    if gated:
        runs["engine, models + entropy gate"] = run(True)

    live = state["fold"] >= scored_from
    rows = [{"policy": "stay flat (reference)", "net": 0.0, "lo": 0.0, "hi": 0.0, "trade_rate": 0.0}]
    rows.append(row("always long", with_pnl(state[live], np.ones(live.sum()), cost)))
    for name, pos in baseline_positions(state).items():
        if name != "always_up":
            rows.append(row(name.replace("follow_", "trade every ") + " signal", with_pnl(state[live], pos[live.to_numpy()], cost)))
    rule = threshold_rule(state, cost)
    rows.append(row("best fixed rule, chosen from the past", rule[rule["fold"] >= scored_from]))
    curves = {"threshold rule": rule.groupby("fold")["pnl_bps"].mean()}
    for name, frames in runs.items():
        merged, rate = averaged_over_seeds([f[f["fold"] >= scored_from] for f in frames])
        rows.append(row(name, merged, rate))
        curves[name] = merged.groupby("fold")["pnl_bps"].mean()

    table = pd.DataFrame(rows)
    lines = [f"# {cfg.experiment}: decision engine", "",
             f"Net basis points per trading opportunity after a {cost} bps round-trip cost, folds {scored_from}+ "
             f"(fold {folds[0]} is the first feedback the engine sees). Flat opportunities count as zero. "
             f"Intervals are 95% day-block bootstrap; engine rows average {len(args.seeds)} training seeds.", "",
             "| policy | net bps/opportunity [95% CI] | trades |", "|---|---|---:|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['policy']} | {r['net']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['trade_rate']:.0%} |")
    curve = pd.DataFrame(curves).loc[lambda d: d.index >= scored_from]
    lines += ["", "## Is it learning from feedback?", "",
              "Net bps per opportunity by fold. A learner should drift up (or stay out) as feedback accumulates.", "",
              "| fold | " + " | ".join(curve.columns) + " |", "|---:|" + "---:|" * len(curve.columns)]
    for fold, r in curve.iterrows():
        lines.append(f"| {fold} | " + " | ".join(f"{v:+.2f}" for v in r) + " |")
    text = "\n".join(lines) + "\n"
    (out / "engine.md").write_text(text)
    curve.to_csv(out / "engine_folds.csv")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
