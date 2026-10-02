#!/usr/bin/env python3
"""Train the decision engine on saved walk-forward predictions and compare it to simple rules.

    python scripts/engine.py --config configs/etf_intraday.yaml
    python scripts/engine.py --config configs/sp500_multihour.yaml --archs logreg always_up
    python scripts/engine.py --config configs/etf_intraday.yaml --feedback full --signal resnet1d
    python scripts/engine.py --config configs/sp500_multihour.yaml --neutral

Needs predictions from scripts/sweep.py. Writes outputs/<experiment>/engine.md,
engine_folds.csv and engine_missed.csv (engine_neutral.* with --neutral). Learning
happens online, fold by fold, from earlier feedback only.

--neutral measures everything relative to the other symbols at the same date and bar.
On raw returns the market's quarter-to-quarter drift dominates and the engine learns
to chase it; relative to the cross-section only stock-specific signal is left. It
needs a wide universe (the S&P 500), not 12 ETFs that are mostly the market.

The reference is staying flat, which earns exactly 0: a policy that trades only
loses money to costs unless it has found something. The engine runs with and
without the entropy gate's verdict as input, and with two kinds of feedback:
bandit (only the reward of the action it took) and full (all three actions scored
on every past sample, so the trades it skipped teach it too). The best fixed rule
chosen from the past is the baseline to beat.

The missed-opportunities table groups samples by gate verdict and signal strength
and shows, per group, the best fixed action in hindsight next to what each engine
earned there.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import DEFAULT_CONFIG, output_dir

from src.config import load_config
from src.engine import (
    EngineConfig, baseline_positions, build_state, market_neutral, missed_opportunities, model_signal,
    run_engine, score_positions, threshold_rule,
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


def missed_lines(missed: pd.DataFrame, modes: list[str], gate_name: str, signal_name: str, gated: bool) -> list[str]:
    lines = ["", "## Missed opportunities: where did trading pay while the engine stayed out?", "",
             f"Scored samples grouped by the entropy gate's verdict ({gate_name}) and by signal strength: quintiles of "
             f"|confidence| of {signal_name}, 1 = weakest. In each group four fixed actions are scored on what actually "
             "happened: long, short, follow the signal, fade the signal. The best one, or staying flat if none paid after "
             "cost, is picked in hindsight for the whole group, never for a single sample. A group whose best action has an "
             "interval above zero (✓) is a real opportunity; where an engine earned clearly less, it missed it. With ten "
             "groups, one can clear that bar by luck."
             + (" The engines here are the ones given the gate's verdict." if gated else ""), "",
             "| gate | strength | samples | best fixed action | its net [95% CI] | "
             + " | ".join(f"{fb} engine: trades · net" for fb in modes) + " |",
             "|---|---:|---:|---|---|" + "---:|" * len(modes)]
    for _, r in missed.iterrows():
        if r["best_action"] == "stay flat":
            ci = "—"
        else:
            ci = f"{r['best_net']:+.2f} [{r['best_lo']:+.2f}, {r['best_hi']:+.2f}]" + (" ✓" if r["best_lo"] > 0 else "")
        cells = [f"{r[f'trades_{fb}']:.0%} · {r[f'net_{fb}']:+.2f}" for fb in modes]
        lines.append(f"| {r['gate']} | {r['strength']} | {r['samples']:,} | {r['best_action']} | {ci} | "
                     + " | ".join(cells) + " |")
    n = missed["samples"].sum()
    best = (missed["samples"] * missed["best_net"]).sum() / n
    earned = ", ".join(f"the {fb} engine {(missed['samples'] * missed[f'net_{fb}']).sum() / n:+.2f}" for fb in modes)
    lines += ["", f"Taking each group's best action earns {best:+.2f} bps per opportunity in hindsight; {earned}. "
              "The gap is what was left on the table, and part of it is luck in picking the best action after the fact."]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--archs", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--cost", type=float, default=None, help="round-trip cost in bps (default: the config's headline cost)")
    parser.add_argument("--feedback", choices=["bandit", "full", "both"], default="both",
                        help="bandit: only the taken action's reward; full: all three, skipped trades included")
    parser.add_argument("--neutral", action="store_true",
                        help="relative to the cross-section at each date and bar (long-short, needs a wide universe)")
    parser.add_argument("--signal", default=None,
                        help="model whose confidence grades situations in the missed-opportunities table "
                             "(default: the average over the trained models)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = output_dir(cfg)
    cost = args.cost if args.cost is not None else float(cfg.evaluation.headline_cost_bps)
    pred = load_predictions(out)
    if pred.empty:
        print(f"no predictions under {out / 'predictions'} — run scripts/sweep.py first")
        return 1

    state = build_state(pred, args.archs)
    if args.neutral:
        before = len(state)
        try:
            state = market_neutral(state)
        except ValueError as err:
            print(f"--neutral: {err}")
            return 1
        print(f"market-neutral: {before - len(state):,} of {before:,} samples dropped for lack of peers")
    tag = "_neutral" if args.neutral else ""
    models = [c[5:] for c in state.columns if c.startswith("conf_")]
    if args.signal is not None and args.signal not in models:
        print(f"--signal {args.signal}: no predictions for it; options: {models}")
        return 1
    statistic = getattr(cfg.screening, "gate_statistic", "trend")
    gate_flag = f"pass_{statistic}"
    gated = gate_flag in state.columns
    folds = np.sort(state["fold"].unique())
    scored_from = folds[1]
    modes = ["bandit", "full"] if args.feedback == "both" else [args.feedback]
    print(f"{cfg.experiment}: {len(state):,} samples, {len(folds)} folds, models {models}, cost {cost} bps, "
          f"gate features: {gated}, feedback: {modes}")

    def run(use_gate: bool, feedback: str):
        frames = []
        for seed in args.seeds:
            print(f"  engine seed {seed} gate={'on' if use_gate else 'off'} feedback={feedback}")
            frames.append(run_engine(state, EngineConfig(cost_bps=cost, seed=seed, feedback=feedback),
                                     use_gate=use_gate, log=lambda _: None))
        return frames

    runs = {(fb, g): run(g, fb) for fb in modes for g in ([False, True] if gated else [False])}

    live = state["fold"] >= scored_from
    rows = [{"policy": "stay flat (reference)", "net": 0.0, "lo": 0.0, "hi": 0.0, "trade_rate": 0.0}]
    rows.append(row("always long" + (" (stock vs. the market)" if args.neutral else ""),
                    with_pnl(state[live], np.ones(live.sum()), cost)))
    for name, pos in baseline_positions(state).items():
        if name != "always_up":
            rows.append(row(name.replace("follow_", "trade every ") + " signal", with_pnl(state[live], pos[live.to_numpy()], cost)))
    rule = threshold_rule(state, cost)
    rows.append(row("best fixed rule, chosen from the past", rule[rule["fold"] >= scored_from]))
    curves = {"rule": rule.groupby("fold")["pnl_bps"].mean()}
    for (fb, g), frames in runs.items():
        merged, rate = averaged_over_seeds([f[f["fold"] >= scored_from] for f in frames])
        rows.append(row(f"engine, {'models + entropy gate' if g else 'models only'}, {fb} feedback", merged, rate))
        curves[fb + (" + gate" if g else "")] = merged.groupby("fold")["pnl_bps"].mean()

    scored = state[live]
    engines = {fb: [f.loc[scored.index, "pos"].to_numpy() for f in runs[(fb, gated)]] for fb in modes}
    missed = missed_opportunities(scored, engines, cost, model_signal(scored, args.signal), gate_flag=gate_flag)

    table = pd.DataFrame(rows)
    lines = [f"# {cfg.experiment}: decision engine" + (", market-neutral" if args.neutral else ""), "",
             f"Net basis points per trading opportunity after a {cost} bps round-trip cost, folds {scored_from}+ "
             f"(fold {folds[0]} is the first feedback the engine sees). Flat opportunities count as zero. "
             f"Intervals are 95% day-block bootstrap; engine rows average {len(args.seeds)} training seeds. "
             "Bandit feedback is the reward of the action taken; full feedback scores all three actions on every past "
             "sample, including the trades the engine skipped."]
    if args.neutral:
        lines += ["", "Market-neutral: every return and every model confidence is measured relative to the average of "
                  "the other symbols at the same date and bar, so a long is a bet that this stock beats the rest, "
                  "paid on the stock leg only. Hedging the market leg would cost extra, so these results are "
                  "optimistic by that amount. \"Always long\" here is long every stock against the market, and a rule "
                  "that outputs the same signal for every symbol has no relative signal at all."]
    lines += ["",
             "| policy | net bps/opportunity [95% CI] | trades |", "|---|---|---:|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['policy']} | {r['net']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['trade_rate']:.0%} |")
    lines += missed_lines(missed, modes, statistic, args.signal or "the trained models' average", gated)
    curve = pd.DataFrame(curves).loc[lambda d: d.index >= scored_from]
    lines += ["", "## Is it learning from feedback?", "",
              "Net bps per opportunity by fold. A learner should drift up (or stay out) as feedback accumulates.", "",
              "| fold | " + " | ".join(curve.columns) + " |", "|---:|" + "---:|" * len(curve.columns)]
    for fold, r in curve.iterrows():
        lines.append(f"| {fold} | " + " | ".join(f"{v:+.2f}" for v in r) + " |")
    text = "\n".join(lines) + "\n"
    (out / f"engine{tag}.md").write_text(text)
    curve.to_csv(out / f"engine{tag}_folds.csv")
    missed.to_csv(out / f"engine{tag}_missed.csv", index=False)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
