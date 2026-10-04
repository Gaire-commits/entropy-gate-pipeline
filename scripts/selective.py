#!/usr/bin/env python3
"""The main case: trade only each model's strongest signals, across the whole universe.

    python scripts/selective.py --config configs/sp500_multihour.yaml
    python scripts/selective.py --config configs/etf_intraday.yaml --models resnet1d gbm_ent

Reads the saved predictions (no retraining) and the cached bars, for each trade's trailing
volatility. Writes outputs/<experiment>/selective.md and selective.csv.

Breadth is what makes a selective strategy trade often: the top 10% of signals is about one trade
a day on 12 ETFs and about 60 a day on the S&P 500. Each quarter's cutoff comes from earlier
quarters only. Read the risk-scaled column before the bps: see src/selective.py.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import DEFAULT_CONFIG, output_dir

from src.baselines import is_rule
from src.config import load_config
from src.data import load_universe, resolve_universe
from src.experiment import ensemble, load_predictions
from src.selective import SHARES, selective_table, trailing_volatility


def fmt(x, spec="+.2f") -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def model_lines(model: str, table: pd.DataFrame, cost: float) -> list[str]:
    lines = [f"## {model}", "",
             "| kept | trades/day | days traded | long | hit rate | gross bps/trade [95% CI] | net/trade | "
             "net bps/day | quarters net > 0 | risk-scaled [95% CI] | volatility vs. all |",
             "|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|"]
    for _, r in table.iterrows():
        label = f"top {r['share']:.0%}" if r["share"] < 1 else "every signal"
        if r["selector"] != "confidence":
            label += " by volatility (placebo)"
        if not r.get("trades"):
            lines.append(f"| {label} | 0 | | | | | | | | | |")
            continue
        lines.append(
            f"| {label} ({r['kept']:.0%}) | {r['trades_per_day']:.1f} | {r['days_traded']:.0%} | {r['long_share']:.0%} "
            f"| {r['hit_rate']:.3f} | {fmt(r['gross'])} [{fmt(r['gross_lo'])}, {fmt(r['gross_hi'])}] "
            f"| {fmt(r['net'])} | {fmt(r['net_per_day'], '+.1f')} | {r['folds_positive']}/{r['folds']} "
            f"| {fmt(r.get('risk_scaled'), '+.1f')} [{fmt(r.get('risk_lo'), '+.1f')}, {fmt(r.get('risk_hi'), '+.1f')}] "
            f"| {fmt(r.get('vol_ratio'), '.2f')}x |")
    return lines + [""]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="*", default=None, help="default: every trained model (rules have no ranking)")
    parser.add_argument("--shares", nargs="*", type=float, default=list(SHARES))
    parser.add_argument("--cost", type=float, default=None, help="round-trip cost in bps (default: the config's headline cost)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = output_dir(cfg)
    cost = args.cost if args.cost is not None else float(cfg.evaluation.headline_cost_bps)
    pred = load_predictions(out)
    if pred.empty:
        print(f"no predictions under {out / 'predictions'} — run scripts/sweep.py first")
        return 1
    models = args.models or sorted(a for a in pred["arch"].unique() if not is_rule(a))
    frames = {m: ensemble(pred[pred["arch"] == m]) for m in models if (pred["arch"] == m).any()}
    if not frames:
        print(f"none of {models} has predictions; options: {sorted(pred['arch'].unique())}")
        return 1

    keys = pd.concat([f[["symbol", "date", "bar_index"]] for f in frames.values()]).drop_duplicates(ignore_index=True)
    bars = load_universe(cfg.data.cache_dir, resolve_universe(cfg.data.universe), cfg.data.timeframe)
    if bars:
        keys["vol"] = trailing_volatility(bars, keys).to_numpy()
        print(f"trailing volatility found for {keys['vol'].notna().mean():.1%} of {len(keys):,} signal bars")
    else:
        print(f"no cached bars in {cfg.data.cache_dir}: skipping the risk-scaled and placebo checks")

    horizon = int(cfg.features.horizon)
    tables = []
    for model, frame in frames.items():
        if "vol" in keys.columns:
            frame = frame.merge(keys, on=["symbol", "date", "bar_index"], how="left")
        print(f"  {model}: {len(frame):,} predictions")
        t = selective_table(frame, cost=cost, horizon=horizon, shares=args.shares,
                            n_boot=int(getattr(cfg.evaluation, "bootstrap", 2000)))
        tables.append(t.assign(model=model))
    table = pd.concat(tables, ignore_index=True)

    lines = [f"# {cfg.experiment}: selective trading", "",
             f"Trade only each model's strongest signals. Each quarter's cutoff comes from earlier quarters only, and "
             f"quarters without enough history to set one are left out at every level, so the rows compare. Net is after "
             f"a {cost} bps round trip; net bps/day sums every trade of a day over all days, traded or not. Intervals are "
             "95% day-block bootstrap.", "",
             "**Skill or volatility?** Risk-scaled is each trade's return divided by its symbol's trailing volatility "
             f"times the square root of the {horizon}-bar hold, x100. Skill raises it as the filter tightens and beats the "
             "volatility placebo (the same share of signals ranked by trailing volatility instead of confidence, trading "
             "the model's direction). If only the bps rise, the strongest signals are just the most volatile moments. "
             "Compare the placebo on risk-scaled, not bps: a model with a little skill everywhere makes the most bps "
             "where moves are biggest.", ""]
    for model in frames:
        lines += model_lines(model, table[table["model"] == model], cost)
    text = "\n".join(lines)
    (out / "selective.md").write_text(text)
    table.to_csv(out / "selective.csv", index=False)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
