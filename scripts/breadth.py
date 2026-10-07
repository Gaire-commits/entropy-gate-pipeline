#!/usr/bin/env python3
"""How diversified are the bets, and what do the models know once the market is out of the way?

    python scripts/breadth.py --config configs/sp500_multihour.yaml
    python scripts/breadth.py --config configs/etf_intraday.yaml --models resnet1d gbm_ent

Reads the saved predictions (no retraining) and, for a universe with sectors, universe/*.csv. Writes
outputs/<experiment>/breadth.md and breadth.csv.

Everything before this report scored each trade against its own direction. Trades placed at the same
moment share the market move, and a model that is nearly always long makes one bet on the market however
many stocks it holds. This report measures that (effective independent bets per day), then scores the
models where the breadth of the universe actually counts: the rank correlation between score and return
across stocks at each moment (IC), and a long-short book that holds the model's top and bottom picks.
See src/crosssection.py for the definitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import DEFAULT_CONFIG, output_dir

from src.baselines import RULES
from src.config import load_config
from src.crosssection import (
    breakeven_ic, cross_sectional_ic, cross_sectional_sigma, directional_book, ensemble_scores, quantile_book,
    score_book, with_sector, wide_moments,
)
from src.experiment import ensemble, load_predictions
from src.holdout import resolve_roles
from src.selective import top_share_mask


def sector_map(universe) -> dict[str, str] | None:
    """symbol -> sector from a universe CSV with a `sector` column, else None."""
    if isinstance(universe, str) and Path(universe).exists():
        table = pd.read_csv(universe)
        if "sector" in table.columns:
            return dict(zip(table["symbol"], table["sector"]))
    return None


def f(x, spec="+.2f") -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def interval(r: dict, lo="gross_lo", hi="gross_hi", key="gross", spec="+.2f") -> str:
    return f"{f(r.get(key), spec)} [{f(r.get(lo), spec)}, {f(r.get(hi), spec)}]" if r.get(key) is not None else "—"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="*", default=None, help="default: every model with a cross-sectional score")
    parser.add_argument("--q", type=float, default=0.1, help="share of the cross-section held long and short")
    parser.add_argument("--n-per-side", type=int, default=None,
                        help="hold this many names long and as many short instead of a share (default: the "
                             "config's evaluation.book_per_side, if set)")
    parser.add_argument("--cost", type=float, default=None, help="round-trip bps per position (default: the config's headline cost)")
    parser.add_argument("--min-peers", type=int, default=None, help="fewest symbols for a moment to count (default: 20, or 60%% of a small universe)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = output_dir(cfg)
    if args.n_per_side is None:
        args.n_per_side = getattr(cfg.evaluation, "book_per_side", None)
    cost = args.cost if args.cost is not None else float(cfg.evaluation.headline_cost_bps)
    n_boot = int(getattr(cfg.evaluation, "bootstrap", 2000))
    pred = load_predictions(out)
    if pred.empty:
        print(f"no predictions under {out / 'predictions'} — run scripts/sweep.py first")
        return 1

    models = [m for m in (args.models or sorted(pred["arch"].unique())) if m != "always_up" and (pred["arch"] == m).any()]
    if not models:
        print(f"no model with a cross-sectional score among {sorted(pred['arch'].unique())} (always_up has none)")
        return 1
    common = set.intersection(*(set(pred[pred["arch"] == m]["fold"].unique()) for m in models))
    sectors = sector_map(cfg.data.universe)
    n_symbols = pred["symbol"].nunique()
    min_peers = args.min_peers or min(20, max(5, int(0.6 * n_symbols)))
    use_sectors = sectors is not None and len(set(sectors.values())) >= 3
    print(f"{cfg.experiment}: {n_symbols} symbols, models {models}, min peers {min_peers}, "
          f"sectors: {'yes' if use_sectors else 'no'}, cost {cost} bps per position")
    load, train, trade = resolve_roles(cfg.data)
    holdout = train != trade
    target = getattr(cfg.features, "target", "direction")
    if args.n_per_side and args.n_per_side * 2 > min_peers:
        min_peers = max(min_peers, 2 * args.n_per_side)
    book_args = {"q": args.q, "min_peers": min_peers, "n_per_side": args.n_per_side}

    frames = {}
    for m in models:
        frame = ensemble(pred[(pred["arch"] == m) & pred["fold"].isin(common)])
        frames[m] = with_sector(frame, sectors)

    rows, books = [], {}
    for m, frame in frames.items():
        wide = wide_moments(frame, min_peers)
        if wide.empty or wide.groupby(["date", "bar_index"])["prob"].std().fillna(0).max() < 1e-9:
            print(f"  {m}: no cross-sectional variation in its scores, skipped")
            continue
        confidence = (frame["prob"] - 0.5).abs().to_numpy()
        top = top_share_mask(confidence, frame["fold"].to_numpy(), 0.10)
        row = {"model": m,
               "all": score_book(directional_book(frame), cost, n_boot),
               "top": score_book(directional_book(frame, mask=top), cost, n_boot),
               "ic": cross_sectional_ic(frame, min_peers=min_peers, n_boot=n_boot),
               "ls": score_book(quantile_book(frame, **book_args), cost, n_boot),
               "sigma": cross_sectional_sigma(frame, min_peers)}
        if use_sectors:
            row["ic_sector"] = cross_sectional_ic(frame, min_peers=min_peers, sector_neutral=True, n_boot=n_boot)
            row["ls_sector"] = score_book(quantile_book(frame, sector_neutral=True, **book_args), cost, n_boot)
        rows.append(row)
        print(f"  {m}: IC {f(row['ic'].get('ic'), '+.4f')}, long-short {f(row['ls'].get('gross'))} bps/position")

    trained = {m: fr for m, fr in frames.items() if m not in RULES and m in [r["model"] for r in rows]}
    similarity = None
    if len(trained) >= 2:
        merged, similarity = ensemble_scores(trained, min_peers)
        merged = with_sector(merged, sectors)
        ens = {"model": f"ensemble of {len(trained)} trained models",
               "ic": cross_sectional_ic(merged, score="ens", min_peers=min_peers, n_boot=n_boot),
               "ls": score_book(quantile_book(merged, score="ens", **book_args), cost, n_boot),
               "sigma": cross_sectional_sigma(merged, min_peers)}
        if use_sectors:
            ens["ic_sector"] = cross_sectional_ic(merged, score="ens", min_peers=min_peers, sector_neutral=True, n_boot=n_boot)
            ens["ls_sector"] = score_book(quantile_book(merged, score="ens", sector_neutral=True, **book_args), cost, n_boot)
        rows.append(ens)

    if not rows:
        print("no model has a cross-sectional score at this universe size")
        return 1

    q_pct = f"{args.n_per_side} names" if args.n_per_side else f"{args.q:.0%}"
    setup = []
    if holdout:
        setup.append(f"Models were trained on {len(train)} symbols and are scored here only on {len(trade)} others "
                     "they never saw (a hold-out across stocks, not across time).")
    if target != "direction":
        setup.append(f"Target: {target} (each return is measured against its universe's average at the same moment).")
    lines = [f"# {cfg.experiment}: breadth and cross-sectional skill", "",
             f"{n_symbols} symbols; moments (a date and bar) with fewer than {min_peers} symbols are left out. Net is after "
             f"{cost} bps per position round trip. Intervals are 95% day-block bootstrap. No retraining: saved walk-forward "
             "predictions only. " + " ".join(setup), "",
             "## 1. How many independent bets is a day's trading?", "",
             "The same model used three ways. **Exposure** is the average of long minus short over positions (1.0 = all long). "
             "**Bets/day** is the effective number of independent bets: n / (1 + (n − 1)ρ), where ρ is the average "
             "correlation between a day's positions, read from how much more the day's average moves than independent "
             "positions would. About 1 means one bet on the market; the number of positions means full diversification.", "",
             f"| model | all signals: positions/day · exposure · bets/day | top 10% by confidence: same | long {q_pct} / short {q_pct}: same |",
             "|---|---|---|---|"]

    def breadth_cell(r):
        return (f"{r['positions_per_day']:.0f} · {r['net_exposure']:+.2f} · **{r['bets_per_day']:.1f}**"
                if r.get("positions") else "—")

    for r in rows:
        if "all" in r:
            lines.append(f"| {r['model']} | {breadth_cell(r['all'])} | {breadth_cell(r['top'])} | {breadth_cell(r['ls'])} |")
    explain = (f"**IC** is the rank correlation between the model's score and the realized return across stocks at each "
               f"moment, averaged over days. **Long-short** holds the top {q_pct} and bottom {q_pct} of the score at each "
               "moment: market-neutral, gross bps per position (long positions earn the return, short positions the negative).")
    if use_sectors:
        explain += (" The **sector** columns do the same within each sector, so a model that only knows which sector will "
                    "move scores nothing.")
    lines += ["", "## 2. Skill across stocks, with the market removed", "", explain, "",
              "| model | IC [95% CI] | t | days IC > 0 | long-short gross [95% CI] | net | bets/day |" +
              (" sector IC [95% CI] | sector long-short gross [95% CI] | net |" if use_sectors else ""),
              "|---|---|---:|---:|---|---:|---:|" + ("---|---|---:|" if use_sectors else "")]
    for r in rows:
        ic, ls = r["ic"], r["ls"]
        line = (f"| {r['model']} | {interval(ic, 'ic_lo', 'ic_hi', 'ic', '+.4f')} | {f(ic.get('t'), '+.1f')} | "
                f"{f(ic.get('share_days_positive'), '.0%')} | {interval(ls)} | {f(ls.get('net'))} | {f(ls.get('bets_per_day'), '.1f')} |")
        if use_sectors:
            ics, lss = r["ic_sector"], r["ls_sector"]
            line += f" {interval(ics, 'ic_lo', 'ic_hi', 'ic', '+.4f')} | {interval(lss)} | {f(lss.get('net'))} |"
        lines.append(line)

    ref = next((r for r in rows if r.get("ic", {}).get("ic_se")), None)
    sigma = float(np.median([r["sigma"] for r in rows]))
    q_eff = args.n_per_side / float(np.median([r["ic"]["peers"] for r in rows if r["ic"].get("peers")])) if args.n_per_side else args.q
    be = breakeven_ic(cost, sigma, q_eff)
    detectable = float(np.median([r["ic"]["ic_min_detectable"] for r in rows if r["ic"].get("ic_min_detectable")]))
    lines += ["", "## 3. What this data can and cannot rule out", "",
              f"- Stocks differ from each other by about **{sigma:.0f} bps** over the holding period (standard deviation across "
              "stocks within a moment, market removed).",
              f"- A long-short book of the top and bottom {q_pct} only pays its {cost} bps cost if the IC is at least "
              f"**{be:.4f}** (cost ÷ [spread × average |z| of the picks]).",
              f"- The smallest IC this much data would detect 80% of the time is about **{detectable:.4f}** "
              "(2.8 standard errors of the mean IC).",
              ("- So a null result **does rule out** a profitable signal of this kind: anything strong enough to pay would "
               "have shown up." if detectable < be else
               "- So a null result **does not rule out** a profitable signal: one could exist and sit below what this data "
               "can detect.")]
    if similarity:
        lines.append(f"- The {similarity['models']} trained models' scores correlate {similarity['mean_corr']:.2f} on average: "
                     f"together they are about **{similarity['independent_models']:.1f} independent models**. "
                     "Averaging models that share inputs and labels buys little diversification.")
    text = "\n".join(lines) + "\n"
    (out / "breadth.md").write_text(text)

    flat = []
    for r in rows:
        for kind in ("all", "top", "ls", "ls_sector", "ic", "ic_sector"):
            if kind in r:
                flat.append({"model": r["model"], "measure": kind, **{k: v for k, v in r[kind].items()}})
    pd.DataFrame(flat).to_csv(out / "breadth.csv", index=False)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
