"""The walk-forward sweep and its report. Scripts are thin wrappers around this.

Predictions are written one fold at a time, so a sweep that dies halfway
(a Colab disconnect, a closed laptop) resumes where it stopped instead of
starting over. Every model and rule is scored on exactly the same test rows.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import is_rule, rule_probability
from .dataset import GATE_COLUMNS
from .features import apply_standardizer, fit_standardizer
from .models import count_parameters
from .screening import GATE_STATISTICS
from .stats import compare_subsets, coverage_eligible, coverage_mask, summarize

KEY = ["fold", "date", "symbol", "bar_index"]


def fold_path(out_dir: Path, arch: str, seed: int, fold: int) -> Path:
    return Path(out_dir) / "predictions" / arch / f"seed{seed}" / f"fold{fold:03d}.parquet"


def run_arch(data: dict, folds: list[dict], arch: str, seed: int, cfg, out_dir: Path, log=print) -> int:
    """Score one architecture (or rule) with one seed on every fold. Returns folds newly run."""
    ran = 0
    deadzone = getattr(cfg.features, "label_deadzone", 0.0)
    for i, fold in enumerate(folds):
        path = fold_path(out_dir, arch, seed, i)
        if path.exists():
            continue
        started = time.time()
        tr, va, te = fold["train"], fold["val"], fold["test"]
        info = {"best_epoch": -1}
        n_params = 0

        if is_rule(arch):
            prob = rule_probability(arch, data, te)
        else:
            # Small moves are mostly noise to learn from, so training may skip
            # them. The test set is never filtered this way: which moves turn out
            # small is only known afterwards.
            tr = tr[np.abs(data["ret"][tr]) >= deadzone]
            va = va[np.abs(data["ret"][va]) >= deadzone]
            if len(tr) < 50 or len(va) < 10:
                log(f"  fold {i:3d} skipped: only {len(tr)} train / {len(va)} val samples")
                continue
            X = data["X"]
            if cfg.features.normalize == "fold_standardize":
                stats = fit_standardizer(X[tr])
                Xtr, Xva, Xte = (apply_standardizer(X[j], stats) for j in (tr, va, te))
            else:
                Xtr, Xva, Xte = X[tr], X[va], X[te]

            from .training import predict, train_fold

            model, info = train_fold(Xtr, data["y"][tr], Xva, data["y"][va], cfg.model, arch, seed)
            prob = predict(model, Xte)
            n_params = count_parameters(model)

        frame = pd.DataFrame(
            {
                "fold": i,
                "date": data["date"][te],
                "symbol": data["symbol"][te],
                "bar_index": data["bar_index"][te],
                "y": data["y"][te],
                "ret": data["ret"][te],
                "prob": prob,
                "arch": arch,
                "seed": seed,
                "n_params": n_params,
                "best_epoch": info["best_epoch"],
            }
        )
        for col in GATE_COLUMNS + ["has_reading"]:
            if col in data:
                frame[col] = data[col][te]

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        frame.to_parquet(tmp)
        tmp.replace(path)
        ran += 1

        taken = frame[frame["prob"] != 0.5]
        direction = np.where(taken["prob"] > 0.5, 1.0, -1.0)
        acc = ((direction > 0) == (taken["y"] == 1)).mean() if len(taken) else float("nan")
        gross = (direction * taken["ret"]).mean() * 1e4 if len(taken) else float("nan")
        undertrained = not is_rule(arch) and info["best_epoch"] == cfg.model.epochs - 1
        log(
            f"  fold {i:3d} [{str(fold['test_start'])[:10]}..{str(fold['test_end'])[:10]}]  "
            f"acc {acc:.3f}  gross {gross:+6.2f}bps  trades {len(taken):,}  {time.time() - started:5.0f}s"
            + ("  (still improving at the last epoch: raise epochs)" if undertrained else "")
        )
    return ran


def load_predictions(out_dir: Path) -> pd.DataFrame:
    parts = sorted((Path(out_dir) / "predictions").glob("*/seed*/fold*.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def ensemble(pred: pd.DataFrame) -> pd.DataFrame:
    """Average each sample's probability across seeds."""
    extra = [c for c in GATE_COLUMNS + ["has_reading", "y", "ret"] if c in pred.columns]
    agg = {c: "first" for c in extra}
    agg["prob"] = "mean"
    return pred.groupby(KEY, as_index=False).agg(agg)


def report(pred: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three tables: overall performance, the gate comparison (Q1), and confidence filtering.

    All three read the same saved predictions, so none of them needs retraining.
    """
    ev = cfg.evaluation
    folds_by_arch = pred.groupby("arch")["fold"].nunique()
    common_folds = set.intersection(*(set(g["fold"].unique()) for _, g in pred.groupby("arch")))
    pred = pred[pred["fold"].isin(common_folds)]

    overall, gate, confidence = [], [], []
    per_symbol = pred["symbol"].nunique() > 1
    levels = getattr(ev, "coverage_levels", [1.0])
    for arch, group in pred.groupby("arch", sort=False):
        per_seed = [summarize(g, n_boot=1, seed=0) for _, g in group.groupby("seed")]
        combined = ensemble(group)
        s = summarize(combined, n_boot=ev.bootstrap)
        row = {
            "arch": arch,
            "params": int(group["n_params"].iloc[0]),
            "seeds": group["seed"].nunique(),
            "folds": len(common_folds),
            "test_days": s["n_days"],
            "trades": s["n_trades"],
            "accuracy": s["accuracy"],
            "accuracy_lo": s["accuracy_lo"],
            "accuracy_hi": s["accuracy_hi"],
            "seed_sd_accuracy": float(np.std([p["accuracy"] for p in per_seed])),
            "gross_bps": s["gross_bps"],
            "gross_bps_lo": s["gross_bps_lo"],
            "gross_bps_hi": s["gross_bps_hi"],
            "seed_sd_gross_bps": float(np.std([p["gross_bps"] for p in per_seed])),
            "breakeven_cost_bps": s["gross_bps"] if s["gross_bps"] > 0 else float("nan"),
        }
        for cost in ev.cost_bps:
            row[f"net_bps_at_{cost}"] = s["gross_bps"] - cost
        overall.append(row)

        if "has_reading" in combined.columns:
            for stat in GATE_STATISTICS:
                mask = combined[f"pass_{stat}"].to_numpy(bool)
                row = {"arch": arch, "gate": stat, **compare_subsets(combined, mask, n_boot=ev.bootstrap)}
                if per_symbol:
                    within = compare_subsets(combined, mask, n_boot=ev.bootstrap, demean_by="symbol")
                    row |= {
                        "within_diff_bps": within["diff_bps"],
                        "within_diff_lo": within["diff_bps_lo"],
                        "within_diff_hi": within["diff_bps_hi"],
                    }
                gate.append(row)

        eligible = coverage_eligible(combined)
        for level in levels:
            keep = coverage_mask(combined, level)
            if not keep.any():
                continue
            c = summarize(combined[keep], n_boot=ev.bootstrap)
            confidence.append({
                "arch": arch,
                "coverage_target": level,
                "coverage_actual": float(keep.sum() / max(int(eligible.sum()), 1)),
                "trades": c["n_trades"],
                "accuracy": c["accuracy"],
                "gross_bps": c["gross_bps"],
                "gross_bps_lo": c["gross_bps_lo"],
                "gross_bps_hi": c["gross_bps_hi"],
                "net_bps": c["gross_bps"] - ev.headline_cost_bps,
            })

    skipped = folds_by_arch[folds_by_arch > len(common_folds)]
    if len(skipped):
        print(f"note: scoring only the {len(common_folds)} folds every arch has finished")
    return pd.DataFrame(overall), pd.DataFrame(gate), pd.DataFrame(confidence)


def _fmt(v, spec):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, spec)


def markdown(overall: pd.DataFrame, gate: pd.DataFrame, confidence: pd.DataFrame, cfg) -> str:
    headline = cfg.evaluation.headline_cost_bps
    lines = [
        f"# {cfg.experiment}",
        "",
        f"{int(overall['folds'].iloc[0])} walk-forward folds, {int(overall['test_days'].iloc[0])} test days. "
        "Intervals are 95% day-block bootstrap. `±seeds` is the spread across training seeds.",
        "",
        f"| model | params | accuracy [95% CI] | ±seeds | gross bps/trade [95% CI] | ±seeds | net @ {headline} bps | breakeven |",
        "|---|---:|---|---:|---|---:|---:|---:|",
    ]
    for _, r in overall.iterrows():
        lines.append(
            f"| {r['arch']} | {r['params']:,} | {_fmt(r['accuracy'], '.3f')} "
            f"[{_fmt(r['accuracy_lo'], '.3f')}, {_fmt(r['accuracy_hi'], '.3f')}] | {_fmt(r['seed_sd_accuracy'], '.3f')} "
            f"| {_fmt(r['gross_bps'], '+.2f')} [{_fmt(r['gross_bps_lo'], '+.2f')}, {_fmt(r['gross_bps_hi'], '+.2f')}] "
            f"| {_fmt(r['seed_sd_gross_bps'], '.2f')} | {_fmt(r[f'net_bps_at_{headline}'], '+.2f')} "
            f"| {'no edge' if not np.isfinite(r['breakeven_cost_bps']) else format(r['breakeven_cost_bps'], '.2f')} |"
        )
    if len(gate):
        within = "within_diff_bps" in gate.columns
        lines += [
            "",
            "## Q1: does the gate pick better days?",
            "",
            "Same predictions, split by whether the gate approved the day. Under pure noise about 5% of days pass.",
            "",
        ]
        if within:
            lines += [
                "The last column repeats the comparison after subtracting each ETF's own average, since the gate "
                "refuses whole ETFs that move only a few cents a bar; without it the split partly compares "
                "expensive ETFs against cheap ones.",
                "",
                "| model | gate | days approved | gross bps approved | gross bps rest | difference [95% CI] | within ETF [95% CI] |",
                "|---|---|---:|---:|---:|---|---|",
            ]
        else:
            lines += [
                "| model | gate | days approved | gross bps approved | gross bps rest | difference [95% CI] |",
                "|---|---|---:|---:|---:|---|",
            ]
        for _, r in gate.iterrows():
            line = (
                f"| {r['arch']} | {r['gate']} | {r['share_in']:.1%} | {_fmt(r['gross_bps_in'], '+.2f')} "
                f"| {_fmt(r['gross_bps_out'], '+.2f')} | {_fmt(r['diff_bps'], '+.2f')} "
                f"[{_fmt(r['diff_bps_lo'], '+.2f')}, {_fmt(r['diff_bps_hi'], '+.2f')}] |"
            )
            if within:
                line += (
                    f" {_fmt(r['within_diff_bps'], '+.2f')} "
                    f"[{_fmt(r['within_diff_lo'], '+.2f')}, {_fmt(r['within_diff_hi'], '+.2f')}] |"
                )
            lines.append(line)

    if len(confidence):
        lines += [
            "",
            "## Does trading only confident predictions help?",
            "",
            "Each fold's confidence cutoff comes from earlier folds only, so nothing here uses the future. "
            "Early folds without enough history to set a cutoff are dropped at every level, so the levels stay "
            "comparable. A cut can only land where confidences differ, so rules whose probabilities are always "
            "0 or 1 keep every trade; the realized share is shown next to the target.",
            "",
            f"| model | target | actual | trades | accuracy | gross bps/trade [95% CI] | net @ {headline} bps |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
        for _, r in confidence.iterrows():
            lines.append(
                f"| {r['arch']} | {r['coverage_target']:.0%} | {r['coverage_actual']:.0%} | {int(r['trades']):,} "
                f"| {_fmt(r['accuracy'], '.3f')} | {_fmt(r['gross_bps'], '+.2f')} "
                f"[{_fmt(r['gross_bps_lo'], '+.2f')}, {_fmt(r['gross_bps_hi'], '+.2f')}] "
                f"| {_fmt(r['net_bps'], '+.2f')} |"
            )
    return "\n".join(lines) + "\n"
