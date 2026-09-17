#!/usr/bin/env python3
"""End-to-end check on synthetic bars with known structure — no credentials needed.

    python scripts/smoke_test.py

1. The gate passes trending series, refuses zig-zags under the trend statistic,
   and refuses tick-constrained prices.
2. Positive controls: planted signals in both setups must be recovered, or a
   null result on real data would say nothing about the market.
3. The sweep and report run end to end and write a summary.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from _common import ROOT

from src.config import load_config
from src.dataset import build_dataset, walk_forward_splits
from src.experiment import load_predictions, markdown, report, run_arch
from src.screening import apply_gate, screen_universe
from src.synthetic import ar_returns, bars_from_returns, half_hour_momentum_returns

SESSIONS = 300
FAST_VALIDATION = SimpleNamespace(train_days=120, val_days=20, test_days=40, step_days=40)


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def gate_checks(cfg) -> bool:
    print("1. gate")
    bars = {
        "TREND": bars_from_returns("TREND", ar_returns(SESSIONS, +0.15, seed=1)),
        "NOISE": bars_from_returns("NOISE", ar_returns(SESSIONS, 0.0, seed=2)),
        "ZIGZAG": bars_from_returns("ZIGZAG", ar_returns(SESSIONS, -0.30, seed=3)),
        "CHEAP": bars_from_returns("CHEAP", ar_returns(SESSIONS, 0.0, scale=0.0008, seed=4), price=12.0, tick=0.01),
    }
    s = cfg.screening
    gated = apply_gate(screen_universe(bars, s), s.gate_statistic, s.alpha, s.max_tie_fraction, s.min_ticks_per_bar)
    rate = gated.groupby("symbol")[["pass_trend", "pass_entropy", "tick_limited"]].mean()
    for sym, r in rate.iterrows():
        print(f"     {sym:7s} trend {r.pass_trend:6.1%}   entropy {r.pass_entropy:6.1%}   tick-limited {r.tick_limited:6.1%}")
    return all([
        check("trend passes a trending series", rate.loc["TREND", "pass_trend"] > 0.3, f"{rate.loc['TREND', 'pass_trend']:.0%}"),
        check("noise passes near the 5% false-positive rate", rate.loc["NOISE", "pass_trend"] < 0.12, f"{rate.loc['NOISE', 'pass_trend']:.0%}"),
        check("entropy gate is fooled by zig-zags", rate.loc["ZIGZAG", "pass_entropy"] > 0.5, f"{rate.loc['ZIGZAG', 'pass_entropy']:.0%}"),
        check("trend gate is not", rate.loc["ZIGZAG", "pass_trend"] < 0.05, f"{rate.loc['ZIGZAG', 'pass_trend']:.0%}"),
        check("tick-constrained prices are refused", rate.loc["CHEAP", "tick_limited"] > 0.9, f"{rate.loc['CHEAP', 'tick_limited']:.0%}"),
    ])


def control(name: str, cfg, bars: dict, archs: list[str], threshold: dict[str, float]) -> bool:
    data = build_dataset(bars, None, cfg)
    folds = walk_forward_splits(data["date"], FAST_VALIDATION)[:3]
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for arch in archs:
            run_arch(data, folds, arch, 0, cfg, Path(tmp), log=lambda _: None)
        pred = load_predictions(Path(tmp))
    for arch in archs:
        p = pred[pred["arch"] == arch]
        taken = p[p["prob"] != 0.5]
        acc = ((taken["prob"] > 0.5) == (taken["y"] == 1)).mean()
        ok &= check(f"{name}: {arch}", acc >= threshold[arch], f"accuracy {acc:.3f} (needs >= {threshold[arch]:.2f})")
    return ok


def main() -> int:
    results = []
    cfg = load_config(ROOT / "configs/etf_intraday.yaml")
    results.append(gate_checks(cfg))

    print("\n2. positive controls")
    rng = np.random.default_rng(7)
    latent = np.zeros(SESSIONS * 78)
    for i in range(1, latent.size):
        latent[i] = 0.98 * latent[i - 1] + rng.normal(0, 0.0002)
    planted = latent + rng.normal(0, 0.0008, latent.size)
    results.append(control(
        "intraday, persistent drift", cfg,
        {f"SIG{i}": bars_from_returns(f"SIG{i}", np.roll(planted, 7 * 78 * i), seed=i) for i in range(3)},
        ["always_up", "momentum_window", "logreg", "resnet1d"],
        {"always_up": 0.0, "momentum_window": 0.60, "logreg": 0.60, "resnet1d": 0.60},
    ))

    gao = load_config(ROOT / "configs/spy_gao.yaml")
    results.append(control(
        "first half-hour -> last half-hour", gao,
        {"SPY": bars_from_returns("SPY", half_hour_momentum_returns(SESSIONS, beta=1.0, seed=5))},
        ["always_up", "momentum_day", "logreg"],
        {"always_up": 0.0, "momentum_day": 0.62, "logreg": 0.60},
    ))

    print("\n3. sweep + report")
    bars = {"SPY": bars_from_returns("SPY", half_hour_momentum_returns(SESSIONS, beta=1.0, seed=6))}
    screen = apply_gate(screen_universe(bars, gao.screening), "trend")
    data = build_dataset(bars, screen, gao)
    folds = walk_forward_splits(data["date"], FAST_VALIDATION)[:2]
    gao.evaluation.bootstrap = 200
    with tempfile.TemporaryDirectory() as tmp:
        for arch in ["always_up", "momentum_day", "logreg"]:
            run_arch(data, folds, arch, 0, gao, Path(tmp), log=lambda _: None)
        rerun = run_arch(data, folds, "logreg", 0, gao, Path(tmp), log=lambda _: None)
        overall, gate = report(load_predictions(Path(tmp)), gao)
        text = markdown(overall, gate, gao)
    results.append(check("interrupted sweeps resume instead of retraining", rerun == 0, f"{rerun} folds re-run"))
    results.append(check("report has every model and a Q1 table", len(overall) == 3 and len(gate) == 9,
                         f"{len(overall)} models, {len(gate)} gate rows"))
    print("\n" + "\n".join("     " + line for line in text.splitlines()[:9]))

    print("\n" + ("pipeline OK — ready for real data" if all(results) else "SOME CHECKS FAILED"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
