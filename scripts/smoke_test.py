#!/usr/bin/env python3
"""End-to-end check on synthetic bars — no credentials needed.

Half the symbols get autocorrelated returns and half are pure random walks, so
the screen can be checked against known ground truth: if the gate cannot
separate these, it will not separate anything on real data.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.dataset import build_dataset, walk_forward_splits
from src.screening import apply_gate, capacity_diagnostic, screen_universe
from src.training import evaluate, predict, train_fold

BARS_PER_SESSION = 78
N_SESSIONS = 200


def synthetic_symbol(name: str, phi: float, seed: int) -> pd.DataFrame:
    """OHLCV bars whose returns follow AR(1) with coefficient phi."""
    rng = np.random.default_rng(seed)
    n = BARS_PER_SESSION * N_SESSIONS

    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + rng.normal(0, 0.0015)

    close = 100 * np.exp(np.cumsum(r))
    spread = np.abs(rng.normal(0, 0.0008, n)) * close
    open_ = np.concatenate([[close[0]], close[:-1]])

    days = pd.bdate_range("2024-01-02", periods=N_SESSIONS)
    offsets = pd.timedelta_range("09:30:00", periods=BARS_PER_SESSION, freq="5min")
    timestamps = pd.DatetimeIndex([d + o for d in days for o in offsets]).tz_localize("America/New_York")
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.lognormal(11, 0.6, n),
            "session_id": np.repeat(np.arange(N_SESSIONS), BARS_PER_SESSION),
        },
        index=pd.MultiIndex.from_arrays(
            [np.full(n, name), timestamps], names=["symbol", "timestamp"]
        ),
    )
    return df


def planted_signal_symbol(name: str, seed: int) -> pd.DataFrame:
    """Bars carrying a genuinely learnable horizon-matched signal.

    A slow latent drift persists far longer than the holding period, so the mean
    of the feature window really does predict the forward return. This is the
    positive control: if training cannot recover a signal that was deliberately
    planted, a null result on real data says nothing about the market.
    """
    rng = np.random.default_rng(seed)
    n = BARS_PER_SESSION * N_SESSIONS

    latent = np.zeros(n)
    for i in range(1, n):
        latent[i] = 0.98 * latent[i - 1] + rng.normal(0, 0.0002)
    r = latent + rng.normal(0, 0.0008, n)

    close = 100 * np.exp(np.cumsum(r))
    spread = np.abs(rng.normal(0, 0.0008, n)) * close
    open_ = np.concatenate([[close[0]], close[:-1]])

    days = pd.bdate_range("2024-01-02", periods=N_SESSIONS)
    offsets = pd.timedelta_range("09:30:00", periods=BARS_PER_SESSION, freq="5min")
    timestamps = pd.DatetimeIndex([d + o for d in days for o in offsets]).tz_localize("America/New_York")

    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.lognormal(11, 0.6, n),
            "session_id": np.repeat(np.arange(N_SESSIONS), BARS_PER_SESSION),
        },
        index=pd.MultiIndex.from_arrays([np.full(n, name), timestamps], names=["symbol", "timestamp"]),
    )


def positive_control(cfg, validation, model_cfg) -> None:
    planted = {f"SIG{i}": planted_signal_symbol(f"SIG{i}", 100 + i) for i in range(4)}
    data = build_dataset(planted, None, cfg, use_gate=False)
    folds = walk_forward_splits(data["date"], validation)

    accuracies = []
    for fold in folds[:3]:
        tr, va, te = fold["train"], fold["val"], fold["test"]
        model, _ = train_fold(
            data["X"][tr], data["y"][tr], data["X"][va], data["y"][va], model_cfg
        )
        metrics = evaluate(predict(model, data["X"][te]), data["y"][te], data["ret"][te], cost_bps=0.0)
        accuracies.append(metrics["accuracy"])
        print(f"  fold acc {metrics['accuracy']:.3f}  gross {metrics['mean_return_bps_gross']:+.2f}bps")

    mean_acc = float(np.mean(accuracies))
    status = "RECOVERS planted signal" if mean_acc > 0.55 else "FAILED to recover planted signal"
    print(f"  mean accuracy {mean_acc:.3f} -> {status}")


def main() -> int:
    cfg = load_config("config.yaml")

    predictable = {f"AR{i}": 0.30 for i in range(5)}
    noisy = {f"RW{i}": 0.0 for i in range(5)}
    truth = {**predictable, **noisy}
    bars = {name: synthetic_symbol(name, phi, seed) for seed, (name, phi) in enumerate(truth.items())}
    print(f"generated {len(bars)} symbols x {N_SESSIONS} sessions x {BARS_PER_SESSION} bars")

    print("\n--- 1. screening ---")
    table = screen_universe(bars, cfg.screening)
    gated = apply_gate(table, cfg.screening.select_quantile)
    by_symbol = gated.groupby("symbol").agg(pe=("pe", "mean"), pass_rate=("passed", "mean"))

    ar_pe = by_symbol.loc[[s for s in truth if s.startswith("AR")], "pe"].mean()
    rw_pe = by_symbol.loc[[s for s in truth if s.startswith("RW")], "pe"].mean()
    for sym, row in by_symbol.sort_values("pe").iterrows():
        kind = "autocorrelated" if sym.startswith("AR") else "random walk"
        print(f"  {sym:6s} PE {row.pe:.4f}  pass {row.pass_rate:5.1%}   ({kind})")
    print(f"\n  mean PE: autocorrelated {ar_pe:.4f}  vs  random walk {rw_pe:.4f}")

    ar_pass = by_symbol.loc[[s for s in truth if s.startswith("AR")], "pass_rate"].mean()
    rw_pass = by_symbol.loc[[s for s in truth if s.startswith("RW")], "pass_rate"].mean()
    verdict = "SEPARATES" if ar_pe < rw_pe and ar_pass > rw_pass else "FAILS TO SEPARATE"
    print(f"  gate {verdict}: selects autocorrelated names {ar_pass:.0%} of sessions "
          f"vs {rw_pass:.0%} for random walks")

    capacity = capacity_diagnostic(gated)
    print(f"  capacity: selected names at {capacity['dv_ratio'].median():.2f}x universe median volume")

    print("\n--- 2. dataset ---")
    validation = SimpleNamespace(train_days=40, val_days=10, test_days=10, step_days=20)
    for use_gate in (True, False):
        data = build_dataset(bars, gated, cfg, use_gate=use_gate)
        folds = walk_forward_splits(data["date"], validation)
        label = "gated" if use_gate else "ungated"
        print(f"  {label:8s} X={data['X'].shape}  labels {data['y'].mean():.3f} up  folds={len(folds)}")
        if use_gate:
            gated_data, gated_folds = data, folds

    print("\n--- 3. training (2 folds, short run) ---")
    model_cfg = SimpleNamespace(
        arch="resnet1d", epochs=4, batch_size=128, lr=1e-3,
        weight_decay=1e-4, dropout=0.2, seed=42,
    )
    for i, fold in enumerate(gated_folds[:2]):
        tr, va, te = fold["train"], fold["val"], fold["test"]
        model, info = train_fold(
            gated_data["X"][tr], gated_data["y"][tr],
            gated_data["X"][va], gated_data["y"][va], model_cfg,
        )
        metrics = evaluate(
            predict(model, gated_data["X"][te]), gated_data["y"][te], gated_data["ret"][te],
            cost_bps=5.0, trading_days=len(np.unique(gated_data["date"][te])),
        )
        print(f"  fold {i}: train {len(tr):,} test {len(te):,}  "
              f"acc {metrics['accuracy']:.3f}  gross {metrics['mean_return_bps_gross']:+.2f}bps  "
              f"net {metrics['mean_return_bps_net']:+.2f}bps")

    print("\n--- 4. positive control (planted signal) ---")
    positive_control(cfg, validation, model_cfg)

    print("\n--- 5. GAF image path (resnet2d) ---")
    gaf_cfg = load_config("config.yaml")
    gaf_cfg.features.encoding = "gaf"
    subset = {k: bars[k] for k in list(bars)[:3]}
    gaf_data = build_dataset(subset, None, gaf_cfg, use_gate=False)
    print(f"  encoded {gaf_data['X'].shape}  ({gaf_data['X'].nbytes / 1e6:.0f} MB — "
          f"{gaf_data['X'].shape[-1]}x the memory of the 1d encoding)")

    gaf_model_cfg = SimpleNamespace(
        arch="resnet2d", epochs=2, batch_size=64, lr=1e-3, weight_decay=1e-4, dropout=0.2, seed=42
    )
    fold = walk_forward_splits(gaf_data["date"], validation)[0]
    tr, va, te = fold["train"], fold["val"], fold["test"]
    model, _ = train_fold(
        gaf_data["X"][tr], gaf_data["y"][tr], gaf_data["X"][va], gaf_data["y"][va], gaf_model_cfg
    )
    metrics = evaluate(predict(model, gaf_data["X"][te]), gaf_data["y"][te], gaf_data["ret"][te], cost_bps=5.0)
    print(f"  train {len(tr):,} test {len(te):,}  acc {metrics['accuracy']:.3f}  "
          f"net {metrics['mean_return_bps_net']:+.2f}bps")

    print("\npipeline wiring OK — ready for real data once .env is filled in")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
