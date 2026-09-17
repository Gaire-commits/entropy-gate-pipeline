"""Reproducible training.

The same seed on the same data must give the same model: otherwise run-to-run
noise is mixed into every comparison between architectures, and on the first
real sweep that noise was larger than the differences being measured.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import build_model


def device() -> torch.device:
    forced = os.environ.get("EGP_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg,
    arch: str,
    seed: int = 0,
) -> tuple[torch.nn.Module, dict]:
    """Train one walk-forward fold, keeping the weights that scored best on validation.

    Stops after `cfg.patience` epochs without a validation improvement.
    """
    set_deterministic(seed)
    dev = device()
    model = build_model(arch, in_channels=X_train.shape[1], length=X_train.shape[2], dropout=cfg.dropout).to(dev)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=len(y_train) > cfg.batch_size,
        generator=generator,
    )
    xv = torch.from_numpy(X_val).to(dev)
    yv = torch.from_numpy(y_val).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    patience = getattr(cfg, "patience", cfg.epochs)

    best = {"val_loss": float("inf"), "epoch": -1, "state": None}
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(_batched_logits(model, xv), yv).item())
        if val_loss < best["val_loss"]:
            best = {
                "val_loss": val_loss,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        elif epoch - best["epoch"] >= patience:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"], "best_val_loss": best["val_loss"], "epochs_run": epoch + 1}


def _batched_logits(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    return torch.cat([model(x[i : i + batch_size]) for i in range(0, len(x), batch_size)])


@torch.no_grad()
def predict(model: torch.nn.Module, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Probability of "up" for each sample."""
    dev = device()
    model.eval().to(dev)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(dev)
        out.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)
