"""Training loop and evaluation.

Evaluation reports cost-adjusted trade economics alongside classification
accuracy. A model can be reliably right about direction and still lose money
once the spread is paid, so accuracy on its own is not an answer about whether
anything here is tradeable.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import build_model, expects_images


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg,
    arch: str | None = None,
    verbose: bool = False,
) -> tuple[torch.nn.Module, dict]:
    """Train one walk-forward fold, keeping the weights that scored best on val."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _device()
    arch = arch or cfg.arch

    is_image_data = X_train.ndim == 4
    if expects_images(arch) != is_image_data:
        shape = "images (n, C, L, L)" if is_image_data else "sequences (n, C, L)"
        want = "GAF/MTF images" if expects_images(arch) else "1-D sequences"
        raise ValueError(
            f"arch '{arch}' expects {want} but the dataset holds {shape}. "
            f"Set features.encoding to {'gaf' if expects_images(arch) else '1d'} in config.yaml."
        )

    model = build_model(
        arch,
        in_channels=X_train.shape[1],
        length=X_train.shape[2],
        dropout=cfg.dropout,
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=len(y_train) > cfg.batch_size,
    )
    xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best = {"val_loss": float("inf"), "epoch": -1, "state": None}
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)

        model.eval()
        with torch.no_grad():
            val_logits = model(xv)
            val_loss = criterion(val_logits, yv).item()
            val_acc = (val_logits.argmax(1) == yv).float().mean().item()

        history.append({"epoch": epoch, "train_loss": total / len(y_train), "val_loss": val_loss, "val_acc": val_acc})
        if val_loss < best["val_loss"]:
            best = {
                "val_loss": val_loss,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        if verbose and epoch % 5 == 0:
            print(f"    epoch {epoch:3d}  train {total/len(y_train):.4f}  val {val_loss:.4f}  acc {val_acc:.3f}")

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {"history": history, "best_epoch": best["epoch"], "best_val_loss": best["val_loss"]}


@torch.no_grad()
def predict(model: torch.nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Class-1 probability for each sample."""
    device = _device()
    model.eval().to(device)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        out.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def evaluate(
    prob_up: np.ndarray,
    y: np.ndarray,
    ret: np.ndarray,
    threshold: float = 0.5,
    cost_bps: float = 0.0,
    trading_days: int | None = None,
) -> dict:
    """Classification and cost-adjusted trade economics.

    `cost_bps` is charged per round trip, so it covers both the spread crossed
    on entry and the one crossed on exit.
    """
    if prob_up.size == 0:
        return {"n": 0}

    confidence = np.abs(prob_up - 0.5)
    take = confidence >= (threshold - 0.5) if threshold > 0.5 else np.ones_like(prob_up, dtype=bool)
    direction = np.where(prob_up > 0.5, 1.0, -1.0)

    gross = direction[take] * ret[take]
    net = gross - cost_bps / 10_000.0

    metrics = {
        "n": int(prob_up.size),
        "n_trades": int(take.sum()),
        "coverage": float(take.mean()),
        "accuracy": float(((prob_up > 0.5).astype(int) == y).mean()),
        "hit_rate": float((gross > 0).mean()) if gross.size else float("nan"),
        "mean_return_bps_gross": float(gross.mean() * 10_000) if gross.size else float("nan"),
        "mean_return_bps_net": float(net.mean() * 10_000) if net.size else float("nan"),
        "total_return_net": float(net.sum()) if net.size else float("nan"),
    }

    if net.size > 1 and net.std() > 0:
        per_trade_sharpe = net.mean() / net.std()
        if trading_days and trading_days > 0:
            trades_per_year = net.size / trading_days * 252
            metrics["sharpe_annual_net"] = float(per_trade_sharpe * np.sqrt(trades_per_year))
        metrics["sharpe_per_trade_net"] = float(per_trade_sharpe)

    if gross.size:
        metrics["breakeven_cost_bps"] = float(gross.mean() * 10_000)
    return metrics
