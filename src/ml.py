"""Gradient-boosted trees: the classical-ML rung between the rules and the deep models.

`gbm` sees summary statistics of the price window; `gbm_ent` sees the same plus the
entropy features, so gbm_ent minus gbm is what entropy adds to a model that can
learn interactions (for example, follow the window's direction only when the
trailing returns are ordered). Trees need no scaling and treat a missing entropy
reading as its own branch.

The number of trees is chosen on the validation fold: the model is grown to
`max_iter` trees and test predictions are taken at the iteration with the lowest
validation log loss, which is early stopping without refitting. Without row
subsampling the fit is deterministic, so one seed is enough.
"""

from __future__ import annotations

import numpy as np

from .entropy_features import entropy_matrix

TREE_MODELS = ("gbm", "gbm_ent")


def is_tree(arch: str) -> bool:
    return arch in TREE_MODELS


def window_summary(X: np.ndarray) -> np.ndarray:
    """(n, channels, length) window -> (n, 5 * channels): last value, mean, spread, min and max per channel."""
    X = np.asarray(X, dtype=np.float64)
    return np.column_stack([X[:, :, -1], X.mean(axis=2), X.std(axis=2), X.min(axis=2), X.max(axis=2)])


def tree_features(data: dict, rows: np.ndarray, arch: str) -> np.ndarray:
    """Inputs for one tree model: window summary and time of day, plus entropy for gbm_ent."""
    parts = [window_summary(data["X"][rows]), (data["bar_index"][rows] / 78.0)[:, None]]
    if arch == "gbm_ent":
        parts.append(entropy_matrix(data, rows))
    return np.column_stack(parts)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_gbm(F_train: np.ndarray, y_train: np.ndarray, F_val: np.ndarray, y_val: np.ndarray,
              seed: int = 0, max_iter: int = 300):
    """Fit, then pick the tree count with the best validation log loss. Returns (model, info).

    A feature with no value in the training rows (the overnight entropy readings when no
    screen was run) is dropped; `info["keep"]` applies the same choice at prediction.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    keep = np.isfinite(F_train).any(axis=0)
    F_train, F_val = F_train[:, keep], F_val[:, keep]

    # 2% of the training rows per leaf, between 20 and 100: SPY alone has ~500 training samples
    # a fold, where a fixed 100 would forbid all but a split or two.
    leaf = int(np.clip(len(y_train) // 50, 20, 100))
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=15, min_samples_leaf=leaf,
        l2_regularization=1.0, early_stopping=False, random_state=seed,
    )
    model.fit(F_train, y_train)
    losses = [_log_loss(y_val, p[:, 1]) for p in model.staged_predict_proba(F_val)]
    best = int(np.argmin(losses))
    leaves = sum(int(tree[0].get_n_leaf_nodes()) for tree in getattr(model, "_predictors", [])[: best + 1])
    return model, {"best_epoch": best, "best_val_loss": losses[best], "n_leaves": leaves, "limit": max_iter,
                   "keep": keep}


def predict_gbm(model, F: np.ndarray, info: dict) -> np.ndarray:
    """Probability of "up" from the trees up to the best validation iteration."""
    for i, p in enumerate(model.staged_predict_proba(F[:, info["keep"]])):
        if i == info["best_epoch"]:
            return p[:, 1]
    raise ValueError(f"model has fewer than {info['best_epoch'] + 1} trees")
