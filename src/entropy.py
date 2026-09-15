"""Ordinal-pattern complexity measures for predictability screening.

Bandt & Pompe (2002) permutation entropy, the Fadlallah et al. (2013) weighted
variant, and the Rosso et al. (2007) complexity-entropy plane.
"""

from __future__ import annotations

from math import factorial, log

import numpy as np

_FACTORIAL = np.array([factorial(i) for i in range(13)], dtype=np.int64)


def embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Delay-embed a 1-D series into (n_vectors, m) takens vectors."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size - (m - 1) * tau
    if n <= 0:
        raise ValueError(f"series of length {x.size} too short for m={m}, tau={tau}")
    idx = np.arange(n)[:, None] + np.arange(m)[None, :] * tau
    return x[idx]


def tie_fraction(vectors: np.ndarray) -> float:
    """Share of embedding vectors containing at least one repeated value.

    Ties are resolved by index order, which biases the pattern histogram toward
    the identity permutation and makes a stale series look artificially
    predictable. Flat quote runs in illiquid names are the usual cause.
    """
    if vectors.size == 0:
        return 0.0
    sorted_v = np.sort(vectors, axis=1)
    has_tie = (np.diff(sorted_v, axis=1) == 0).any(axis=1)
    return float(has_tie.mean())


def ordinal_patterns(
    x: np.ndarray,
    m: int = 4,
    tau: int = 1,
    tie_handling: str = "stable",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode each embedding vector as a Lehmer code in [0, m!).

    Returns (codes, vectors).
    """
    if not 2 <= m <= 12:
        raise ValueError(f"embedding_dim must be in [2, 12], got {m}")
    vectors = embed(x, m, tau)

    if tie_handling == "jitter":
        rng = rng or np.random.default_rng(0)
        scale = np.abs(vectors).max() or 1.0
        vectors = vectors + rng.normal(0.0, 1e-10 * scale, vectors.shape)
    elif tie_handling != "stable":
        raise ValueError(f"unknown tie_handling: {tie_handling}")

    order = np.argsort(vectors, axis=1, kind="stable")
    codes = np.zeros(order.shape[0], dtype=np.int64)
    for i in range(m - 1):
        smaller = (order[:, i + 1 :] < order[:, i : i + 1]).sum(axis=1)
        codes += smaller * _FACTORIAL[m - 1 - i]
    return codes, vectors


def pattern_distribution(
    x: np.ndarray,
    m: int = 4,
    tau: int = 1,
    weighted: bool = False,
    tie_handling: str = "stable",
) -> np.ndarray:
    """Relative frequency of each of the m! ordinal patterns.

    Weighted mode scales each occurrence by the variance of its embedding
    vector, so large moves dominate microstructure noise.
    """
    codes, vectors = ordinal_patterns(x, m, tau, tie_handling)
    n_patterns = _FACTORIAL[m]
    if weighted:
        w = vectors.var(axis=1)
        total = w.sum()
        if total <= 0:
            return np.full(n_patterns, 1.0 / n_patterns)
        counts = np.bincount(codes, weights=w, minlength=n_patterns)
        return counts / total
    counts = np.bincount(codes, minlength=n_patterns).astype(np.float64)
    return counts / counts.sum()


def _shannon(p: np.ndarray) -> float:
    nz = p[p > 0]
    h = -(nz * np.log(nz)).sum()
    return float(h) if h != 0 else 0.0


def permutation_entropy(
    x: np.ndarray,
    m: int = 4,
    tau: int = 1,
    weighted: bool = False,
    normalize: bool = True,
    tie_handling: str = "stable",
) -> float:
    """Permutation entropy. Normalized to [0, 1]: 0 = fully ordered, 1 = random."""
    p = pattern_distribution(x, m, tau, weighted, tie_handling)
    h = _shannon(p)
    return h / log(_FACTORIAL[m]) if normalize else h


def statistical_complexity(
    x: np.ndarray,
    m: int = 4,
    tau: int = 1,
    weighted: bool = False,
    tie_handling: str = "stable",
) -> tuple[float, float]:
    """Point on the complexity-entropy plane: (normalized PE, Jensen-Shannon complexity).

    Complexity peaks at intermediate entropy. Two series with identical
    entropy can sit at different complexity, which is what separates
    structured-but-noisy dynamics from plain stochastic noise.
    """
    p = pattern_distribution(x, m, tau, weighted, tie_handling)
    n = float(_FACTORIAL[m])
    p_uniform = np.full(int(n), 1.0 / n)

    h_norm = _shannon(p) / log(n)
    js = _shannon(0.5 * (p + p_uniform)) - 0.5 * _shannon(p) - 0.5 * _shannon(p_uniform)
    q0 = -2.0 / (((n + 1) / n) * log(n + 1) - 2 * log(2 * n) + log(n))
    return h_norm, float(q0 * js * h_norm)


def rolling_permutation_entropy(
    x: np.ndarray,
    window: int,
    step: int = 1,
    m: int = 4,
    tau: int = 1,
    weighted: bool = False,
    tie_handling: str = "stable",
) -> tuple[np.ndarray, np.ndarray]:
    """PE over sliding windows.

    Returns (end_indices, pe_values) where end_index is the last bar included,
    so a value is only ever attached to data available at that bar.
    """
    x = np.asarray(x, dtype=np.float64)
    if window < _FACTORIAL[m] * 5:
        raise ValueError(
            f"window={window} too short for m={m}: need >> {_FACTORIAL[m]} samples "
            "for a meaningful pattern histogram"
        )
    ends, values = [], []
    for start in range(0, x.size - window + 1, step):
        seg = x[start : start + window]
        values.append(permutation_entropy(seg, m, tau, weighted, True, tie_handling))
        ends.append(start + window - 1)
    return np.asarray(ends), np.asarray(values)


def multiscale_permutation_entropy(
    x: np.ndarray,
    scales: int = 5,
    m: int = 4,
    tau: int = 1,
    weighted: bool = False,
) -> np.ndarray:
    """PE of coarse-grained series at scales 1..scales.

    A name can be noisy bar-to-bar but ordered at a coarser horizon; the curve
    shows which horizon, if any, carries the structure.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.full(scales, np.nan)
    for s in range(1, scales + 1):
        n = x.size // s
        if n < _FACTORIAL[m] * 5:
            break
        coarse = x[: n * s].reshape(n, s).mean(axis=1)
        out[s - 1] = permutation_entropy(coarse, m, tau, weighted)
    return out
