"""Properties permutation entropy must satisfy. Run: python -m pytest tests/ -v"""

import sys
from math import factorial
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.entropy import (
    embed,
    multiscale_permutation_entropy,
    ordinal_patterns,
    pattern_distribution,
    permutation_entropy,
    rolling_permutation_entropy,
    statistical_complexity,
    tie_fraction,
)

RNG = np.random.default_rng(0)


def test_monotonic_series_has_zero_entropy():
    assert permutation_entropy(np.arange(500.0), m=4) == pytest.approx(0.0, abs=1e-12)


def test_white_noise_approaches_maximum_entropy():
    assert permutation_entropy(RNG.normal(size=20_000), m=4) > 0.99


def test_periodic_signal_sits_below_noise():
    sine = np.sin(np.linspace(0, 100 * np.pi, 5000))
    assert permutation_entropy(sine, m=4) < permutation_entropy(RNG.normal(size=5000), m=4)


def test_chaos_has_higher_complexity_than_noise_at_lower_entropy():
    """The property that makes the complexity plane worth computing at all."""
    x = np.zeros(10_000)
    x[0] = 0.4
    for i in range(1, x.size):
        x[i] = 3.9999 * x[i - 1] * (1 - x[i - 1])

    h_chaos, c_chaos = statistical_complexity(x, m=6)
    h_noise, c_noise = statistical_complexity(RNG.normal(size=10_000), m=6)

    assert h_chaos < h_noise
    assert c_chaos > c_noise * 5


def test_pattern_distribution_is_normalized():
    for weighted in (False, True):
        p = pattern_distribution(RNG.normal(size=2000), m=4, weighted=weighted)
        assert p.size == factorial(4)
        assert p.sum() == pytest.approx(1.0)
        assert (p >= 0).all()


def test_lehmer_codes_cover_the_full_pattern_space():
    codes, _ = ordinal_patterns(RNG.normal(size=50_000), m=4)
    assert codes.min() >= 0
    assert codes.max() == factorial(4) - 1
    assert np.unique(codes).size == factorial(4)


def test_embedding_shape_and_content():
    vectors = embed(np.arange(10.0), m=3, tau=2)
    assert vectors.shape == (6, 3)
    np.testing.assert_array_equal(vectors[0], [0.0, 2.0, 4.0])


def test_entropy_is_invariant_to_monotonic_rescaling():
    """Ordinal patterns depend on rank only, so price level must not matter."""
    x = RNG.normal(size=3000)
    assert permutation_entropy(x, m=4) == pytest.approx(permutation_entropy(x * 137.0 + 40.0, m=4))


def test_tie_fraction_flags_stale_series():
    stale = np.repeat(RNG.normal(size=100), 20)
    _, vectors = ordinal_patterns(stale, m=4)
    assert tie_fraction(vectors) > 0.9
    # the trap: a stale series reads as highly ordered
    assert permutation_entropy(stale, m=4) < 0.3


def test_jitter_breaks_ties_and_raises_entropy():
    stale = np.repeat(RNG.normal(size=200), 10)
    assert permutation_entropy(stale, m=4, tie_handling="jitter") > permutation_entropy(stale, m=4)


def test_weighted_entropy_responds_to_amplitude():
    x = RNG.normal(size=4000) * 0.01
    x[::50] *= 100.0
    assert permutation_entropy(x, m=4, weighted=True) != pytest.approx(
        permutation_entropy(x, m=4, weighted=False), abs=1e-6
    )


def test_rolling_indices_are_causal():
    ends, values = rolling_permutation_entropy(RNG.normal(size=1000), window=200, step=50, m=4)
    assert ends[0] == 199
    assert values.size == ends.size
    assert (np.diff(ends) == 50).all()


def test_rolling_rejects_windows_too_short_for_the_histogram():
    with pytest.raises(ValueError, match="too short"):
        rolling_permutation_entropy(RNG.normal(size=500), window=50, m=5)


def test_multiscale_returns_one_value_per_scale():
    out = multiscale_permutation_entropy(RNG.normal(size=20_000), scales=4, m=4)
    assert out.size == 4
    assert np.isfinite(out).all()


def test_embedding_rejects_series_shorter_than_the_embedding():
    with pytest.raises(ValueError, match="too short"):
        embed(np.arange(3.0), m=5, tau=2)
