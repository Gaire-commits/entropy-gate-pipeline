"""Every architecture takes the same sequences, trains, and trains reproducibly."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["EGP_DEVICE"] = "cpu"

from src.features import gaf_encode
from src.models import ARCHITECTURES, build_model, count_parameters, gaf_images, needs_entropy
from src.training import predict, train_fold

BATCH, CHANNELS, LENGTH, EXTRA = 8, 4, 48, 11


def _model_and_inputs(arch):
    """Entropy architectures take the entropy features as a second input."""
    model = build_model(arch, in_channels=CHANNELS, length=LENGTH, n_extra=EXTRA)
    inputs = [torch.randn(BATCH, CHANNELS, LENGTH)] + ([torch.randn(BATCH, EXTRA)] if needs_entropy(arch) else [])
    return model, inputs


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_every_architecture_accepts_sequences(arch):
    model, inputs = _model_and_inputs(arch)
    with torch.no_grad():
        out = model.eval()(*inputs)
    assert out.shape == (BATCH, 2)


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_every_architecture_produces_gradients(arch):
    model, inputs = _model_and_inputs(arch)
    loss = torch.nn.functional.cross_entropy(model.train()(*inputs), torch.randint(0, 2, (BATCH,)))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_in_model_gaf_matches_the_numpy_encoder():
    x = np.random.default_rng(0).normal(size=(3, CHANNELS, 16)).astype(np.float32)
    torch_images = gaf_images(torch.from_numpy(x)).numpy()
    for c in range(CHANNELS):
        np.testing.assert_allclose(torch_images[:, c], gaf_encode(x[:, c]), atol=1e-5)


def test_resnet2d_downsamples_across_stages():
    model = build_model("resnet2d", in_channels=CHANNELS, length=LENGTH).eval()
    x = gaf_images(torch.randn(2, CHANNELS, LENGTH))
    sizes = []
    with torch.no_grad():
        for stage in model.body:
            x = stage(x)
            sizes.append(x.shape[-1])
    assert sizes == [48, 24, 12]


def test_cnn1d_refuses_a_window_too_short_to_pool():
    with pytest.raises(ValueError, match="needs length >= 8"):
        build_model("cnn1d", in_channels=CHANNELS, length=6)


def test_short_first_half_hour_windows_work_for_the_other_models():
    for arch in ("logreg", "resnet1d", "inceptiontime", "resnet2d"):
        model = build_model(arch, in_channels=CHANNELS, length=6).eval()
        with torch.no_grad():
            assert model(torch.randn(BATCH, CHANNELS, 6)).shape == (BATCH, 2)


def test_capacity_ladder_is_ordered():
    sizes = {a: count_parameters(build_model(a, in_channels=CHANNELS, length=LENGTH)) for a in ("logreg", "cnn1d", "resnet1d")}
    assert sizes["logreg"] < sizes["cnn1d"] < sizes["resnet1d"]


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="unknown arch"):
        build_model("transformer9000", in_channels=CHANNELS, length=LENGTH)


def _fit(seed):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, CHANNELS, 16)).astype(np.float32)
    y = (x[:, 0, -1] > 0).astype(np.int64)
    cfg = SimpleNamespace(epochs=3, patience=3, batch_size=32, lr=1e-3, weight_decay=0.0, dropout=0.2)
    model, _ = train_fold(x[:150], y[:150], x[150:], y[150:], cfg, "resnet1d", seed=seed)
    return predict(model, x[150:])


def test_same_seed_gives_identical_predictions():
    """The first real sweep moved more between identical reruns than between models."""
    np.testing.assert_array_equal(_fit(seed=3), _fit(seed=3))


def test_different_seeds_give_different_models():
    assert not np.array_equal(_fit(seed=3), _fit(seed=4))


def test_resnet1d_ent_shares_resnet1d_body_so_the_difference_is_the_entropy_branch():
    plain = build_model("resnet1d", in_channels=CHANNELS, length=LENGTH)
    with_entropy = build_model("resnet1d_ent", in_channels=CHANNELS, length=LENGTH, n_extra=EXTRA)
    body = lambda m: [tuple(p.shape) for p in m.body.parameters()]
    assert body(plain) == body(with_entropy)
    assert count_parameters(with_entropy) > count_parameters(plain)


def test_resnet1d_ent_refuses_to_run_without_entropy_features():
    with pytest.raises(ValueError, match="needs entropy features"):
        build_model("resnet1d_ent", in_channels=CHANNELS, length=LENGTH)


def _fit_with_entropy(seed, informative=True, n=800):
    """Only the entropy input carries signal; the price window is pure noise, which a
    500k-parameter body memorizes within an epoch or two."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, CHANNELS, 12)).astype(np.float32)
    e = rng.normal(size=(n, 3)).astype(np.float32)
    y = (e[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(np.int64) if informative else rng.integers(0, 2, n)
    s = int(n * 0.75)
    cfg = SimpleNamespace(epochs=8, patience=3, batch_size=128, lr=1e-3, weight_decay=1e-4, dropout=0.2)
    model, info = train_fold(x[:s], y[:s], x[s:], y[s:], cfg, "resnet1d_ent", seed=seed,
                             E_train=e[:s], E_val=e[s:])
    return predict(model, x[s:], e[s:]), y[s:]


def test_resnet1d_ent_uses_the_entropy_input_even_when_the_window_is_noise():
    """Without the warm start, early stopping froze the network before it used the entropy
    features: 0.50 accuracy where about 0.85 was available."""
    prob, y = _fit_with_entropy(seed=0)
    assert ((prob > 0.5) == (y == 1)).mean() > 0.75


def test_resnet1d_ent_finds_nothing_when_entropy_carries_nothing():
    prob, y = _fit_with_entropy(seed=0, informative=False)
    assert abs(((prob > 0.5) == (y == 1)).mean() - 0.5) < 0.1


def test_resnet1d_ent_is_reproducible():
    np.testing.assert_array_equal(_fit_with_entropy(seed=2)[0], _fit_with_entropy(seed=2)[0])
