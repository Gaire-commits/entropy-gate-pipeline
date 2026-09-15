"""Every architecture must accept the shape its encoding produces, and refuse the
other one loudly rather than failing deep inside a conv."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import ARCHITECTURES, IMAGE_ARCHITECTURES, build_model, count_parameters, expects_images
from src.training import train_fold

BATCH, CHANNELS, LENGTH = 8, 4, 48
SEQUENCE_ARCHS = sorted(set(ARCHITECTURES) - IMAGE_ARCHITECTURES)


@pytest.mark.parametrize("arch", SEQUENCE_ARCHS)
def test_sequence_architectures_accept_sequences(arch):
    model = build_model(arch, in_channels=CHANNELS, length=LENGTH).eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, CHANNELS, LENGTH))
    assert out.shape == (BATCH, 2)


def test_resnet2d_accepts_gaf_shaped_images():
    model = build_model("resnet2d", in_channels=CHANNELS, length=LENGTH).eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, CHANNELS, LENGTH, LENGTH))
    assert out.shape == (BATCH, 2)


def test_resnet2d_downsamples_across_stages():
    model = build_model("resnet2d", in_channels=CHANNELS, length=LENGTH).eval()
    x = torch.randn(2, CHANNELS, LENGTH, LENGTH)
    sizes = []
    with torch.no_grad():
        for stage in model.body:
            x = stage(x)
            sizes.append(x.shape[-1])
    assert sizes == [48, 24, 12]


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_every_architecture_produces_gradients(arch):
    shape = (BATCH, CHANNELS, LENGTH, LENGTH) if expects_images(arch) else (BATCH, CHANNELS, LENGTH)
    model = build_model(arch, in_channels=CHANNELS, length=LENGTH).train()
    loss = torch.nn.functional.cross_entropy(model(torch.randn(*shape)), torch.randint(0, 2, (BATCH,)))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_capacity_ladder_is_ordered():
    """The comparison is only meaningful if the baseline is genuinely smaller."""
    sizes = {
        arch: count_parameters(build_model(arch, in_channels=CHANNELS, length=LENGTH))
        for arch in ["logreg", "cnn1d", "resnet1d"]
    }
    assert sizes["logreg"] < sizes["cnn1d"] < sizes["resnet1d"]


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="unknown arch"):
        build_model("transformer9000", in_channels=CHANNELS, length=LENGTH)


def _cfg(arch):
    return SimpleNamespace(
        arch=arch, epochs=1, batch_size=8, lr=1e-3, weight_decay=0.0, dropout=0.2, seed=0
    )


def test_image_architecture_refuses_sequence_data():
    x = np.random.randn(16, CHANNELS, LENGTH).astype(np.float32)
    y = np.random.randint(0, 2, 16)
    with pytest.raises(ValueError, match="expects GAF/MTF images"):
        train_fold(x, y, x, y, _cfg("resnet2d"))


def test_sequence_architecture_refuses_image_data():
    x = np.random.randn(16, CHANNELS, LENGTH, LENGTH).astype(np.float32)
    y = np.random.randint(0, 2, 16)
    with pytest.raises(ValueError, match="expects 1-D sequences"):
        train_fold(x, y, x, y, _cfg("resnet1d"))
