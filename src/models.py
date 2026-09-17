"""Model zoo for the feature-extraction experiment.

All models take (batch, channels, length) and return logits over 2 classes.
ResNet2D builds its Gramian Angular Field images inside the forward pass, so it
consumes the same sequences as everything else and the image stack never has to
exist in memory for the whole dataset at once.

Global pooling is a plain mean rather than adaptive pooling: the adaptive
pooling backward pass has no deterministic GPU implementation, and a model
whose results move between identical runs cannot be compared to its neighbours.

The lineup is deliberately a ladder, not a pile: a linear model that uses no
representation learning at all, then three architectures of increasing
capacity. H3 ("does deep feature extraction beat engineered features?") is only
answerable if the bottom rung is actually run.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GlobalAvgPool(nn.Module):
    """Mean over every dimension after (batch, channels)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).mean(dim=2)


def _head(c_in: int, n_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(GlobalAvgPool(), nn.Dropout(dropout), nn.Linear(c_in, n_classes))


def gaf_images(x: torch.Tensor) -> torch.Tensor:
    """(batch, channels, L) -> (batch, channels, L, L) summation GAF, per window and channel."""
    lo = x.amin(dim=2, keepdim=True)
    hi = x.amax(dim=2, keepdim=True)
    span = torch.where(hi - lo > 0, hi - lo, torch.ones_like(hi))
    s = torch.clamp(2 * (x - lo) / span - 1, -1.0, 1.0)
    c = torch.sqrt(torch.clamp(1 - s**2, min=0.0))
    return s.unsqueeze(3) * s.unsqueeze(2) - c.unsqueeze(3) * c.unsqueeze(2)


class LogisticBaseline(nn.Module):
    """Flattened window into a linear layer. The control, not a contender."""

    def __init__(self, in_channels: int, length: int, n_classes: int = 2, **_):
        super().__init__()
        self.net = nn.Linear(in_channels * length, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


class CNN1D(nn.Module):
    """Plain stacked convolutions — the simplest thing that learns a filter bank."""

    def __init__(
        self,
        in_channels: int,
        length: int,
        n_classes: int = 2,
        channels: tuple[int, ...] = (64, 128, 128),
        kernel_size: int = 7,
        dropout: float = 0.2,
        **_,
    ):
        super().__init__()
        if length < 2 ** len(channels):
            raise ValueError(
                f"cnn1d halves the sequence {len(channels)} times and needs length >= "
                f"{2 ** len(channels)}; got {length}"
            )
        layers, c_in = [], in_channels
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size, padding="same"),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            c_in = c_out
        self.body = nn.Sequential(*layers)
        self.head = _head(c_in, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class _ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernels: tuple[int, ...] = (8, 5, 3)):
        super().__init__()
        convs, c_in = [], in_ch
        for k in kernels:
            convs += [nn.Conv1d(c_in, out_ch, k, padding="same"), nn.BatchNorm1d(out_ch), nn.ReLU()]
            c_in = out_ch
        convs = convs[:-1]
        self.convs = nn.Sequential(*convs)
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.convs(x) + self.shortcut(x))


class ResNet1D(nn.Module):
    """Wang et al. (2017) time-series ResNet: 3 residual blocks, GAP head."""

    def __init__(
        self,
        in_channels: int,
        length: int,
        n_classes: int = 2,
        filters: tuple[int, ...] = (64, 128, 128),
        dropout: float = 0.2,
        **_,
    ):
        super().__init__()
        blocks, c_in = [], in_channels
        for f in filters:
            blocks.append(_ResidualBlock1D(c_in, f))
            c_in = f
        self.body = nn.Sequential(*blocks)
        self.head = _head(c_in, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class _InceptionModule(nn.Module):
    def __init__(self, in_ch: int, n_filters: int = 32, kernels: tuple[int, ...] = (39, 19, 9)):
        super().__init__()
        self.bottleneck = (
            nn.Conv1d(in_ch, n_filters, 1, bias=False) if in_ch > 1 else nn.Identity()
        )
        bott_ch = n_filters if in_ch > 1 else in_ch
        self.convs = nn.ModuleList(
            [nn.Conv1d(bott_ch, n_filters, k, padding="same", bias=False) for k in kernels]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_ch, n_filters, 1, bias=False)
        )
        self.norm = nn.Sequential(nn.BatchNorm1d(n_filters * (len(kernels) + 1)), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bottleneck(x)
        branches = [conv(b) for conv in self.convs] + [self.pool_branch(x)]
        return self.norm(torch.cat(branches, dim=1))


class InceptionTime(nn.Module):
    """Ismail Fawaz et al. (2020).

    Parallel kernels of several widths in every module, so one layer sees both
    fast and slow structure. That is the property worth having here: intraday
    momentum and the slower drift it rides on do not share a timescale, and a
    single fixed kernel width has to pick one.
    """

    def __init__(
        self,
        in_channels: int,
        length: int,
        n_classes: int = 2,
        n_filters: int = 32,
        depth: int = 6,
        dropout: float = 0.2,
        **_,
    ):
        super().__init__()
        self.depth = depth
        self.modules_ = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        c_in = in_channels
        out_ch = n_filters * 4
        for d in range(depth):
            self.modules_.append(_InceptionModule(c_in, n_filters))
            if d % 3 == 2:
                res_in = in_channels if d == 2 else out_ch
                self.shortcuts.append(
                    nn.Sequential(nn.Conv1d(res_in, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch))
                )
            c_in = out_ch
        self.act = nn.ReLU()
        self.head = _head(out_ch, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        for d, module in enumerate(self.modules_):
            x = module(x)
            if d % 3 == 2:
                x = self.act(x + self.shortcuts[d // 3](res))
                res = x
        return self.head(x)


class _ResidualBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch)
            )
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.convs(x) + self.shortcut(x))


class ResNet2D(nn.Module):
    """ResNet over Gramian Angular Field images built from the input sequences.

    Deliberately a structural twin of ResNet1D -- same three residual stages,
    same 64/128/128 filter progression -- so a difference in results reads as
    the sequence-vs-image framing rather than a difference in depth or capacity.

    No stride-2 stem or input maxpool: a 48x48 encoding is small enough that the
    usual ImageNet downsampling would discard most of it before the first block.
    """

    def __init__(
        self,
        in_channels: int,
        length: int,
        n_classes: int = 2,
        filters: tuple[int, ...] = (64, 128, 128),
        dropout: float = 0.2,
        **_,
    ):
        super().__init__()
        blocks, c_in = [], in_channels
        for i, f in enumerate(filters):
            blocks.append(_ResidualBlock2D(c_in, f, stride=1 if i == 0 else 2))
            c_in = f
        self.body = nn.Sequential(*blocks)
        self.head = _head(c_in, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(gaf_images(x)))


ARCHITECTURES = {
    "logreg": LogisticBaseline,
    "cnn1d": CNN1D,
    "resnet1d": ResNet1D,
    "inceptiontime": InceptionTime,
    "resnet2d": ResNet2D,
}

def build_model(arch: str, in_channels: int, length: int, **kwargs) -> nn.Module:
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch '{arch}'; options: {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch](in_channels=in_channels, length=length, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
