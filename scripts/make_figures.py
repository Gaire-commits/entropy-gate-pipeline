#!/usr/bin/env python3
"""Thesis figures, computed with the pipeline's own code.

    python scripts/make_figures.py        # -> figures/ordinal_patterns.{png,svg}
"""

from __future__ import annotations

from itertools import permutations
from math import factorial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from _common import ROOT

from src.entropy import _row_entropy, batched_pattern_distributions, ordinal_patterns, surrogate_test
from src.synthetic import ar_returns

M, N, K = 4, 390, 99
PATTERNS = factorial(M)
ACCENT, MUTED, RULE = "#2C5B85", "#B4BCC6", "#5C6674"
TREND_C, ZIG_C = "#1F7A5C", "#B23B2E"

plt.rcParams.update({
    "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 8.5, "axes.titlesize": 9.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "figure.dpi": 200,
})


def pattern_labels(m: int) -> list[str]:
    """Lehmer code -> the pattern written out: '1234' always rises, '4321' always falls."""
    labels = [""] * factorial(m)
    for order in permutations(range(m)):
        code = sum(
            sum(1 for later in order[i + 1:] if later < order[i]) * factorial(m - 1 - i)
            for i in range(m)
        )
        labels[code] = "".join(str(o + 1) for o in order)
    return labels


def panel_windows(ax, returns, labels, start=0):
    r = returns[start:start + 12] * 1e4
    codes, _ = ordinal_patterns(returns[start:start + 12], M, 1)
    low, high = r.min(), r.max()

    ax.axhline(0, color=RULE, lw=0.6, zorder=1)
    for start in (0, 4, 8):
        ax.add_patch(Rectangle((start - 0.4, low - 4), 3.8, high - low + 8, facecolor=MUTED,
                               alpha=0.3, edgecolor="none", zorder=0))
        ax.annotate(labels[codes[start]], (start + 1.5, low - 6), ha="center", va="top",
                    fontsize=9.5, family="monospace", color=ACCENT, weight="bold")
    ax.plot(range(12), r, "-o", color=ACCENT, ms=3.5, lw=1.2, zorder=3)

    ax.set_xticks(range(12))
    ax.set_xticklabels(range(start, start + 12))
    ax.set_xlabel("5-minute bar")
    ax.set_ylabel("return (bps)")
    ax.set_ylim(low - 13, high + 4)
    ax.set_title("(a)  each 4-bar window is replaced by the order of its returns", loc="left")


def panel_histogram(ax, returns, labels, color, title):
    rng = np.random.default_rng(0)
    stats = surrogate_test(returns, M, 1, K, rng=np.random.default_rng(0))
    shuffled = rng.permuted(np.tile(returns, (K, 1)), axis=1)
    real, _ = batched_pattern_distributions(returns[None, :], M, 1)
    sur, _ = batched_pattern_distributions(shuffled, M, 1)

    x = np.arange(PATTERNS)
    mono = np.zeros(PATTERNS, dtype=bool)
    mono[[0, PATTERNS - 1]] = True
    lo, hi = np.percentile(sur, 2.5, axis=0), np.percentile(sur, 97.5, axis=0)

    ax.bar(x[~mono], real[0][~mono] * 100, color=MUTED, width=0.74, label="this series")
    ax.bar(x[mono], real[0][mono] * 100, color=color, width=0.74, label="steady runs (1234, 4321): trending")
    ax.vlines(x, lo * 100, hi * 100, color=RULE, lw=0.9)
    ax.plot(x, sur.mean(0) * 100, "_", color="black", ms=7, mew=1.2,
            label="same returns shuffled (mean, 95% range)")

    shuffle_pe = _row_entropy(sur, M).mean()
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, family="monospace")
    ax.set_xlim(-0.8, PATTERNS - 0.2)
    ax.set_ylabel("share of windows (%)")
    ax.set_title(title, loc="left")
    ax.set_ylim(0, max(real[0].max(), hi.max()) * 145)
    ax.annotate(
        f"entropy {stats['pe_unweighted']:.3f}, shuffled {shuffle_pe:.3f}"
        f"   >  entropy test p = {stats['p_unweighted']:.2f}\n"
        f"steady runs {stats['monotone_share'] * 100:.1f}%, shuffled "
        f"{(stats['monotone_share'] - stats['monotone_excess']) * 100:.1f}%"
        f"   >  trend test p = {stats['p_trend']:.2f}",
        (0.5, 0.97), xycoords="axes fraction", ha="center", va="top", linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=RULE, lw=0.6),
    )
    return stats


def main() -> int:
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    labels = pattern_labels(M)
    sessions = N // 78 + 1
    trending = ar_returns(sessions, +0.35, seed=1)[:N]
    # phi tuned so both series land on the same entropy: only the trend test tells them apart
    zigzag = ar_returns(sessions, -0.45, seed=2)[:N]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.8), height_ratios=[1.0, 1.35, 1.35])
    panel_windows(axes[0], trending, labels, start=67)
    a = panel_histogram(axes[1], trending, labels, TREND_C,
                        "(b)  a trending series: steady runs beat the shuffles, so both tests fire")
    b = panel_histogram(axes[2], zigzag, labels, ZIG_C,
                        "(c)  a zig-zag series: the same entropy, but no steady runs, so only entropy fires")

    handles, legend_labels = axes[1].get_legend_handles_labels()
    handles.append(axes[2].containers[1])
    legend_labels.append("steady runs: zig-zag")
    fig.legend(handles, legend_labels, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(h_pad=2.2)
    for ext in ("png", "svg"):
        fig.savefig(out / f"ordinal_patterns.{ext}", bbox_inches="tight")

    print(f"trending: entropy {a['pe_unweighted']:.3f}  runs {a['monotone_share']:.3f}  p_trend {a['p_trend']:.2f}")
    print(f"zigzag  : entropy {b['pe_unweighted']:.3f}  runs {b['monotone_share']:.3f}  p_trend {b['p_trend']:.2f}")
    print("wrote", out / "ordinal_patterns.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
