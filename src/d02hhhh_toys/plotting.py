"""Binned efficiencies and plotting helpers (matplotlib is imported lazily)."""

from __future__ import annotations

from typing import Any

import numpy as np

#: categorical slots (fixed order) and an ordinal blue ramp
CATEGORICAL = ("#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17")
ORDINAL_BLUE = ("#84b9f3", "#257de0", "#0f4a96")
TEXT, TEXT_SECONDARY, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def binned_efficiency(
    x: np.ndarray, accepted: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of ``accepted`` events in each bin of ``x``, and its binomial uncertainty."""
    total, _ = np.histogram(x, edges)
    passed, _ = np.histogram(x[accepted], edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = passed / total
        err = np.sqrt(eff * (1 - eff) / total)
    return eff, err


def quantile_bins(x: np.ndarray, n: int) -> np.ndarray:
    """Index ``0..n-1`` of the quantile bin of each value of ``x``."""
    edges = np.quantile(x, np.linspace(0, 1, n + 1))
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n - 1)


def style_axes(ax: Any) -> None:
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT)
