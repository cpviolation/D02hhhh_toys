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


PARAMETER_LABELS = {
    "x_cp_mix": "x_CP",
    "y_cp_mix": "y_CP",
    "dx_mix": "Δx",
    "dy_mix": "Δy",
}


def _pyplot():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_shifts(summary: dict, path) -> None:
    """Paired shift of each scenario w.r.t. ``closure``, in units of σ_stat, one panel per parameter."""
    plt = _pyplot()
    shifts = summary["shifts"]
    names = list(shifts)
    fig, axes = plt.subplots(
        1, 4, figsize=(13, 0.45 * len(names) + 1.6), sharey=True, layout="constrained"
    )
    y = np.arange(len(names))[::-1]
    for ax, (param, label) in zip(axes, PARAMETER_LABELS.items(), strict=True):
        mean = np.array([shifts[s][param]["mean"] for s in names])
        error = np.array([shifts[s][param]["error"] for s in names])
        ax.errorbar(
            mean,
            y,
            xerr=error,
            fmt="o",
            color=CATEGORICAL[0],
            markersize=5,
            linewidth=2,
        )
        ax.axvline(0, color="#9c9c9c", linewidth=1)
        style_axes(ax)
        ax.set_title(label, color=TEXT, fontsize=10)
        ax.set_xlabel("shift w.r.t. closure [σ_stat]", color=TEXT)
    axes[0].set_yticks(y, names)
    fig.suptitle(
        f"{summary.get('study', '')}: paired shifts, {summary['n_toys']} toys",
        color=TEXT,
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_closure_pulls(records: list[dict], truth: dict, path) -> None:
    """Histograms of the closure pulls, with a unit Gaussian for reference."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), layout="constrained")
    edges = np.linspace(-4, 4, 17)
    ok = [r for r in records if r["results"]["closure"]["valid"]]
    for ax, (param, label) in zip(axes, PARAMETER_LABELS.items(), strict=True):
        true = truth.get(param, 0.0) / 1e-3
        pull = np.array(
            [
                (r["results"]["closure"]["values"][param] - true)
                / r["results"]["closure"]["errors"][param]
                for r in ok
            ]
        )
        ax.hist(pull, edges, color=CATEGORICAL[0], edgecolor="white", linewidth=1)
        grid = np.linspace(-4, 4, 200)
        ax.plot(
            grid,
            len(pull)
            * (edges[1] - edges[0])
            * np.exp(-(grid**2) / 2)
            / np.sqrt(2 * np.pi),
            color=CATEGORICAL[1],
            linewidth=2,
        )
        style_axes(ax)
        mean, std = (
            (pull.mean(), pull.std(ddof=1)) if len(pull) > 1 else (np.nan, np.nan)
        )
        ax.set_title(
            f"{label}: mean {mean:+.2f}, width {std:.2f}", color=TEXT, fontsize=10
        )
        ax.set_xlabel("closure pull", color=TEXT)
    fig.savefig(path, dpi=150)
    plt.close(fig)
