"""Phase 5 diagnostics: acceptance versus decay time, and its correlation with phase space.

Writes docs/notes/figures/acceptance_*.png and prints the numbers quoted in
docs/notes/acceptance.md. Events are flat four-body phase space with decay times uniform
in [0, 8), so the binned efficiencies are the acceptance itself.

    .venv/bin/python scripts/acceptance_diagnostics.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from aman_examples import d02kkpipi
from tensorwaves.data.rng import NumpyUniformRNG

from d02hhhh_toys.acceptance import AcceptanceConfig, LHCbAcceptance
from d02hhhh_toys.model import from_sample
from d02hhhh_toys.plotting import (
    CATEGORICAL,
    ORDINAL_BLUE,
    TEXT,
    binned_efficiency,
    quantile_bins,
    style_axes,
)

ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs" / "acceptance"
FIGURES = ROOT / "docs" / "notes" / "figures"
SIZE, TAU_MAX = 1_000_000, 8.0
EDGES = np.linspace(0, TAU_MAX, 33)
CENTRES = (EDGES[1:] + EDGES[:-1]) / 2
HLT1 = {
    "(a) 1 track, pT > 1 GeV, IPχ² > 7.4": "hlt1_a.yaml",
    "(b) 2 tracks, pT > 0.5 GeV, IPχ² > 4": "hlt1_b.yaml",
    "(a) or (b)": "hlt1_a_or_b.yaml",
    "50/50 mixture of (a) and (b)": "hlt1_mixture.yaml",
}


def mass(events, indices):
    p = events[:, list(indices)].sum(axis=1)
    return np.sqrt(p[:, 3] ** 2 - np.sum(p[:, :3] ** 2, axis=1))


def main() -> None:
    model = d02kkpipi.formulate_model()
    events = from_sample(
        d02kkpipi.phase_space_generator(model).generate(SIZE, NumpyUniformRNG(seed=1))
    )
    tau = np.random.default_rng(2).uniform(0, TAU_MAX, SIZE)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # 1. acceptance versus decay time, per HLT1 configuration
    accepted = {}
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    for (label, file), color in zip(HLT1.items(), CATEGORICAL, strict=False):
        acc = LHCbAcceptance(AcceptanceConfig.from_yaml(CONFIGS / file))
        accepted[label] = acc.accept(events, tau, np.random.default_rng(3))
        eff, err = binned_efficiency(tau, accepted[label], EDGES)
        ax.errorbar(
            CENTRES, 100 * eff, 100 * err, color=color, linewidth=2, label=label
        )
        print(
            f"{label:40s} overall {100 * accepted[label].mean():.2f}%  "
            f"max {100 * np.nanmax(eff):.2f}% at tau = {CENTRES[np.nanargmax(eff)]:.2f}"
        )
    # offline IPchi2 requirement on N daughters, with HLT1 (a)
    config_a = AcceptanceConfig.from_yaml(CONFIGS / "hlt1_a.yaml")
    for n in (1, 2, 4):
        acc = LHCbAcceptance(config_a.with_overrides({"offline.n_tracks": n}))
        passed = acc.accept(events, tau, np.random.default_rng(3))
        eff, _ = binned_efficiency(tau, passed, EDGES)
        half = CENTRES[np.argmax(eff >= 0.5 * eff[-1])]
        print(
            f"HLT1 (a), offline IPchi2 > 4 on >= {n} daughters: overall {100 * passed.mean():.2f}%"
            f"  last bin {100 * eff[-1]:.2f}%  half of it at tau = {half:.2f}"
        )
    eff, _ = binned_efficiency(tau, accepted[next(iter(HLT1))], EDGES)
    print(
        f"HLT1 (a), no offline IPchi2 cut: half of the last bin at tau = "
        f"{CENTRES[np.argmax(eff >= 0.5 * eff[-1])]:.2f}"
    )
    style_axes(ax)
    ax.set_xlabel("decay time τ = t/τ(D⁰)", color=TEXT)
    ax.set_ylabel("acceptance [%]", color=TEXT)
    ax.set_title(
        "Acceptance versus decay time, by HLT1 configuration (toy values)",
        color=TEXT,
        loc="left",
        fontsize=11,
    )
    ax.set_xlim(0, TAU_MAX)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9)
    fig.savefig(FIGURES / "acceptance_vs_tau.png", dpi=150)

    # 2. the same curve in bins of phase-space variables, relative to the average
    label = next(iter(HLT1))
    acc_a = accepted[label]
    eff_all, _ = binned_efficiency(tau, acc_a, EDGES)
    variables = {
        "m(K⁺K⁻)": mass(events, (0, 1)),
        "m(π⁺π⁻)": mass(events, (2, 3)),
        "m(K⁺π⁻)": mass(events, (0, 3)),
        "min p* of the daughters": np.min(
            np.linalg.norm(events[..., :3], axis=-1), axis=1
        ),
    }
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True, layout="constrained")
    for ax, (name, values) in zip(axes, variables.items(), strict=True):
        bins = quantile_bins(values, 3)
        spread = []
        for b, (color, tag) in enumerate(
            zip(ORDINAL_BLUE, ("low", "middle", "high"), strict=True)
        ):
            eff, err = binned_efficiency(tau[bins == b], acc_a[bins == b], EDGES)
            ratio, ratio_err = eff / eff_all, err / eff_all
            ax.errorbar(
                CENTRES,
                ratio,
                ratio_err,
                color=color,
                linewidth=2,
                label=f"{tag} third",
            )
            spread.append(ratio)
        spread = np.array(spread)
        early, late = CENTRES < 1, (CENTRES > 2) & (CENTRES < 6)
        print(
            f"{name:24s} ratio low/high third: tau<1 {np.nanmean(spread[0, early] / spread[2, early]):.2f}, "
            f"2<tau<6 {np.nanmean(spread[0, late] / spread[2, late]):.2f}"
        )
        style_axes(ax)
        ax.axhline(1, color="#9c9c9c", linewidth=1)
        ax.set_xlabel("τ", color=TEXT)
        ax.set_title(name, color=TEXT, fontsize=10)
        ax.set_xlim(0, TAU_MAX)
    axes[0].set_ylabel("ε(τ | bin) / ε(τ)", color=TEXT)
    axes[0].legend(
        frameon=False, labelcolor=TEXT, fontsize=9, title="bin of the variable"
    )
    fig.suptitle(
        f"Acceptance in thirds of phase-space variables, relative to the average: HLT1 {label}",
        color=TEXT,
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.savefig(FIGURES / "acceptance_vs_tau_in_phase_space_bins.png", dpi=150)


if __name__ == "__main__":
    main()
