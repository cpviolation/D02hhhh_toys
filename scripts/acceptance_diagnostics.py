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
SIZE, TAU_MAX = 1_000_000, 16.0
EDGES = np.linspace(0, TAU_MAX, 49)
CENTRES = (EDGES[1:] + EDGES[:-1]) / 2
HLT1 = {
    "two-track line (default)": "base.yaml",
    "one-track line only": "hlt1_one_track.yaml",
    "50/50 mixture of the two": "hlt1_mixture.yaml",
    "box cut, 1 track (cross-check)": "hlt1_box_a.yaml",
    "box cut, 2 tracks (cross-check)": "hlt1_box_b.yaml",
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
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    for (label, file), color in zip(HLT1.items(), CATEGORICAL, strict=False):
        acc = LHCbAcceptance(AcceptanceConfig.from_yaml(CONFIGS / file))
        accepted[label] = acc.accept(events, tau, np.random.default_rng(3))
        eff, err = binned_efficiency(tau, accepted[label], EDGES)
        ax.errorbar(
            CENTRES, 100 * eff, 100 * err, color=color, linewidth=2, label=label
        )
        print(
            f"{label:34s} overall {100 * accepted[label].mean():.2f}%  "
            f"max {100 * np.nanmax(eff):.2f}% at tau = {CENTRES[np.nanargmax(eff)]:.2f}"
        )
    # offline IPchi2 requirement on N daughters, with the default HLT1
    config_a = AcceptanceConfig.from_yaml(CONFIGS / "base.yaml")
    for n in (1, 2, 4):
        acc = LHCbAcceptance(config_a.with_overrides({"offline.n_tracks": n}))
        passed = acc.accept(events, tau, np.random.default_rng(3))
        eff, _ = binned_efficiency(tau, passed, EDGES)
        half = CENTRES[np.argmax(eff >= 0.5 * np.nanmax(eff))]
        print(
            f"default HLT1, offline IPchi2 > 4 on >= {n} daughters: overall {100 * passed.mean():.2f}%"
            f"  max {100 * np.nanmax(eff):.2f}%  half of it at tau = {half:.2f}"
        )
    eff, _ = binned_efficiency(tau, accepted[next(iter(HLT1))], EDGES)
    print(
        f"default HLT1, no offline IPchi2 cut: half of the maximum at tau = "
        f"{CENTRES[np.argmax(eff >= 0.5 * np.nanmax(eff))]:.2f}"
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
    ax.legend(
        frameon=False,
        labelcolor=TEXT,
        fontsize=8.5,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )
    fig.savefig(FIGURES / "acceptance_vs_tau.png", dpi=150)
    two_only = accepted[next(iter(HLT1))]
    either = LHCbAcceptance(
        AcceptanceConfig.from_yaml(CONFIGS / "hlt1_one_or_two_track.yaml")
    )
    default = either.accept(events, tau, np.random.default_rng(3))
    print(
        f"two-track line (default): overall {100 * two_only.mean():.2f}%;"
        f" one-or-two-track and two-track-only decisions differ on {np.mean(default != two_only):.2e} of events"
    )

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
        early, late = (
            CENTRES < 1,
            (CENTRES > 2) & (CENTRES < 6),
        )  # before the foil cut-off
        del early, late

        def pooled(b, lo, hi, bins=bins):
            """Efficiency of third ``b`` with lo <= tau < hi, from pooled counts."""
            sel = (bins == b) & (tau >= lo) & (tau < hi)
            return acc_a[sel].mean()

        print(
            f"{name:24s} ratio low/high third (pooled):"
            f" tau<1 {pooled(0, 0, 1) / pooled(2, 0, 1):.2f},"
            f" 2<tau<6 {pooled(0, 2, 6) / pooled(2, 2, 6):.2f},"
            f" 10<tau<16 {pooled(0, 10, 16) / pooled(2, 10, 16):.2f}"
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

    # 2b. the fall-off from the RF-foil cut, in thirds of the D0 pT
    acc = LHCbAcceptance()
    rng = np.random.default_rng(3)
    lab = acc.lab_event(events, tau, rng)
    passed = acc.offline(lab) & acc.inside_foil(lab) & acc.hlt1(lab, rng)
    pt_d = np.hypot(lab.d_momentum[:, 0], lab.d_momentum[:, 1])
    bins = quantile_bins(pt_d, 3)
    edges_pt = np.quantile(pt_d, [0, 1 / 3, 2 / 3, 1])
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    for b, color in enumerate(ORDINAL_BLUE):
        eff, err = binned_efficiency(tau[bins == b], passed[bins == b], EDGES)
        label = f"{edges_pt[b]:.1f} < pT(D⁰) < {edges_pt[b + 1]:.1f} GeV"
        ax.errorbar(
            CENTRES, 100 * eff, 100 * err, color=color, linewidth=2, label=label
        )
        # tau at which the radial distance reaches 3.5 mm for the bin's median pT
        median = np.median(pt_d[bins == b])
        tau_foil = 3500 / (median / 1.86484 * 122.9)
        print(
            f"pT(D0) third {b}: median {median:.2f} GeV, rho = 3.5 mm at tau = {tau_foil:.1f}; "
            f"eff at tau~4 {100 * eff[np.argmin(abs(CENTRES - 4))]:.2f}%, at tau~12 {100 * eff[np.argmin(abs(CENTRES - 12))]:.2f}%"
        )
    style_axes(ax)
    ax.set_xlabel("decay time τ = t/τ(D⁰)", color=TEXT)
    ax.set_ylabel("acceptance [%]", color=TEXT)
    ax.set_xlim(0, TAU_MAX)
    ax.set_ylim(bottom=0)
    ax.set_title(
        "Acceptance in thirds of the D⁰ pT: the RF-foil cut (ρ < 3.5 mm) removes long flights",
        color=TEXT,
        loc="left",
        fontsize=10,
    )
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9)
    fig.savefig(FIGURES / "acceptance_vs_tau_in_d0_pt_bins.png", dpi=150)

    # 3. the one-track boundary over the reconstructed toy tracks (tau from e^-tau)
    acc = LHCbAcceptance()
    rng = np.random.default_rng(4)
    lab = acc.lab_event(events[:200_000], rng.exponential(size=200_000), rng)
    good = lab.reconstructed
    line = AcceptanceConfig.from_yaml(CONFIGS / "hlt1_one_track.yaml").hlt1.lines[0]
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    ax.hist2d(
        lab.pt[good],
        np.log(lab.ipchi2[good]),
        bins=(np.linspace(0, 6, 121), np.linspace(-4, 12, 81)),
        cmap="Blues",
        cmin=1,
    )
    pt = np.linspace(line.pt_wall + 1e-3, 6, 400)
    ax.plot(
        pt,
        np.log(line.ipchi2_threshold(pt)),
        color=CATEGORICAL[1],
        linewidth=2,
        label=f"one-track: ln(IPχ²/{line.ipchi2_floor:g}) > {line.k:g} GeV/(pT − {line.pt_wall:g} GeV)",
    )
    ax.axhline(
        np.log(4),
        color=CATEGORICAL[2],
        linewidth=2,
        linestyle="--",
        label="two-track track preselection: IPχ² > 4 (and pT > 0.2 GeV)",
    )
    style_axes(ax)
    ax.set_xlabel("track pT [GeV]", color=TEXT)
    ax.set_ylabel("ln IPχ²", color=TEXT)
    ax.set_ylim(-4, 12)
    ax.set_title(
        "Reconstructed toy daughters (τ from e^−τ) and the HLT1 track requirements",
        color=TEXT,
        loc="left",
        fontsize=11,
    )
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="upper right")
    fig.savefig(FIGURES / "hlt1_one_track_boundary.png", dpi=150)
    above = lab.ipchi2[good] > line.ipchi2_threshold(lab.pt[good])
    print(f"reconstructed daughters above the one-track boundary: {above.mean():.4f}")


if __name__ == "__main__":
    main()
