# Decisions

This page records every physics and design choice made in the project. The first part
lists the open questions, each with its options and their impact. The second part
lists the confirmed decisions.

- **Rule:** no physics value (cut, acceptance number, scenario size) is used for
  results until it appears here as confirmed (ground rule 3).
- **Answering:** reply with the decision ID and the option, e.g. *"P1: A"* or *"P2: B
  with 2·10⁴ per tag"*. The decision then moves to the confirmed list, with the date.

## Open decisions

None at the moment.

## Confirmed decisions

| ID | Date | Decision | Where |
|---|---|---|---|
| C1 | 2026-11-25 | Depend on `aman-examples` at tag `0.1.0`. Reference `DtoKKpipi_v2.opt` (GPL-2.0-only) through the package, never copy it. | `pyproject.toml`, [inventory](notes/example_inventory.md) |
| C2 | 2026-11-25 | Reuse the AmpGen model, the peaked phase-space generator and the TensorWaves lambdification. Write our own component matrix, CP map, time evolution, likelihood, generator, acceptance emulator and driver. | [inventory](notes/example_inventory.md) |
| C3 | 2026-11-25 | Use our own \|A_f\|²/r accept-reject sampler with a fixed, pre-estimated maximum, not TensorWaves' default generator: that generator accepts each bunch against its own maximum, which biases the sample. | `generate.py`, [conventions](notes/conventions.md) |
| C4 | 2026-11-25 | Keep the model exactly as AmpGen's. Unpaired chains are evaluated explicitly at p̄: 3 chains have no mirror, and the a₁ and K₁ chains have spline knots that differ between the charges. | `model.py`, [conventions](notes/conventions.md) |
| C5 | 2026-11-25 | Use a weighted normalisation sample. | `fit.py`, [normalisation](notes/normalisation.md) |
| C6 | 2026-11-25 | Make the normalisation density the peaked phase space (w = 1/r) by default. `flat` and `proposal` remain as cross-checks. | `normalisation.py` |
| C7 | 2026-11-25 | Apply no baseline offline IPχ² cut (`offline.n_tracks = 0`). Displacement is required at the level of the HLT1 lines. | `configs/acceptance/base.yaml` |
| C8 | 2026-11-25 | Write our own HLT1 lines in the spirit of the inclusive hadron lines. Neither Allen code nor weights are used, so no citation is needed. Using Allen directly remains an option: see [acceptance](notes/acceptance.md#using-the-real-lhcb-hlt1-code-instead). | `acceptance.py` |
| C9 | 2026-11-25 | Use the two-track line by default. The one-track line, the mixture and the box lines are available as alternative configs. | `configs/acceptance/` |
| C10 | 2026-11-25 | Require the decay vertex to be inside the RF foil: ρ < 3.5 mm from the beam line. | `decay_vertex.rho_max_um` |
| C11 | 2026-11-26 | Keep the toy D0 production spectrum and the other toy acceptance values for now. They are to be tuned on Run 3 simulation later. | `configs/acceptance/base.yaml` |
| C12 | 2026-11-26 | Use one branch and one pull request per phase, from Phase 7 on. | [CLAUDE.md](https://github.com/cpviolation/D02hhhh_toys/blob/main/CLAUDE.md) |
| C13 | 2026-11-26 | Evaluate the normalisation from decay-time moments, generated in chunks (Phase 7b), then measure v. | `moments.py`, `variance.py`, [moments](notes/moments.md) |
| C14 | 2026-11-26 | Sizes of the data/simulation differences (was P1, option A): IP resolution scale 1.05 and 0.95; two-track `track_ipchi2_min` 4 → 4.5; `production.pt_scale` 2.5 → 2.3 GeV; `tracking.width` 1.5 → 2.0 GeV. Shifts are assumed linear and can be rescaled. | `configs/study/default.yaml`, `inflated*.yaml` |
| C15 | 2026-11-26 | Normalisation size (was P2, option A): 10⁵ events per tag, n = 4·10⁷, so σ_norm ≈ 0.10σ (y_CP), 0.09σ (Δx). | `n_norm` in `configs/study/*.yaml`, [moments](notes/moments.md) |
| C16 | 2026-11-26 | Inflated cross-check (was P3, option C): run both `inflated.yaml` (x_CP = 30e-3, y_CP = 25e-3, Δx = 5e-3, Δy = −3e-3) and `inflated_no_cpv.yaml` (same, with Δx = Δy = 0). | `configs/study/inflated*.yaml` |

## Corrections to the plan

These are places where the code deliberately departs from `TOY_STUDY_PLAN.md`.

- **The y_CP sign:** a CP-even state decays as e^{−(1+y_CP)τ} in this convention,
  not Γ(1 − y_CP) as the plan says. See [conventions](notes/conventions.md).
- **The normalisation density:** the plan has flat phase space. The default is the
  peaked phase space (C6).
- **The normalisation size:** the plan asks for N_norm ≫ N_data. The measured v ≈ 3.8
  for y_CP and 3.4 for Δx make it N_norm ≈ 400 N_data for a 0.1σ offset (C15).
- **The normalisation evaluation:** the normalisation is evaluated from decay-time
  moments, not from the events. This is exact up to a truncation below 1e-15
  ([moments](notes/moments.md)).
