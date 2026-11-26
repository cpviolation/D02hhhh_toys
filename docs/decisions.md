# Decisions

This page records every physics and design choice made in the project. The first part
lists the open questions, each with its options and their impact. The second part
lists the confirmed decisions.

- **Rule:** no physics value (cut, acceptance number, scenario size) is used for
  results until it appears here as confirmed (ground rule 3).
- **Answering:** reply with the decision ID and the option, e.g. *"P1: A"* or *"P2: B
  with 2·10⁴ per tag"*. The decision then moves to the confirmed list, with the date.

## Open decisions

### P1 — Sizes of the data/simulation differences

!!! question "Needed before Phase 8"
    Every override scenario in `configs/study/default.yaml` and `inflated.yaml`
    depends on these sizes.

Each scenario fits the data with an analyst's acceptance model that differs from the
true one by a single parameter.

| Scenario | Parameter | True | Proposed analyst's value |
|---|---|---|---|
| `ip_resolution_±5%` | `ip_resolution.scale` | 1.0 | 1.05 and 0.95 |
| `ipchi2_cut_+0.5` | two-track `track_ipchi2_min` | 4.0 | 4.5 |
| `softer_d0_pt` | `production.pt_scale` | 2.5 GeV | 2.3 GeV |
| `slower_tracking_turn_on` | `tracking.width` | 1.5 GeV | 2.0 GeV |

| Option | What it means |
|---|---|
| **A** (proposed) | Use the values in the table. |
| **B** | Use other values. Name them, e.g. a size you know from a Run 3 data/simulation comparison. |
| **C** | Use two sizes per effect, e.g. the proposed value and twice it, to check that the bias scales linearly. Each added size costs one more base normalisation sample and one more fit per toy, but no extra toy generation. |

- **Impact:** for small differences the shift should be roughly linear in the size of
  the difference, so a result can be rescaled to another size afterwards. C checks this
  assumption.
- **Recommendation:** A. The sizes are moderate, so the shifts stay in the linear
  regime. Rescaling covers other sizes.

### P2 — Size of the normalisation sample

!!! question "Needed before Phase 8"
    This sets the cost of the study and how precisely the closure can be tested.

- **The problem:** all toys share one normalisation sample, so its statistical
  fluctuation shifts every toy's fit by the same amount. This common offset does not
  average away over toys.
- **Its size:** σ_norm ≈ √(v·N_data/N_norm) in units of σ_stat, with v ≈ 10 for y_CP
  (measured in the smoke study; see [study notes](notes/study.md)). x_CP, Δx and Δy
  have a much smaller v.
- **Current default:** N_norm/N_data = 2·10⁶/10⁵ = 20, which gives σ_norm(y_CP) ≈ 0.7.

| Option | Setup | σ_norm(y_CP) | Cost |
|---|---|---|---|
| **A** | 10⁵ per tag, N_norm = 10⁸ (ratio 1000) | ≈ 0.1 | Per base sample: ~5 GB and ~6 CPU-h. There are 7 base samples: true, 5 overrides, none. Each fit is ~50× slower (≈ 50 min per scenario and toy), so 10 scenarios × 100 toys ≈ 800 CPU-h. |
| **B** (recommended) | 10⁴ per tag, N_norm = 10⁷ (ratio 1000) | ≈ 0.1 | Small, similar to the current default. Shifts are quoted in absolute units (1e-3), which do not depend on N_data. |
| **C** | Keep 10⁵ per tag and N_norm = 2·10⁶ | ≈ 0.7 | Current cost. σ_norm is reported and quoted as a known common offset of the closure. |

- **Effect on paired shifts:** the common random numbers make the offset almost the
  same for every scenario. It therefore largely cancels in the paired shifts of the
  `full`-mode overrides and of `factorised`. It cancels less for `ignore_time` and
  `ignore_ps`, which redraw τ or the points.
- **Recommendation:** B.
  - It tests the closure to ~0.1σ.
  - The absolute shifts are what carries over to the real analysis.
  - A could follow for one or two scenarios, if a shift turns out to depend on the
    sample size.

### P3 — Inflated mixing values for the Phase 8 cross-check

!!! info "Proposed, can wait until Phase 8 step 2"

`configs/study/inflated.yaml` uses x_CP = 30e-3, y_CP = 25e-3, Δx = 5e-3,
Δy = −3e-3. The question is whether the biases depend on the true values.

| Option | What it means |
|---|---|
| **A** (proposed) | Use these values. They are about 5–7 times the realistic x_CP and y_CP, with large CP violation, so a dependence on the truth would show. |
| **B** | Inflate x_CP and y_CP only, with Δx = Δy = 0. This keeps CP-symmetric effects separate from the CP-violation terms. |
| **C** | Run both A and B. |

**Recommendation:** C.
- In the smoke study, Δx and Δy moved under CP-symmetric mis-modellings only because
  the true Δx and Δy were non-zero.
- B isolates the effect of the true x_CP and y_CP. A tests the CP-violation terms.

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

## Corrections to the plan

These are places where the code deliberately departs from `TOY_STUDY_PLAN.md`.

- **The y_CP sign:** a CP-even state decays as e^{−(1+y_CP)τ} in this convention,
  not Γ(1 − y_CP) as the plan says. See [conventions](notes/conventions.md).
- **The normalisation density:** the plan has flat phase space. The default is the
  peaked phase space (C6).
- **The normalisation size:** the plan asks for N_norm ≫ N_data. The measured v ≈ 10
  for y_CP makes the requirement stricter (P2).
