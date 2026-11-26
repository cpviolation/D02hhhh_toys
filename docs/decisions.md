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

!!! question "Needed before Phase 8 (revised on 2026-11-26 after Phase 7b)"
    This sets how precisely the closure can be tested. Since Phase 7b, a large sample
    is cheap: see [moments and v](notes/moments.md).

- **The problem:** all toys share one normalisation sample, so its statistical
  fluctuation shifts every toy's fit by the same amount. This common offset does not
  average away over toys.
- **Its size:** σ_norm = √(v·N/n) in units of σ_stat, for N data events per tag and n
  normalisation events.
- **v on the full model:** 3.8 for y_CP, 3.4 for Δx, 0.4 for Δy and ≈ 0 for x_CP. This
  is predicted and validated against fits. The earlier "v ≈ 10" came from the 5-chain
  smoke model.
- **Current default:** n/N = 20, which gives σ_norm(y_CP) ≈ 0.43.
- **Cost now:**
  - the fits do not depend on n, and nothing per event is stored;
  - generation costs 0.126 ms per normalisation event, and is spread over all workers.

| Option | Setup | σ_norm(y_CP), σ_norm(Δx) | Normalisation cost (7 base samples) |
|---|---|---|---|
| **A** (recommended) | 10⁵ per tag, n = 4·10⁷ | 0.10, 0.09 | 10 CPU-h, ~15 min on 46 cores |
| **B** | 10⁵ per tag, n = 10⁸ | 0.06, 0.06 | 25 CPU-h, ~35 min on 46 cores |
| **C** | Keep n = 2·10⁶ | 0.43, 0.41 | negligible; quote σ_norm as a known offset |

- **Effect on paired shifts:** the common random numbers make the offset almost the
  same for every scenario, so it largely cancels in the paired shifts. It cancels less
  for `ignore_time` and `ignore_ps`.
- **Recommendation:** A. The cost is small next to the toy generation, which takes
  about an hour for 100 toys on 46 cores. B is cheap enough if the closure has to be
  tested more precisely. The earlier option of smaller toys (10⁴ per tag) is no longer
  needed.

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
| C13 | 2026-11-26 | Evaluate the normalisation from decay-time moments, generated in chunks (Phase 7b), then measure v. | `moments.py`, `variance.py`, [moments](notes/moments.md) |

## Corrections to the plan

These are places where the code deliberately departs from `TOY_STUDY_PLAN.md`.

- **The y_CP sign:** a CP-even state decays as e^{−(1+y_CP)τ} in this convention,
  not Γ(1 − y_CP) as the plan says. See [conventions](notes/conventions.md).
- **The normalisation density:** the plan has flat phase space. The default is the
  peaked phase space (C6).
- **The normalisation size:** the plan asks for N_norm ≫ N_data. The measured v ≈ 3.8
  for y_CP and 3.4 for Δx make it N_norm ≈ 400 N_data for a 0.1σ offset (P2).
- **The normalisation evaluation:** the normalisation is evaluated from decay-time
  moments, not from the events. This is exact up to a truncation below 1e-15
  ([moments](notes/moments.md)).
