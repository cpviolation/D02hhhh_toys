# Decisions

This page records every physics and design choice made in the project. The first part
lists the open questions, each with its options and their impact. The second part
lists the confirmed decisions.

- **Rule:** no physics value (cut, acceptance number, scenario size) is used for
  results until it appears here as confirmed (ground rule 3).
- **Answering:** reply with the decision ID and the option, e.g. *"P1: A"* or *"P2: B
  with 2·10⁴ per tag"*. The decision then moves to the confirmed list, with the date.

## Open decisions

The next phase (Phase 9, charge-asymmetric acceptance) needs P4–P6.

**Why only the momentum dependence matters:** every event has a K⁺ and a K⁻ (and a π⁺
and a π⁻). With per-charge efficiencies ε±(p) = ε(p)(1 ± A(p)/2), the efficiency of an
event is a product over its tracks. A constant A multiplies every event by the same
(1 + A/2)(1 − A/2), which the fit normalises away. Only the change of A with momentum
makes the acceptance differ between a phase-space point p and its CP conjugate p̄.
That difference is what can fake CP violation, i.e. move Δx and Δy.

### P4 — Which asymmetries to model

| Option | Sources | Code |
|---|---|---|
| **A** | Kaon detection asymmetry A_K(p) only: K⁻ interacts more in the material than K⁺. It is the dominant source at LHCb. | Small: a per-charge factor on the tracking efficiency. One normalisation sample still serves both tags, because the acceptance depends only on the final-state momenta. |
| **B** (recommended) | A, plus the smaller pion detection asymmetry A_π(p). | As A. |
| **C** | B, plus a flavour-tag or production asymmetry that depends on the D0 kinematics (pT, η), e.g. from the slow pion of D*± → D0π±. | Larger: the acceptance then depends on the flavour, so each tag needs its own normalisation sample. A constant tag asymmetry has no effect, because each tag is normalised separately; only its correlation with the decay-time acceptance (through the D0 pT) can matter. |

- **Recommendation:** B. C is a separate mechanism, through the decay-time acceptance
  rather than the phase space. It could be a later phase.
- **Not proposed:** left-right asymmetries from the magnet. They change sign with the
  polarity, so they cancel at first order in the sum of both polarities.

### P5 — Form and size of the asymmetries

Proposed form, per track of momentum p (in GeV), applied to the tracking efficiency:

    A_h(p) = (ε(h⁺) − ε(h⁻)) / (ε(h⁺) + ε(h⁻)) = a_h + b_h / p

| | a_h | b_h | A at p = 5 GeV | A at 20 GeV | A at 50 GeV |
|---|---|---|---|---|---|
| Kaon (proposed) | +0.5% | +5% GeV | +1.5% | +0.75% | +0.6% |
| Pion (proposed) | 0 | +0.5% GeV | +0.1% | +0.03% | +0.01% |

- **Meaning of the values:** toy values. They are of the order of the kaon detection
  asymmetries LHCb measures, larger at low momentum. They are not taken from a
  measurement and are to be tuned on Run 3 calibration samples later (like C11).
- **Sign:** K⁺ is found more often than K⁻, so A_K > 0. The pion asymmetry is small,
  with the same sign, as a placeholder.
- **Impact:** only b_h matters (see above). Its expected effect is linear, so shifts can
  be rescaled to other sizes.

| Option | What it means |
|---|---|
| **A** (recommended) | Use the values in the table. |
| **B** | Use other values, e.g. from a Run 3 calibration you know. |
| **C** | Use the values in the table and 5 times them, to check that the shifts scale linearly and to see them more clearly. This costs one extra study of 100 toys (about 2 hours). |

### P6 — Scenarios

The true acceptance is the default one plus the asymmetries of P5. Each scenario fits
with an analyst's model that differs in one way:

| Scenario | Analyst's model |
|---|---|
| `closure` | the true asymmetries |
| `no_asymmetry` | no asymmetry (the acceptance of Phase 8) |
| `kaon_+20%`, `kaon_-20%` | b_K scaled by 1.2 and 0.8: a mis-calibrated momentum dependence |
| `kaon_constant` | A_K fixed at its value at 10 GeV, without momentum dependence (the same as `no_asymmetry` for the fit, as a check of the cancellation) |
| `no_pion_asymmetry` | the kaon asymmetry only |

| Option | Truth |
|---|---|
| **A** (recommended) | The realistic truth of `default.yaml` (Δx = Δy = 0), so that any Δx, Δy is fake CP violation. Also the inflated truth of `inflated.yaml`, to check the dependence on the true values. |
| **B** | The realistic truth only. |

- **Cost:** about 2 hours of 40 cores per study of 100 toys of 10⁵ events per tag, with
  the toys stored (`--store-toys`) for later refits.
- **Recommendation:** A, with the scenarios in the table.

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
