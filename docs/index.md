# D0 → K⁺K⁻π⁺π⁻ acceptance toys

This project uses toy studies to measure how much a mis-modelled LHCb-like acceptance
biases the mixing parameters x_CP, y_CP, Δx and Δy. These are measured in a
time-dependent amplitude fit of D0 → K⁺K⁻π⁺π⁻.

The amplitude model is the AmpGen model of JHEP 02 (2019) 126, taken from
[aman-examples](https://github.com/cpviolation/aman-examples) at tag `0.1.0`.

!!! warning "Open decisions"
    Phase 8 needs two answers first: the sizes of the data/simulation differences
    (**P1**) and the size of the normalisation sample (**P2**).
    See [Decisions](decisions.md).

## Status

| Phase | Content | Status | Notes |
|---|---|---|---|
| 0 | Inventory of the example | done | [inventory](notes/example_inventory.md) |
| 1 | Repository skeleton, CI | done | |
| 2 | Amplitude adapter, CP conjugation | done | [conventions](notes/conventions.md) |
| 3 | Time evolution | done | [conventions](notes/conventions.md) |
| 4 | Toy generation with mixing | done | [conventions](notes/conventions.md) |
| 5 | LHCb-like acceptance emulator | done | [acceptance](notes/acceptance.md) |
| 6 | Likelihood with external normalisation | done | [normalisation](notes/normalisation.md) |
| 7 | Scenarios and study driver, documentation site | done ([#1](https://github.com/cpviolation/D02hhhh_toys/pull/1), [#2](https://github.com/cpviolation/D02hhhh_toys/pull/2)) | [study](notes/study.md) |
| 7b | Normalisation as τ moments; variance budget of v | in review | [moments and v](notes/moments.md) |
| 8 | Validation and first results | waiting for P1, P2 | |

The full plan is in the [study plan](TOY_STUDY_PLAN.md).
[Decisions](decisions.md) lists where the code departs from it.

## Key numbers so far

- **The fraction of CP-even decays** in the model is F₊ ≈ 0.73. CLEO-c measured
  0.75 ± 0.04.
- **About 10% of decays pass the default acceptance** (two-track HLT1 and the RF foil,
  over τ < 16). The acceptance rises to half its maximum at τ ≈ 1.2.
- **Decay time and phase-space position are correlated in the acceptance.** At large τ
  the acceptance in the lowest third of m(K⁺K⁻) is 1.5 times that in the highest third.
- **The finite normalisation sample offsets the closure** by σ_norm = √(v·N_data/N_norm),
  in units of σ_stat. On the full model v = 3.8 for y_CP and 3.4 for Δx; for Δy and
  x_CP it is small.
- **The fits no longer depend on the normalisation size.** The normalisation is
  evaluated from decay-time moments, which makes a likelihood call 24 times faster and
  shrinks the stored samples to a few kB.
- **Ignoring the time acceptance biases y_CP by about 20σ.** This comes from the smoke
  study (5 chains, inflated mixing). It tests the driver and is not a result.

## Quick start

```bash
uv sync                                   # add --extra cuda for the GPU
.venv/bin/python -m pytest                # add -m "slow or not slow" for full-model tests
.venv/bin/d02hhhh-toys study configs/study/smoke.yaml --out results/smoke --workers 8
.venv/bin/d02hhhh-toys summarise results/smoke
```

To build this site locally, run `uv run --group docs mkdocs serve`.
