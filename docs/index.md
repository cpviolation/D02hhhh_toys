# D0 → K⁺K⁻π⁺π⁻ acceptance toys

This project uses toy studies to measure how much a mis-modelled LHCb-like acceptance
biases the mixing parameters x_CP, y_CP, Δx and Δy. These are measured in a
time-dependent amplitude fit of D0 → K⁺K⁻π⁺π⁻.

The amplitude model is the AmpGen model of JHEP 02 (2019) 126, taken from
[aman-examples](https://github.com/cpviolation/aman-examples) at tag `0.1.0`.

!!! warning "Open decisions"
    Phase 9 (charge-asymmetric acceptance) needs three answers first: which asymmetries
    to model (**P4**), their size (**P5**) and the scenarios (**P6**).
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
| 7b | Normalisation as τ moments; variance budget of v | done ([#3](https://github.com/cpviolation/D02hhhh_toys/pull/3)) | [moments and v](notes/moments.md) |
| 8 | Validation and first results | done ([#4](https://github.com/cpviolation/D02hhhh_toys/pull/4)) | [results](notes/results.md) |
| 9 | Charge-asymmetric acceptance | waiting for P4–P6 | [decisions](decisions.md#open-decisions) |

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
- **Phase 8 results** (100 toys of 10⁵ events per tag, [results](notes/results.md)):
    - the closure is unbiased, with unit pull widths;
    - every moderate data/simulation difference moves **y_CP by 1–5σ_stat**, linearly in
      its size and almost independently of the true values;
    - x_CP and Δy move by less than 0.03σ, and Δx only in proportion to a non-zero true
      Δx (≤ 0.3σ at Δx = 5·10⁻³);
    - factorising the acceptance into ε(τ)ε(p) shifts y_CP by 8.5σ, and ignoring the
      time acceptance by about 300σ.

## Quick start

```bash
uv sync                                   # add --extra cuda for the GPU
.venv/bin/d02hhhh-toys setup /path/to/large/disk/D02hhhh_toys/output   # once, see below
.venv/bin/python -m pytest                # add -m "slow or not slow" for full-model tests
.venv/bin/d02hhhh-toys study configs/study/smoke.yaml --out results/smoke --workers 8
.venv/bin/d02hhhh-toys summarise results/smoke
```

- **The output folder:** `setup` creates the folder and makes `results/` a symbolic
  link to it, so that the large outputs (stored toys: 3.4 GB per study) stay out of the
  repository. It never deletes data:
    - if `results/` already holds outputs, add `--move` to move them into the folder;
    - if it links to another folder, add `--force` to replace the link (the old folder
      is kept).

To build this site locally, run `uv run --group docs mkdocs serve`.
