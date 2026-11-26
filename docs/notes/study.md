# Phase 7 — Scenarios and study driver

## Running a study

    .venv/bin/d02hhhh-toys study configs/study/default.yaml --out results/default --workers 46
    .venv/bin/d02hhhh-toys summarise results/default     # rewrite summary.json and the plots

- **Configs:** a study is a YAML file in `configs/study/`.
  - `default.yaml`: realistic mixing (x_CP = 4e-3, y_CP = 6e-3, no CP violation).
  - `inflated.yaml`: large mixing and CP violation.
  - `smoke.yaml`: 5 chains and small toys, which takes minutes.
- **Resume:** a study resumes where it stopped: toys already in `toys.jsonl` are
  skipped, and the normalisation samples in `norm/` are reused.
- **Parallelism:** workers are forked processes that share the model and the
  normalisation samples. They use the NumPy backend, one BLAS thread each. JAX cannot
  be forked: `backend: jax` needs `--workers 1`.

## Scenarios

The data are always generated with the true acceptance (`acceptance:` in the study
config). A scenario is the analyst's description of the acceptance, used for the
normalisation sample: a mode and, optionally, overrides of the analyst's acceptance
model.

| Mode | Assumption | Normalisation sample |
|---|---|---|
| `full` | correct ε(τ, p) | accepted events |
| `factorised` | ε_PS(p)·ε_t(τ) | accepted events, with their τ permuted |
| `ignore_time` | ε_PS(p) only | accepted events, with new τ from e^{−τ} |
| `ignore_ps` | ε_t(τ) only | accepted τ, paired with points drawn without acceptance |
| `none` | no acceptance | events without acceptance |

`closure` must be `full` without overrides. The proposed data/simulation differences
are below. **Their magnitudes are proposals, not yet confirmed:** see decision
[P1](../decisions.md#p1-sizes-of-the-datasimulation-differences).

| Scenario | Override of the analyst's model |
|---|---|
| `ip_resolution_±5%` | `ip_resolution.scale`: 1.05 and 0.95 |
| `ipchi2_cut_+0.5` | two-track `track_ipchi2_min`: 4 → 4.5 |
| `softer_d0_pt` | `production.pt_scale`: 2.5 → 2.3 GeV |
| `slower_tracking_turn_on` | `tracking.width`: 1.5 → 2.0 GeV |

## Common random numbers

Every normalisation sample uses the same seed. The emulator draws its random numbers
per event and per batch, whatever passes a cut: HLT1 is evaluated only on events that
passed the earlier steps, with random numbers drawn for the whole batch. Different
acceptance models therefore accept nearly the same events: 5% more IP resolution
changes fewer than 20% of the accepted events (`test_common_random_numbers`).

Each distinct *base* sample is generated once: the true acceptance, each override, and
no acceptance. It is generated in chunks of `norm_chunk` events, each with its own
seed (the same for every acceptance model), spread over all workers. The scenarios are
derived from the base samples and stored as decay-time moments in
`norm/<scenario>.npz`, and the closure parts in `norm/closure.part<k>.npz`; see
[moments](moments.md). The couplings are fixed, so A_f and Ā_f replace the A_k.

## Outputs

- **`toys.jsonl`:** one line per toy, with each scenario's values, errors, validity,
  −ln L and number of calls, the generation and fit times, and the fits with the parts
  of the closure normalisation sample (below).
- **`summary.json`:**
  - `closure_pulls`: the mean and width of the closure pulls, with errors;
  - `shifts`: for each scenario, the paired shift (v_s − v_closure)/σ_closure, mean ±
    error over the toys, and its mean in units of 1e-3;
  - `norm_offset_sigma` (below).
- **`shifts.png`, `closure_pulls.png`.**

## The finite normalisation sample

All toys share the normalisation samples, so their finite size shifts every toy's fit
by the same amount. That offset does not average away over toys.

- **On the full model v is 3.8 for y_CP and 3.4 for Δx**; see [Phase 7b](moments.md),
  which predicts v without toys. The rest of this section is the smoke-study finding
  that led there.
- **The size of this offset was a surprise.** In the smoke study (5 chains, 2000
  events per tag, inflated mixing), the closure pull of y_CP had a mean of
  −0.84 ± 0.14 with 40k normalisation events (N_norm/N_data = 20). It was +0.06 ± 0.14
  with 400k, for the same toys.
- **The driver now measures it.** Each toy is also fitted with each of
  `norm_check_parts` = K disjoint parts of the closure sample. The mean shift of part
  k over the toys, m_k, is one draw of the offset of a K-times-smaller sample, so
  σ_norm = sd_k(m_k)/√K estimates the size of the full sample's offset, in units of
  σ_stat.
  - In the smoke study σ_norm(y_CP) = 0.72 at 40k and 0.28 at 400k, consistent with
    the observed offsets and with 1/√N_norm.
  - At equal N_norm/N_data, σ_norm(y_CP) ≈ √(v·N_data/N_norm) with v ≈ 10 in the
    smoke model (7.3 predicted for its setup). y_CP is hit hard because both tags share
    the sample and a normalisation error moves them in the same direction.
  - The spread over toys of a difference between parts does *not* measure the offset,
    because the offset is the same for every toy. A first version did that and
    reported 0.02.
- **Paired shifts are much less affected** wherever the common random numbers make
  the scenarios' offsets nearly equal. This is the case for `full` with overrides and
  for `factorised`, and less so for `ignore_time` and `ignore_ps`, whose τ or points
  are redrawn.
- **Consequence for Phase 8** (decision [P2](../decisions.md#p2-size-of-the-normalisation-sample)).
  For σ_norm ≲ 0.1, N_norm/N_data must be about 400. Since Phase 7b this costs only
  the generation: about 6 CPU-h for all 7 base samples at 10⁵ events per tag, with
  the shared generation.

## Cost

Measured on the full model, NumPy, one core:

| Step | Cost |
|---|---|
| model and sampler setup | 30 s + 40 s per process |
| generation with the default acceptance | 8.3 ms per accepted event |
| normalisation sample (`peaked`) | 0.078 ms per accepted event and base sample (shared), once per study |
| likelihood call (2 × 10⁵ data events, any normalisation size) | 24 ms |

Generation dominates: a 2 × 10⁵-event toy takes about 28 minutes per core, so the
default study (100 toys) takes about an hour on 46 cores.

## First look: smoke study

48 toys, 5 chains, 2000 events per tag, inflated mixing and CP violation, 40k
normalisation events. Paired shifts in units of σ_stat (mean ± error):

| Scenario | x_CP | y_CP | Δx | Δy |
|---|---|---|---|---|
| `factorised` | +0.01 ± 0.00 | −0.32 ± 0.00 | +0.10 ± 0.01 | −0.00 ± 0.00 |
| `ignore_time` | +7.2 ± 0.4 | −22.4 ± 0.4 | +23.5 ± 2.5 | −2.1 ± 0.3 |
| `ignore_ps` | −0.06 ± 0.02 | +1.24 ± 0.01 | −0.55 ± 0.03 | −0.09 ± 0.01 |
| `none` | +7.5 ± 0.5 | −19.1 ± 0.4 | +15.2 ± 3.3 | −1.9 ± 0.3 |
| `ip_resolution_+5%` | −0.01 ± 0.00 | +0.25 ± 0.00 | −0.02 ± 0.02 | +0.02 ± 0.00 |

- **Ignoring the time acceptance** mimics a lifetime change, and biases y_CP by about
  −20σ.
- **Δx and Δy also move here,** because the true Δx and Δy are not zero in these toys.
  A bias of both tags' z± carries into Δz. Phase 8 checks Δx and Δy at Δx = Δy = 0.
- **These numbers are for 5 chains and small toys:** they test the driver and are not
  results.
