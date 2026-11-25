# Phase 6 — Likelihood and normalisation sample

## Likelihood

The likelihood is implemented in `src/d02hhhh_toys/fit.py`:

    −ln L = −Σ_tags [ Σ_i ln|R_tag(τ_i, p_i)|² − N_tag ln( Σ_j w_j |R_tag(τ_j, p_j)|² / Σ_j w_j ) ]

- **Rates:** they are relative to e^{−τ}; the e^{−τ_i} of the data does not depend on
  the parameters.
- **Normalisation events:** distributed as g(p)·e^{−τ}·ε_model(τ, p) on [0, τ_max),
  where ε_model is the analyst's acceptance, applied by accept-reject, and weighted
  with w ∝ 1/g. The sum then estimates the normalisation integral up to a constant
  factor.
- **Tags:** each tag is normalised separately. One sample can serve both tags, because
  neither g nor the emulated acceptance depends on the flavour.
- **Units:** the mixing parameters are fitted in units of 1e-3.
- **Couplings:** fixed by default (`MixingNLL`, which needs only A_f and Ā_f per event,
  precomputed once per toy). `CouplingMixingNLL` frees the real and imaginary parts of
  chosen couplings, using the precomputed A_k(p) and A_k(p̄) and costing N·K operations
  per call; at least one coupling stays fixed.

## Choice of the normalisation density

The agreed principle is a weighted normalisation sample. Three densities g are
implemented in `src/d02hhhh_toys/normalisation.py`. The table measures them on the
full model, with 2·10⁵ events each and no acceptance. The first column is N times the
variance of the estimated mixing-dependent ratios Re⟨A*Ā⟩, Im⟨A*Ā⟩ and
⟨|A|² − |Ā|²⟩, each divided by ⟨(|A|² + |Ā|²)/2⟩; lower is better.

| Method | g | N·variance | amplitude evaluations per event |
|---|---|---|---|
| `flat` (the plan) | 1 | 0.86–2.32 | 1 |
| `peaked` (**default**) | r(p), the peaked phase space of aman-examples | 0.53–0.70 | 1 |
| `proposal` | (\|A_f\|² + \|Ā_f\|²)/2 | 0.21–0.29 | 38.5 |

- **Per amplitude evaluation, `peaked` is about 15 times more efficient than
  `proposal`,** and 2–3 times better than `flat`. The variance of `proposal` is only
  2.5 times smaller, while each of its events costs an accept-reject on |A_f|²/r.
- **With an acceptance, `flat` and `peaked` gain another factor of ~10:** the
  acceptance needs only the kinematics and τ, so it is applied before any amplitude is
  evaluated. `proposal` needs the amplitudes for its own accept-reject.
- **The default is therefore `peaked`.** The two other methods remain as options and
  cross-checks; `test_normalisation_methods_agree` checks that all three estimate the
  same ratios.

**Finite-sample offset.** A finite normalisation sample gives a common offset of order
σ_stat·√(N_data·v/N_norm), with v the variance factor above (about 0.7 for `peaked`).
Keep N_norm ≫ N_data·v.

## Generation with acceptance

`MixingGenerator.generate(..., acceptance=...)` processes each candidate in this order:

1. phase-space kinematics, CP-conjugation choice and τ;
2. the acceptance;
3. the amplitudes, for the events that passed;
4. accept-reject on |A_f|²/r;
5. accept-reject on the mixing weight.

These steps are independent factors of the target density, so their order does not
change it (`test_acceptance_before_amplitudes_is_unbiased`). It makes about
1/ε_acc ≈ 10 times fewer amplitude evaluations.

## Tests

- **Closures:** a single-toy fit closes, with pulls below 4, in each of these cases:
  - with the toy LHCb-like acceptance applied to the data and the same acceptance
    model in the normalisation (`test_single_toy_closure_with_acceptance`, on 5 chains);
  - the same with the full model, on the GPU
    (`test_single_toy_closure_with_acceptance_full_model`);
  - with two complex couplings free (`test_single_toy_closure_with_free_couplings`).
- **Consistency:** `CouplingMixingNLL` at the generation couplings equals `MixingNLL`
  to 1e-12.
