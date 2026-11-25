# Conventions and findings, Phases 2–4

Each point is pinned by the test named in brackets.

## CP conjugation (`model.py`)

- p̄ swaps K+ ↔ K− and π+ ↔ π−, **and** flips p⃗ → −p⃗. The flip matters: the 6
  P-wave vector-vector chains contain an ε tensor and change sign under it
  (`test_parity_flip_is_needed`).
- Ā_f(p) = Σ c_k A_k(p̄), with c̄_k = c_k. This means CP|D0⟩ = +|D̄0⟩. With it, φρ (S
  wave) is CP-even, φ(ππ)_S and K*K̄* (P wave) are CP-odd (`test_cp_eigenstate_components`),
  and the full model has F+ ≈ 0.73 (`test_full_model_cp_even_fraction`). CLEO-c measured
  0.75 ± 0.04.
- **CP map:** for 36 of the 55 chains, A_k(p̄) = ±A_π(k)(p) exactly (to 1e-9). The
  other 19 are evaluated explicitly at p̄ (`test_full_model_unpaired_chains`):
  - 3 chains whose CP mirror is not in the model: K*(1680)K̄*(892) in the S and D
    waves, and K̄₂*(1430)K*(892);
  - the 2 a₁(1260) chains and the 14 K₁(1270) chains, because in `DtoKKpipi_v2.opt`
    the spline running widths differ between the two charges. Knot 26 of the a₁ spline
    is 1.2 for a₁⁺ and 1.20331 for a₁⁻, and three K₁ knots differ at the 1e-6 level.

  The model is used as it stands, so A_f and Ā_f are exact CP images of each other.

## Time evolution (`timeevolution.py`)

- R_D0 = cosh(wτ/2) A + sinh(wτ/2)/w (z_CP + Δz) Ā, and R_D̄0 = cosh(wτ/2) Ā +
  sinh(wτ/2)/w (z_CP − Δz) A. This form depends only on w² = z_CP² − Δz², so there is
  no branch choice. It agrees with an explicit diagonalisation of the 2×2 Hamiltonian
  (`test_explicit_diagonalisation`, to 1e-12) and with
  `aman_examples.mixing.formulate_mixing_model_zcp` (`test_agrees_with_aman_examples`).
- **A CP-even final state decays as e^{−(1+y_CP)τ}, and a CP-odd one as
  e^{−(1−y_CP)τ}** (`test_decay_time_slope_of_cp_eigenstate`). The plan's "Γ(1 − y_CP)"
  has the wrong sign for this convention. The convention is HFLAV's:
  y_CP = Γ(CP-even)/Γ − 1 at first order.

## Generation (`generate.py`)

- **|A_f|² sampler:** accept-reject on |A_f|²/r, using aman-examples'
  `peaked_phase_space_generator` (density ratio r). The maximum is estimated once and
  multiplied by 1.5; an event above it raises an error. The efficiency is 2.7% for the
  full model, against 0.8% from flat phase space.
- **Mixing step:** the proposal is ½(|A_f|² + |Ā_f|²) e^{−τ} on [0, τ_max = 20). The
  bound c = max_τ 2(|cosh|² + |sinh/w|²|z ± Δz|²) is analytic, from Cauchy–Schwarz, and
  is about 2.014 for x_CP = 4e-3, y_CP = 6e-3, so about 50% of candidates are accepted.
- **Cost:** about 9 s for 20k D0 events from the full model on the GPU (RTX A6000), and
  about 0.5 ms per event for 5 chains on CPU.

## Likelihood (`fit.py`, used by the Phase 4 closure)

- The normalisation sample can carry weights w ∝ 1/g(p) for any known density g.
  **Agreed (2026-11-25): the default normalisation sample is the generation proposal,
  ½(|A_f|² + |Ā_f|²) e^{−τ}, with w = 2/(|A_f|² + |Ā_f|²).** This makes w|R|² nearly
  constant, so a far smaller normalisation sample gives the same precision. In Phase 6
  the analyst's acceptance model is applied to it by accept-reject. Flat phase space
  (w = 1), the plan's original choice, remains available as a cross-check.

## Practical

- Building the 55 chains takes about 30 s per process. With JAX, the first call
  compiles for about 40 s, and each new array shape compiles again, so `AmplitudeModel`
  pads to fixed-size chunks. NumPy is the default backend: on CPU it is as fast as JAX
  and needs no compilation.
- Several processes on one GPU need `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Without it,
  pytest-xdist workers run out of GPU memory. The Phase 7 driver will need the same
  setting.
