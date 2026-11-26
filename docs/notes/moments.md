# Phase 7b — The normalisation as moments, and the factor v

## The normalisation as decay-time moments

With fixed couplings, the normalisation sum of the likelihood is

    Σ_j w_j |R(τ_j, p_j)|² = Σ_j w_j [ |c|² |A|² + |s|² |ζ|² |Ā|² + 2 Re(c̄ s ζ A* Ā) ]

with c = cosh(wτ/2), s = sinh(wτ/2)/w, and ζ = z_CP ± Δz.

- **Moments:** |c|², |s|² and c̄s are power series in τ whose coefficients depend only
  on the mixing parameters. The sum is therefore a combination of the moments
  M_n = Σ_j w_j τ_j^n × {|A|², |Ā|², A*Ā}, for n ≤ 20.
- **Free couplings:** the moments become K × K matrices of the component amplitudes.
- **Code:** see `src/d02hhhh_toys/moments.py`, in particular `NormMoments`.
- **Accuracy:** the series is truncated, and this is the only approximation. At
  τ_max = 20 the truncation error is below 1e-15 for realistic and inflated mixing.
  - It is bounded at every likelihood call.
  - A point where the bound exceeds 1e-10 is returned as a bad step. Such points are
    |w| ≳ 0.4, far from any minimum.
  - The likelihood agrees with the event sum to 1e-12 (`test_moments.py`).

### What changes

| | Before | Now |
|---|---|---|
| Storage per normalisation sample | (τ, A, Ā, w) per event: 5 GB at 10⁸ | 3 × 21 complex numbers and 21 τ moments: a few kB |
| Likelihood call, 2 × 10⁵ data, 2 × 10⁶ normalisation events | 576 ms | 24 ms (data only) |
| Generation | one process per base sample | chunks of `norm_chunk` = 250k events, each with its own seed, spread over all workers |

- **Study modes:** each one is now a moment operation. A mode that replaces the decay
  times (`factorised`, `ignore_time`, `ignore_ps`) multiplies the zeroth moments by the
  moments ⟨τⁿ⟩ of the new decay times. This is the exact limit of infinitely many
  permutations, so the permutation noise is gone.
- **Normalisation-offset check:** the `norm_check_parts` are now groups of whole
  chunks, so they cost nothing extra.
- **Common random numbers:** chunk i has the same seed for every acceptance model.
- **Old study directories:** their `norm/*.npz` hold events, which the new code cannot
  read. It says so: delete `norm/` and the moments are rebuilt.

## The factor v

A finite normalisation sample of n events shifts the fit of N events per tag by an
offset that is common to all toys:

    σ_norm² = v · (N/n) · σ_stat²

- **The method:** to first order the sample enters only through the gradient of the
  normalisation term. So v follows from the per-event influence U and the Fisher
  information I: v_i = (I⁻¹ Cov(U) I⁻¹)_ii / (I⁻¹)_ii.
- **Code:** see `src/d02hhhh_toys/variance.py`. It needs one weighted sample and no
  toys.
- **Running it:**
  `.venv/bin/python scripts/variance_budget.py configs/study/default.yaml`.
- **Validation:** the same data were fitted with 80 independent normalisation samples
  (smoke model, 5000 events per tag, 10⁴ normalisation events each).

  | | x_CP | y_CP | Δx | Δy |
  |---|---|---|---|---|
  | v measured (± 16%) | 0.04 | 6.75 | 1.69 | 0.12 |
  | v predicted | 0.02 | 7.26 | 1.85 | 0.14 |

  `test_v_factor_predicts_the_spread_of_fits` repeats this check on a smaller scale.

### Results for the full model

These are for the default acceptance, at the realistic truth. The inflated truth gives
the same numbers within a few per cent. Each row is a design of the normalisation
sample, and v is per normalisation event of that design.

| Design | x_CP | y_CP | Δx | Δy |
|---|---|---|---|---|
| **Now:** peaked phase space, one τ and one acceptance decision per point | 0.000 | **3.77** | **3.36** | 0.39 |
| of which phase space | 0.000 | 0.11 | 0.13 | 0.02 |
| of which τ and acceptance | 0.000 | 3.66 | 3.23 | 0.37 |
| Averaged over M τ and acceptance draws per point, per point (M → ∞) | 0.000 | 2.33 | 2.72 | 0.41 |
| … per-replica term, added as `within`/M | 0.002 | 78.4 | 69.3 | 7.8 |
| Phase-space density ∝ \|A_f\|² + \|Ā_f\|², per event | 0.000 | 1.38 | 1.31 | 0.16 |
| Both | 0.000 | 0.94 | 1.02 | 0.16 |

- **v is smaller than the smoke study suggested:** 3.8 for y_CP, not about 10. **Δx
  is as exposed as y_CP**, which the earlier notes missed.
- **x_CP is insensitive to the normalisation.** For a CP-symmetric acceptance,
  ∫ Im(A*Ā) vanishes by CP symmetry, so the normalisation does not depend on x_CP at
  first order.
- **The large weights drive v.** Most of v shows up as "τ and acceptance" because each
  point's single τ draw is multiplied by a weight 1/r. The weights are large wherever
  the peaked density r is far from |A|². With a density ∝ |A|², v would drop by 2.7.

### What each design costs

On the full model, NumPy, one core:
- an amplitude evaluation costs 0.077 ms;
- one acceptance candidate costs 0.0023 ms;
- the acceptance keeps 4.7% of the candidates, i.e. 21 candidates per accepted event.

| Design | Cost per normalisation event | v(y_CP) | Cost × v |
|---|---|---|---|
| Now | 0.126 ms | 3.77 | **0.47** |
| Averaging, at the best M ≈ 34 | 0.154 ms | 4.7 | 0.72 |
| Density ∝ \|A\|² by accept-reject (`proposal`) | ~30 × more | 1.38 | ≫ 1 |
| Density ∝ \|A\|², if it were free | 0.126 ms | 1.38 | 0.17 (ideal) |

- **Averaging over τ and the acceptance does not pay.** The per-replica term is large,
  so many replicas are needed, and their acceptance evaluations cost more than they
  save.
- **The only real gain left is a better phase-space density.** It would have to be
  closer to |A|² than the peaked phase space, which has peaks at φ, ρ and ω only, and
  still be sampled directly, without accept-reject. Adding K*(892) peaks in the Kπ
  masses is the obvious candidate. The gain is at most 2.7.
- **It is not needed for Phase 8.** With the moments, the current design reaches
  σ_norm ≈ 0.1 cheaply (decision [P2](../decisions.md#p2-size-of-the-normalisation-sample)):

  | N per tag | n | σ_norm(y_CP), σ_norm(Δx) | Per base sample | 7 base samples on 46 cores |
  |---|---|---|---|---|
  | 10⁵ | 4·10⁷ | 0.10, 0.09 | 1.4 CPU-h | ~15 min |
  | 10⁵ | 10⁸ | 0.06, 0.06 | 3.5 CPU-h | ~35 min |
