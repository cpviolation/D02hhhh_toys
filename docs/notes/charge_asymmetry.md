# Phase 9 — Charge-asymmetric acceptance

Can a charge asymmetry in the tracking efficiency fake CP violation in Δx and Δy?
Decisions [C17–C19](../decisions.md#confirmed-decisions) fix the model, its size and
the scenarios.

## The model

A track of charge ± and momentum p is found with efficiency ε(p) · (1 ± A_h(p)), with

    A_h(p) = (ε⁺ − ε⁻)/(ε⁺ + ε⁻) = a_h + b_h / p,      h = K, π.

| | a_h | b_h | A_h at 5 GeV | A_h at 20 GeV |
|---|---|---|---|---|
| kaons | 0.005 | 0.05 GeV | 1.5% | 0.75% |
| pions | 0 | 0.005 GeV | 0.1% | 0.03% |

This is `ChargeAsymmetry` in `src/d02hhhh_toys/acceptance.py`, set in
`configs/acceptance/charge_asymmetric.yaml`. The factor is applied before the tracking
decision, and it never pushes the efficiency above 1: the plateau is 0.97, and the
largest factor is 1.03.

### Why only b_h matters

Every event has one track of each charge per species, so the event efficiency
contains the product

    (1 + A_K(p_K⁺)) (1 − A_K(p_K⁻)) (1 + A_π(p_π⁺)) (1 − A_π(p_π⁻)).

- **A constant A_h** gives (1 − A_K²)(1 − A_π²) for every event. The normalisation
  absorbs it, and the fit cannot see it.
- **A momentum-dependent A_h** gives, to first order,
  1 + b_K (1/p_K⁺ − 1/p_K⁻) + b_π (1/p_π⁺ − 1/p_π⁻). The CP conjugate of an event swaps
  the charges and keeps the momenta, so this term changes sign: the acceptance is no
  longer the same for an event and its CP conjugate.
- **The acceptance is the same for D0 and D0bar** decays, because it depends only on
  the final-state momenta. One normalisation sample still serves both tags.

`test_charge_asymmetry_factors` pins both properties. `test_kaon_detection_asymmetry`
checks that the measured A_K is a + b⟨1/p⟩.

## The studies

Each study has 100 toys of 10⁵ events per tag, a normalisation of 4·10⁷ events, and the
charge-asymmetric acceptance in both the generation and the closure fit.

| Study | Config | x_CP | y_CP | Δx | Δy |
|---|---|---|---|---|---|
| realistic | `charge_asymmetry.yaml` | 4e-3 | 6e-3 | 0 | 0 |
| inflated | `charge_asymmetry_inflated.yaml` | 30e-3 | 25e-3 | 5e-3 | −3e-3 |

The scenarios fit the same toys with a different model of the asymmetries:

| Scenario | The fit assumes |
|---|---|
| `no_asymmetry` | no asymmetry at all |
| `kaon_+20%`, `kaon_-20%` | b_K = 0.06 or 0.04 GeV |
| `kaon_constant` | A_K = 0.01 (its value at 10 GeV), no pion asymmetry |
| `no_pion_asymmetry` | no pion asymmetry |

`kaon_constant` is a control. By the argument above it is the same model as
`no_asymmetry`, so the two must fit alike.

The toys are stored (`--store-toys`, in `results/charge_asymmetry*/toys/`).

## Closure

All 200 closure fits converge.

| Study | | x_CP | y_CP | Δx | Δy |
|---|---|---|---|---|---|
| realistic | pull mean | +0.07 ± 0.10 | −0.07 ± 0.10 | −0.03 ± 0.09 | +0.02 ± 0.10 |
| | pull width | 1.05 ± 0.07 | 1.01 ± 0.07 | 0.91 ± 0.06 | 1.04 ± 0.07 |
| | σ_norm | 0.00 | 0.17 | 0.12 | 0.02 |
| inflated | pull mean | −0.07 ± 0.09 | −0.16 ± 0.11 | −0.10 ± 0.10 | +0.05 ± 0.10 |
| | pull width | 0.91 ± 0.06 | 1.05 ± 0.07 | 1.05 ± 0.07 | 1.01 ± 0.07 |
| | σ_norm | 0.00 | 0.08 | 0.07 | 0.02 |

The fit is unbiased, and its errors are right.

## Shifts of the scenarios

The shifts are in units of σ_stat. Their statistical errors are below 10⁻⁴σ, because the
fits are paired.

| Scenario | Study | x_CP | y_CP | Δx | Δy |
|---|---|---|---|---|---|
| `no_asymmetry` | realistic | +0.000 | +0.010 | −0.025 | **+0.016** |
| | inflated | −0.000 | +0.010 | −0.002 | **+0.012** |
| `kaon_constant` | realistic | +0.000 | +0.002 | −0.002 | **+0.015** |
| | inflated | −0.000 | +0.004 | −0.009 | **+0.014** |
| `kaon_+20%` | realistic | +0.000 | +0.002 | −0.003 | −0.002 |
| | inflated | +0.000 | +0.001 | −0.006 | −0.001 |
| `kaon_-20%` | realistic | +0.000 | +0.002 | −0.000 | +0.002 |
| | inflated | −0.000 | +0.001 | −0.000 | +0.002 |
| `no_pion_asymmetry` | realistic | +0.000 | +0.001 | −0.004 | +0.003 |
| | inflated | −0.000 | −0.000 | +0.001 | +0.003 |

### The noise of the normalisation

Every scenario has its own normalisation sample. The samples share their candidates
(common random numbers), but a different efficiency accepts slightly different
candidates. Each shift therefore carries some normalisation noise, which the paired
errors do not include.

The control measures that noise. `no_asymmetry` and `kaon_constant` are the same model,
so their difference is noise alone:

| Study | x_CP | y_CP | Δx | Δy |
|---|---|---|---|---|
| realistic | 0.000 | 0.008 | 0.023 | 0.001 |
| inflated | 0.000 | 0.006 | 0.007 | 0.002 |

The ±20% scenarios accept fewer different candidates, so their noise is smaller.

### What the shifts show

- **The asymmetries fake essentially no CP violation.** Ignoring them altogether moves
  Δy by 0.012–0.016σ, i.e. about 2·10⁻⁵. Δx moves by no more than the normalisation
  noise (0.025σ, or 4·10⁻⁵).
- **The Δy shift is real, and it is linear in b.** Both `no_asymmetry` and
  `kaon_constant` find it, so it is not noise. The ±20% scenarios give
  dΔy/db_K ≈ −0.21σ per GeV. Removing b_K = 0.05 GeV thus gives +0.011σ, and removing
  the pion asymmetry adds +0.003σ: +0.014σ in all, against the measured 0.012–0.016σ.
- **It does not depend on the truth.** The realistic and inflated studies give the same
  shifts. A real Δx of 5·10⁻³ does not enhance them.
- **x_CP and y_CP do not move** (≤ 0.01σ for y_CP, the size of the noise).
- **Scaling.** A real detector could have b_K several times larger than here, which
  would make the Δy shift several times larger. If the shift stays linear, a b_K ten
  times larger would give about 0.15σ. The asymmetry would have to be mis-modelled by much more than
  its own size before it matters at 10⁵ events per tag.

Charge asymmetries of this kind are a negligible systematic. By comparison, the
CP-symmetric data/simulation differences of [Phase 8](results.md) move y_CP by 1–5σ.

## Costs

- **Generation** took 48–55 min per toy on one core, as in Phase 8.
- **Wall time:** each study took 4–5 hours on 40 workers. The machine was shared, so
  this is more than the 2 hours of Phase 8.
- **Stored toys:** 3.2 GB per study.
