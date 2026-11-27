# Phase 8 — Validation and first results

Three studies with the full model and the default acceptance (two-track HLT1, RF foil),
each with 100 toys of 10⁵ events per tag and a normalisation of 4·10⁷ events
(decisions [C14–C16](../decisions.md#confirmed-decisions)):

| Study | Config | x_CP | y_CP | Δx | Δy |
|---|---|---|---|---|---|
| default | `default.yaml` | 4e-3 | 6e-3 | 0 | 0 |
| inflated | `inflated.yaml` | 30e-3 | 25e-3 | 5e-3 | −3e-3 |
| inflated, no CPV | `inflated_no_cpv.yaml` | 30e-3 | 25e-3 | 0 | 0 |

σ_stat at 10⁵ events per tag is about 1.8·10⁻³ for x_CP and Δx, and 1.4·10⁻³ for y_CP
and Δy.

## Closure

All 300 closure fits converge.

| Study | | x_CP | y_CP | Δx | Δy |
|---|---|---|---|---|---|
| default | pull mean | +0.04 ± 0.10 | −0.10 ± 0.12 | −0.01 ± 0.10 | −0.02 ± 0.10 |
| | pull width | 1.02 ± 0.07 | 1.16 ± 0.08 | 1.01 ± 0.07 | 1.05 ± 0.07 |
| inflated | pull mean | −0.05 ± 0.10 | +0.13 ± 0.10 | −0.29 ± 0.10 | +0.06 ± 0.10 |
| | pull width | 0.97 ± 0.07 | 0.99 ± 0.07 | 1.00 ± 0.07 | 1.00 ± 0.07 |
| inflated, no CPV | pull mean | +0.16 ± 0.09 | +0.07 ± 0.10 | +0.15 ± 0.11 | +0.08 ± 0.10 |
| | pull width | 0.90 ± 0.06 | 1.02 ± 0.07 | 1.06 ± 0.08 | 0.95 ± 0.07 |
| all | normalisation offset σ_norm | 0.00 | 0.07–0.12 | 0.06–0.11 | 0.02–0.03 |

- **The fit is unbiased and its errors are right.** Of the 24 numbers, the largest
  deviations are the inflated Δx mean (−0.29 ± 0.10) and the default y_CP width
  (1.16 ± 0.08). They are not repeated in the other studies.
- **The normalisation offset is as predicted.** σ_norm comes from the spread of the
  fits with 4 disjoint quarters of the normalisation, rescaled to the full sample.
  It matches √(v·N/n) with the v of [Phase 7b](moments.md). It is a common offset
  of all toys, so the inflated Δx mean is (−0.29 ± 0.10 ± 0.11): 2σ.

## Shifts of the scenarios

Each scenario fits the same toys as the closure, so the shift is paired: its error is
far smaller than σ_stat. All the tables are in units of σ_stat. The errors of the
shifts are below 0.01σ for the data/simulation differences, and are given otherwise.

### Simplified acceptance treatments

| Scenario | Study | x_CP | y_CP | Δx | Δy |
|---|---|---|---|---|---|
| `factorised` | default | −0.04 | **−8.84** | +0.01 | −0.00 |
| | inflated | −0.05 | **−8.37** | +0.18 | +0.00 |
| | no CPV | −0.06 | **−8.45** | −0.11 | −0.01 |
| `ignore_ps` | default | −0.00 | −0.14 | +0.15 | +0.04 |
| | inflated | +0.00 | +0.67 | +0.31 | +0.08 |
| | no CPV | +0.00 | +0.51 | −0.17 | −0.08 |
| `ignore_time` | default | −0.9 ± 0.3 | **−312** | −1.0 ± 2.5 | −0.4 ± 0.6 |
| | inflated | −1.8 ± 0.2 | **−316** | **+28 ± 2** | −1.6 ± 0.3 |
| | no CPV | −2.7 ± 0.2 | **−318** | +1.7 ± 2.3 | +0.3 ± 0.4 |
| `none` | default | −0.8 ± 0.3 | **−309** | −0.6 ± 2.8 | −0.5 ± 0.6 |
| | inflated | −1.9 ± 0.2 | **−311** | **+36 ± 2** | −1.0 ± 0.3 |
| | no CPV | −2.7 ± 0.2 | **−314** | +1.1 ± 2.9 | −0.4 ± 0.4 |

- **The default `ignore_time` and `none` rows come from 25 toys**, from
  `default_extremes.yaml`. The 100-toy run had used the order-20 moment series, which
  does not converge so far from the truth (see below). The 25 toys are the first 25 of
  the default run, and their closure fits are identical.
- **Ignoring the time acceptance is not an option.** The fit moves y_CP to about −0.43,
  so that the decay-time distribution falls slowly enough to mimic the turn-on of the
  acceptance. With CP violation in the truth, Δx moves by 30σ too.
- **The phase-space acceptance matters little on its own.** Ignoring it shifts y_CP by
  less than 0.7σ and Δx by less than 0.3σ.
- **The correlation between decay time and phase space matters.** Factorising the
  acceptance into ε(τ) ε(p) shifts y_CP by 8.5σ, whatever the truth.

### Data/simulation differences

| Scenario | Study | x_CP | y_CP | Δx | Δy |
|---|---|---|---|---|---|
| `ip_resolution_+5%` | default | +0.01 | **+2.80** | +0.03 | +0.00 |
| | inflated | +0.01 | **+2.22** | −0.17 | −0.02 |
| | no CPV | +0.01 | **+2.23** | −0.04 | +0.01 |
| `ip_resolution_-5%` | default | −0.01 | **−3.16** | +0.01 | −0.02 |
| | inflated | −0.01 | **−2.48** | +0.20 | +0.02 |
| | no CPV | −0.02 | **−2.53** | +0.00 | −0.02 |
| `ipchi2_cut_+0.5` | default | +0.02 | **+4.86** | +0.03 | −0.00 |
| | inflated | +0.02 | **+3.90** | −0.29 | −0.02 |
| | no CPV | +0.02 | **+3.92** | −0.05 | +0.01 |
| `softer_d0_pt` | default | +0.01 | **+3.04** | +0.08 | +0.00 |
| | inflated | +0.01 | **+2.67** | −0.10 | −0.02 |
| | no CPV | +0.02 | **+2.63** | −0.06 | +0.00 |
| `slower_tracking_turn_on` | default | −0.00 | −1.10 | +0.04 | −0.00 |
| | inflated | −0.00 | −0.85 | +0.07 | +0.00 |
| | no CPV | −0.01 | −0.82 | −0.05 | +0.01 |

- **y_CP is the only parameter that matters.** Every moderate difference moves it by
  1–5σ, i.e. by 1.5–7·10⁻³: as much as the realistic y_CP itself. The differences all
  change how fast the acceptance rises with τ, which the fit absorbs in the decay
  rate of the CP-even component.
- **x_CP and Δy do not move** (< 0.03σ).
- **Δx moves only if CP is violated in the truth.** Without CP violation the shifts are
  below 0.08σ; with Δx = 5·10⁻³ they reach 0.3σ, i.e. 2–10% of the true Δx. This
  confirms that Δx and Δy are insensitive to CP-symmetric acceptance effects (plan,
  Phase 8 step 3), up to a scale error on a non-zero Δx.
- **The shifts are linear in the size of the difference.** ±5% in the IP resolution
  gives +2.80 and −3.16σ: symmetric to about 10%. Results can be rescaled to other sizes.
- **The y_CP shift barely depends on the truth.** It is 20% smaller for the inflated
  mixing than for the realistic one, with or without CP violation. In absolute units
  it is 3.9·10⁻³ against 3.1·10⁻³ for `ip_resolution_+5%`.

## Technical issues found in this phase

- **Accept-reject maximum.** The 10⁵-event toys draw about 6·10⁸ candidates. One of them
  exceeded the maximum of |A|²/r estimated from 5·10⁵ candidates (× 1.2), and the
  generator stopped, as designed. In 4·10⁷ candidates the largest weight is 4.42,
  with a fast-falling tail. The studies now set `sampler_maximum: 6.0`, which costs
  1.4 times more candidates.
- **Moment series far from the truth.** The `ignore_time` and `none` fits end at
  |w| ≈ 0.4, where the order-20 series no longer converges and the likelihood returns
  its penalty. Their decay times are independent of the point, so their moments are now
  the zeroth moments times the analytic ⟨τⁿ⟩, to order 80
  (`test_analytic_moments_converge_far_from_the_truth`). The other modes stay at
  order 20.

## Costs

- A toy of 10⁵ events per tag takes about 48 min of generation on one core and 1.5 min
  of fits for all 14 fits.
- 100 toys on 40 workers take about 2 hours. The normalisation (7 base samples of 4·10⁷
  events) takes a few minutes more.
- Generation dominates. These toys were not stored, so another scenario means
  generating them again. Since then, `--store-toys` keeps them (34 MB per toy, see
  [study](study.md#running-a-study)) and `--toys-from` refits them.
