# Phase 5 — Acceptance emulator: diagnostics

**All numbers come from the proposed toy values in `configs/acceptance/base.yaml`,
which are not yet confirmed.** They show how the emulator behaves; they are not a
prediction for Run 3. Reproduce them with
`.venv/bin/python scripts/acceptance_diagnostics.py`: 10⁶ events of flat four-body phase
space with τ uniform in [0, 8), which takes about 12 s.

## The emulator

See `src/d02hhhh_toys/acceptance.py`. Each event is processed in five steps:

1. random orientation of the decay in the D0 rest frame;
2. D0 production (pT spectrum, η, φ) and boost to the laboratory;
3. flight distance L = (p/m)·cτ·τ and impact parameters IP = |L × û|, smeared with
   σ_IP = s·(a + b/pT);
4. track requirements (η, p, pT) and a momentum-dependent tracking efficiency;
5. IPχ² > 4 for all four daughters, then HLT1.

HLT1 is a list of lines. Each line requires at least N daughters with pT and IPχ²
above cuts. The lines are combined in one of two modes:
- `any`: the event passes if any line fires, as in the real trigger.
- `mixture`: each event is assigned one line at random with `fractions`, for example
  for data-taking periods with different trigger settings.

The configurations are:

| File | HLT1 |
|---|---|
| `hlt1_a.yaml` (= `base.yaml`) | (a) ≥ 1 track with pT > 1 GeV and IPχ² > 7.4 |
| `hlt1_b.yaml` | (b) ≥ 2 tracks with pT > 0.5 GeV and IPχ² > 4 |
| `hlt1_a_or_b.yaml` | (a) or (b) |
| `hlt1_mixture.yaml` | (a) for 50% of events, (b) for the other 50% |

Phase 7 overrides take dotted keys, for example
`config.with_overrides({"ip_resolution.scale": 1.05})`.

The emulator never uses the charges, so ε(τ, p̄) = ε(τ, p), and a test checks this
(`test_cp_symmetric`). CP-symmetric mis-modellings should therefore leave Δx and Δy
unbiased; Phase 8 will check that.

## Acceptance versus decay time

![Acceptance versus decay time](figures/acceptance_vs_tau.png)

| HLT1 | ε over τ ∈ [0, 8) | ε in the last bin (τ ≈ 7.9) |
|---|---|---|
| (a) | 5.8% | 9.1% |
| (b) | 7.3% | 11.6% |
| (a) or (b) | 7.4% | 11.9% |
| 50/50 mixture | 6.5% | 10.4% |

- **The turn-on is steep and late:** ε ≈ 0 below τ ≈ 0.7, and the curve reaches half
  of its τ = 8 value at τ ≈ 2. The cause is the offline IPχ² > 4 required on *all four*
  daughters.
- **There is no plateau:** ε is still rising at τ = 8, because nothing in the emulator
  removes long-lived decays. In LHCb, the D0 IPχ² requirement against secondary
  charm, the decay-vertex position and the VELO geometry all make the acceptance fall
  at large t. None of these are modelled, and the "Secondary D from B" extension would
  need some of them.
- **The overall level is low,** at 6–7%. Most of the loss comes from the track
  requirements. For example, only 28% of events have all four daughters with
  pT > 0.25 GeV; the median daughter pT is 0.42 GeV.
- **HLT1 (b) is the more efficient option.** Adding (a) to it, in "(a) or (b)", gains
  almost nothing. The mixture lies between the two, by construction.

## Correlation between decay time and phase space

The figure shows ε(τ | bin)/ε(τ) in thirds of four variables, for HLT1 (a). If the
acceptance factorised as ε_t(τ)·ε_PS(p), all three curves in each panel would be flat
lines.

![Acceptance in phase-space bins](figures/acceptance_vs_tau_in_phase_space_bins.png)

Ratio of the lowest to the highest third:

| Variable | τ < 1 | 2 < τ < 6 |
|---|---|---|
| m(K⁺K⁻) | 0.60 | 1.03 |
| m(π⁺π⁻) | 0.58 | 0.95 |
| m(K⁺π⁻) | 1.63 | 0.94 |
| min p* of the daughters (D0 frame) | 0.36 | 0.67 |

- **The correlation is strong at small τ** and mostly disappears by τ ≈ 2 in the
  two-body masses. The low-τ region is where a factorised acceptance model will be
  wrong.
- **It follows the softest daughter.** A soft track in the D0 frame is soft and
  nearly collinear with the D0 in the laboratory. It therefore needs a longer flight
  to pass the IPχ² cut. The effect persists at large τ (0.67), because the pT > 0.25 GeV
  and tracking-efficiency requirements do not depend on τ.
- The two-body masses inherit this effect through the momentum sharing between the
  daughters. That is why m(K⁺π⁻) shows the opposite trend to m(K⁺K⁻) and m(π⁺π⁻).

## Open items

- **Confirm or tune the toy values,** in particular:
  - the pT > 0.25 GeV track cut and the D0 pT spectrum, which set the overall level;
  - the IPχ² > 4 on all four daughters, which sets the turn-on.
- **Choose the default HLT1 configuration** for Phase 7.
- **Decide whether to model the high-τ fall-off,** for example with a D0 IPχ²
  requirement or a maximum flight distance.
