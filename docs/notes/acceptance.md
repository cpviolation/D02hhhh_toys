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
5. all four daughters reconstructed; offline, at least N daughters with IPχ² > 4
   (`offline.n_tracks`, **0 by default**: the IPχ² requirement enters only through the
   one- or two-track HLT1 lines); then HLT1.

HLT1 is a list of lines. Each line requires at least N daughters with pT and IPχ²
above cuts. The lines are combined in one of two modes:
- `any`: the event passes if any line fires, as in the real trigger.
- `mixture`: each event is assigned one line at random with `fractions`, for example
  for data-taking periods with different trigger settings.

The configurations are:

| File | HLT1 |
|---|---|
| `hlt1_a.yaml` (= `base.yaml`, no offline IPχ² cut) | (a) ≥ 1 track with pT > 1 GeV and IPχ² > 7.4 |
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

With no offline IPχ² cut (`offline.n_tracks: 0`):

| HLT1 | ε over τ ∈ [0, 8) | ε at its plateau (maximum bin) |
|---|---|---|
| (a) | 7.8% | 9.7% |
| (b) | 10.1% | 12.5% |
| (a) or (b) | 10.6% | 12.9% |
| 50/50 mixture | 8.9% | 11.1% |

- **Turn-on:** ε rises from 0 to half of its plateau by τ ≈ 1.1, and is flat from
  τ ≈ 3.
- **The offline requirement matters a lot,** as the table below shows for HLT1 (a).
  The earlier default, IPχ² > 4 on all four daughters, moved the half point to
  τ ≈ 2.6 and left no plateau below τ = 8. With N = 1 the selection is the same as with
  none, because HLT1 (a) already requires one track with IPχ² > 7.4.

  | Offline IPχ² > 4 on | ε over τ ∈ [0, 8) | half of the plateau at |
  |---|---|---|
  | no daughter (default) | 7.8% | τ ≈ 1.1 |
  | ≥ 1 daughter | 7.8% | τ ≈ 1.1 |
  | ≥ 2 daughters | 7.7% | τ ≈ 1.4 |
  | all 4 daughters | 5.8% | τ ≈ 2.6 |

- **No fall-off at large τ:** nothing in the emulator removes long-lived decays. In
  LHCb, the D0 IPχ² requirement against secondary charm and the vertex geometry make
  the acceptance fall at large t.
- **The overall level is set by the track requirements.** Only 28% of events have
  all four daughters with pT > 0.25 GeV; the median daughter pT is 0.42 GeV.
- **HLT1 (b) is the more efficient option,** and adding (a) to it, in "(a) or (b)",
  gains little.

## Correlation between decay time and phase space

The figure shows ε(τ | bin)/ε(τ) in thirds of four variables, for HLT1 (a) with no
offline IPχ² cut. If the acceptance factorised as ε_t(τ)·ε_PS(p), all three curves in
each panel would be flat lines.

![Acceptance in phase-space bins](figures/acceptance_vs_tau_in_phase_space_bins.png)

Ratio of the lowest to the highest third:

| Variable | τ < 1 | 2 < τ < 6 | earlier default (IPχ² > 4 on all 4), τ < 1 |
|---|---|---|---|
| m(K⁺K⁻) | 0.88 | 1.09 | 0.60 |
| m(π⁺π⁻) | 1.43 | 1.02 | 0.58 |
| m(K⁺π⁻) | 0.98 | 0.96 | 1.63 |
| min p* of the daughters (D0 frame) | 1.19 | 0.95 | 0.36 |

- **The t × p correlation is much weaker** without the all-four-daughter cut. That cut
  required even the softest daughter to be displaced, and the softest daughter
  dominated the correlation. Now only one hard displaced track is needed, and the
  remaining effects are 10–40% at τ < 1 and a few percent beyond τ ≈ 2.
- **The m(π⁺π⁻) dependence at small τ (1.43) has the opposite sign to before.** The
  same is true of min p*, which now favours events with a soft daughter, because those
  leave more momentum to the other tracks for the pT > 1 GeV HLT1 track.
- The size of the correlation, and therefore the bias of a factorised model, depends
  strongly on how the IPχ² requirement is distributed among the tracks. The Phase 7
  overrides can scan `offline.n_tracks` and the HLT1 lines.

## Open items

- **Confirm or tune the toy values,** in particular:
  - the pT > 0.25 GeV track cut and the D0 pT spectrum, which set the overall level;
  - `offline.n_tracks`, now 0, and the HLT1 IPχ² cuts, which set the turn-on and the
    t × p correlation.
- **Choose the default HLT1 configuration** for Phase 7.
- **Decide whether to model the high-τ fall-off,** for example with a D0 IPχ²
  requirement or a maximum flight distance.
