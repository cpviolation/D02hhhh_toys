# Plan — Toy study of acceptance biases in a time-dependent D0 → hhhh amplitude fit

Development plan for a Claude Code session. Put it at `docs/TOY_STUDY_PLAN.md`
and start the session with: *"Read docs/TOY_STUDY_PLAN.md and do Phase 0."*

Starting point: the example notebook
`https://github.com/cpviolation/aman-examples/blob/main/docs/d02kkpipi.ipynb`
(plain toy generation and amplitude structure for D0 → K+K−π+π−).

---

## Goal

A repository that runs **full toy studies**: generate D0 and D0bar toys with
mixing and an LHCb-like acceptance, fit them with a time-dependent amplitude
likelihood, and measure **how much a mis-modelled acceptance biases**
x_CP, y_CP, Δx, Δy.

Parametrisation: `z_CP ± Δz = −(q/p)^{±1}(y + ix)`, `x_CP = −Im z_CP`,
`y_CP = −Re z_CP`, `Δx = −Im Δz`, `Δy = −Re Δz`.
In code use `x_cp_mix, y_cp_mix, dx_mix, dy_mix`: this y_CP is **not** the
lifetime-ratio observable.

## Ground rules for the session

1. **Reuse, don't rewrite.** Amplitude structure and plain toy generation come
   from the example notebook / its library. New code wraps or extends it.
2. **Conventions are pinned by tests**, never assumed: daughter ordering, CP
   conjugation, mixing signs.
3. **Ask before choosing physics** (cuts, acceptance numbers, scenarios).
   Propose, explain the impact, wait.
4. Vectorised code only, float64 in the likelihood, explicit seeds,
   `pytest` green before a phase is closed.

---

## Phase 0 — Understand the example (no new code)

Read the notebook and the library it imports. Write
`docs/notes/example_inventory.md` answering:

- **Library and version:** which package and version the notebook relies on,
  and what its license is.
- **Amplitude definition:** components, lineshapes, spin factors, couplings,
  and how the model is configured (code, YAML, JSON…).
- **Data layout:** daughter ordering and four-vector layout, and which frame
  the momenta are in.
- **CP conjugation:** whether there is a CP-conjugation function, or a way to
  get Ā_f(p) = A_f(p̄).
- **Toy generation:** how phase space is generated and how events are weighted
  or unweighted, and whether there is a maximum-weight estimate.
- **Normalisation:** how the normalisation integrals are computed.
- **Backend:** NumPy, JAX, TF, C++? Is there GPU support or batch evaluation?
- **Gaps:** what is missing for this project, e.g. decay time, CP conjugate,
  D0bar, acceptance, likelihood with an external normalisation sample.

**Done when** the inventory is written and the decisions on what to reuse and
what to add have been confirmed by me.

## Phase 1 — Repository skeleton

Set up the repository skeleton:

- `pyproject.toml`, with the example's library pinned as a dependency.
- `ruff`, `pytest`, `pre-commit`.
- `configs/`, `src/<pkg>/`, `tests/`, `docs/`, and a `.gitignore` that
  excludes `results/` and `*.npz`.
- A `CLAUDE.md` with the ground rules above.

Suggested modules:

- `model.py`: an adapter around the example's amplitude.
- `timeevolution.py`
- `acceptance.py`
- `generate.py`
- `fit.py`
- `study.py`
- `plotting.py`
- `cli.py`

**Done when** CI runs `ruff` and `pytest`.

## Phase 2 — Amplitude adapter

The adapter exposes three methods:

- `amplitudes(p) -> (A_f(p), Ā_f(p))`, with Ā_f(p) = Σ c̄_k A_k(p̄).
- `component_amplitudes(p) -> (n, K)`, which allows precomputation so the
  fit is fast.
- `cp_even_fraction()`, as a diagnostic.

Tests:

- **CP conjugation is an involution:** applying it twice gives back the event.
- **CP-eigenstate components** satisfy A_k(p̄) = ±A_k(p).
- **CP-mirror components map onto each other**, e.g. K1+K− and K1−K+.
- **Bose symmetry** holds for 4π, if 4π is added.
- **Without direct CPV, F+ for a pure CP-even model is 1.**

If the example evaluates the model twice for p and p̄, add a CP map, i.e. a
partner index and sign per component, to halve the cost, and test it.

## Phase 3 — Time evolution

Exact evolution: with w = y + ix and w² = z_CP² − Δz²,

    R_D0    = cosh(wτ/2) A_f + [sinh(wτ/2)/w] (z_CP + Δz) Ā_f
    R_D0bar = cosh(wτ/2) Ā_f + [sinh(wτ/2)/w] (z_CP − Δz) A_f
    rate    = e^{−τ} |R|²,   τ = Γt

Tests:

- **Explicit diagonalisation:** compare with a direct diagonalisation of the
  2×2 effective Hamiltonian (|D1,2⟩ = p|D0⟩ ± q|D0bar⟩) for non-trivial
  x, y, |q/p| and φ.
- **First-order limit:** the expansion matches for small τ.
- **Known limits:** no mixing gives the identity, and the (x, y, |q/p|, φ) to
  (x_CP, y_CP, Δx, Δy) conversion gives x_CP = x, y_CP = y when there is no CPV.

## Phase 4 — Toy generation with mixing

Build on the example's no-mixing generator rather than accept-reject from flat
phase space. Pure accept-reject on a peaky |A|² keeps only about 3% of events.

1. **Proposal:** draw half the events from the example's |A_f|² generator, and
   half from the same generator followed by CP conjugation of the points,
   which samples |Ā_f|². Draw τ from an exponential. The proposal density is
   q ∝ (|A_f|² + |Ā_f|²)/2 × e^{−τ}.
2. **Accept-reject** with |R_tag(τ, p)|² / (c·q). Because mixing is small, the
   ratio is bounded, c ≈ 2·(1 + small), and the efficiency is about 50%.
   Estimate c from a sample and raise an error if it is exceeded.
3. **Separate tags:** generate D0 and D0bar samples independently.

Tests:

- The generated time distribution of a CP-even-only model has a slope
  consistent with Γ(1 − y_CP) at first order.
- A single-toy fit without acceptance closes, with inflated mixing values.

## Phase 5 — LHCb-like acceptance emulator

Parametric, configured by a YAML dataclass. Its parameters are toy values to
be tuned on Run 3 simulation.

- **D0 production:** pT spectrum, η range, uniform φ, boost to the lab.
- **Flight and IP:** flight distance L = (p/m)·c·t, and each daughter's
  IP = |L × û|.
- **IP resolution:** σ_IP = scale·(a + b/pT) and IPχ² = (IP_meas/σ)².
- **Track requirements:** η, p and pT cuts, plus a momentum-dependent
  tracking efficiency.
- **Selection:** an offline IPχ² cut on every daughter, plus an HLT1-like
  requirement of at least N tracks with pT and IPχ² above cut.

Diagnostics, with a short note in `docs/notes/`:

- The acceptance as a function of τ.
- The same curve in bins of phase-space variables, which shows the t × p
  correlation.

## Phase 6 — Likelihood with external normalisation

    −ln L = −Σ_tags [ Σ_i ln|R_tag(τ_i, p_i)|² − N_tag ln mean_j |R_tag(τ_j, p_j)|² ]

- **Normalisation:** use a sample drawn from flat phase space × e^{−τ} ×
  the analyst's model of the acceptance, normalised separately per tag.
- **Speed:** precompute the component amplitudes for data and the
  normalisation sample once per toy.
- **Units:** fit the mixing parameters in units of 1e-3. Couplings are fixed
  by default, with an option to float them.

Test: a single-toy closure with acceptance, where the true acceptance is also
used in the normalisation.

## Phase 7 — Scenarios and study driver

Normalisation modes, i.e. how the analyst models the acceptance:

| mode | assumption |
|---|---|
| `full` | correct ε(t, p) |
| `factorised` | ε_PS(p)·ε_t(t); permute τ among accepted events |
| `ignore_time` | ε_PS(p) only |
| `ignore_ps` | ε_t(t) only |
| `none` | no acceptance |

In addition, `acceptance_overrides` mimic data/MC differences, for example:

- IP resolution off by 5%.
- A shifted IPχ² cut.
- A softer D pT spectrum.
- A slower tracking-efficiency turn-on.

Driver design:

- **Generate each toy once** and fit it with every scenario.
- **Common random numbers:** generate the normalisation samples with the same
  seed for every scenario, and store them as (τ, A_k) in `.npz` files.
- **Paired shift:** report each scenario's shift w.r.t. `closure`, toy by toy,
  as a mean ± error in units of σ_stat.
- **Closure checks:** report the closure pull mean and width.
- **Parallelism:** multiprocessing over toys.
- **Outputs:**
  - `toys.jsonl`
  - `summary.json`
  - plots of the shifts and of the closure pulls.

Warning to document: a finite normalisation sample gives a common offset of
order σ_stat·√(N_data/N_norm). Keep N_norm ≫ N_data.

## Phase 8 — Validation and first results

1. Run 100+ closure toys: pulls should be centred with unit width.
2. Run the default scenarios at realistic statistics, with true x_CP and y_CP
   around 4e-3 and 6e-3. Also run with inflated mixing values, to check
   whether the biases depend on the true values.
3. Write a short results note: which mis-modellings matter, and which
   parameters are most sensitive to them. Δx and Δy should be insensitive to
   CP-symmetric acceptance effects; check this.

## Later extensions

- **Decay-time resolution**, in generation and, optionally, in the fit.
- **Secondary D from B**, with its own time distribution.
- **Charge-asymmetric acceptance**, e.g. from the slow pion or K+/K−
  differences, which bias Δx and Δy.
- **Direct CPV** per component.
- **Simultaneous fit of 4π and KKππ.**
- **The K3π right-sign ratio method** for ε_t(t | ξ).
- **A GPU or JAX likelihood**, if the example's library doesn't already
  provide one.
