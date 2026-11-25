# Phase 0 — Inventory of the D0 → K+K−π+π− example

Source read: `cpviolation/aman-examples` at commit `c1bc005` (local clone in
`../aman-examples`, in sync with `origin/main`): `docs/d02kkpipi.ipynb`, the whole of
`src/aman_examples/` and `tests/test_ampgen_d02kkpipi.py`. The figures below marked
*(checked)* come from a throw-away script on 10⁵ flat phase-space events (RTX A6000,
JAX float64). The script lives in the session scratchpad, not in this repository.

## 1. Library and version

| Item | Value |
|---|---|
| Package | `aman-examples` (import `aman_examples`), the user's own repository |
| Version | **not released**: no git tag, not on PyPI (`no-pypi = true`), version from `setuptools_scm` (fallback `0.0.0`). It has to be pinned by commit: `aman-examples @ git+https://github.com/cpviolation/aman-examples@c1bc005` |
| Python | `>=3.12,<3.14` (no JAX wheels for 3.14) |
| Licence | BSD-3-Clause for the code. `data/ampgen/DtoKKpipi_v2.opt` is an unmodified copy of AmpGen's file (commit `fc651a7`) and is **GPL-2.0-only**. `particles.json` is PDG 2008 data |
| Main dependencies (lock) | ampform 0.16.2, tensorwaves[jax] 0.4.17, jax 0.12.0.1, sympy 1.14.0, iminuit 2.33.0, numpy 2.5.4 (all Apache-2.0, BSD or MIT; iminuit also bundles LGPL Minuit2) |
| Repository rule | dependencies at least two weeks old (`exclude-newer` in `[tool.uv]`) |

Because of the GPL file, this repository should **reference** the options file through
the package (`aman_examples.d02kkpipi.OPTIONS`) rather than copy it.

## 2. Amplitude definition

- **Source of the model:** an AmpGen options file, `DtoKKpipi_v2.opt`. This is the
  LHCb model of JHEP 02 (2019) 126. It is parsed by `aman_examples.ampgen.options` and
  built as SymPy expressions by `aman_examples.ampgen.model.formulate_ampgen_model`.
  The entry point is `d02kkpipi.formulate_model()`, which returns an `AmpGenModel`.
- **Components:** 55 fully expanded decay chains (`model.chains: name → expression`,
  without couplings). Each chain's coupling is the **product** of the coupling symbols
  of the rules it uses (`model.couplings`), for example `C_{D0{K1(1270)+,K-}} ·
  C_{K1(1270)+{K*0,pi+}}`. The total is `model.amplitude = Σ_k c_k A_k`, and
  `model.intensity` is `AbsSquared(A)`. The chains are:
  - VV in S, P and D waves: K*(892)K̄*(892), K*(1680)K̄*(892), φρ, φω, φρ(1450), and
    K*K̄₂*(1430).
  - Resonance plus S-wave: φ·ππ (K-matrix), ρ/ω·KK̄ (K-matrix), K*(1680)·Kπ (FOCUS),
    a₀(980)·f₂(1270), φ·f₂(1270), and KK̄ S-wave·ππ S-wave.
  - Axial cascades: K₁(1270)± → K*π, Kπ S-wave, ρK, ρ(1450)K; K₁(1400)± → K*π;
    a₁(1260)± → φπ.
- **Couplings:** 25 free couplings of D0 rules (`C_{D0…}`), in polar coordinates. The
  remaining couplings are fixed. Some are aliases: K₁⁻ and a₁⁻ sub-decays equal the K₁⁺
  and a₁⁺ ones, with a phase of −π on some of them. `model.free_couplings` contains
  both the D0 rules and the sub-decay rules; the notebook keeps only the D0 ones.
- **Lineshapes** (`ampgen/lineshapes.py`): AmpGen Breit–Wigner with Blatt–Weisskopf
  factors, Gounaris–Sakurai (ρ), ρ–ω through `rhoOmega` pseudo-particles, Flatté with
  k-factor (a₀), the FOCUS Kπ S-wave (Kπ, I=3/2, Kη′ components), AmpGen's five-channel
  K-matrix pole and production terms for ππ and KK̄, and GSpline running widths for
  K₁(1270) and a₁(1260). The latter use 40-knot cubic splines read from the options
  file.
- **Spin factors** (`ampgen/covariant.py`): AmpGen's covariant vertices (`S_VV_S`,
  `S_VV_P`, `S_VV_D`, `S_VS_P`, `V_VS_S`, …) written in the four-momentum components.
  `S_VV_P` contains a Levi-Civita tensor and is the only parity-odd factor.
- **Validation:** every chain agrees with AmpGen to about 1e-5 of its rms on 2000
  reference events (`tests/test_ampgen_d02kkpipi.py`, using
  `tests/data/ampgen_d02kkpipi_v2_reference.npz`).

## 3. Data layout

- **Daughter order:** `(K+, K−, π+, π−)`, indices 0 to 3. `formulate_model` raises an
  error if the options file gives a different order.
- **Variables:** a flat `dict` of 16 float arrays named `p<i>_x, p<i>_y, p<i>_z, p<i>_E`
  (GeV). `model.data_from_momenta(arr)` converts an `(N, 16)` array ordered
  `(x, y, z, E)` per particle.
- **Frame:** the **D0 rest frame**. The phase-space generator produces events in the
  parent rest frame (*checked*: Σp = 0 and ΣE = m_D0 = 1.86484 GeV). The amplitude is
  Lorentz invariant, so any frame works, apart from the sign of the ε-tensor terms
  under parity. A lab-frame boost (Phase 5) therefore does not need to be undone before
  the amplitude is evaluated.
- **Masses:** AmpGen's 2008 PDG table (`model.table`), for example m_D0 = 1.86484 GeV.

## 4. CP conjugation

There is **no** CP-conjugation function for KKππ. The `mixing.conjugate_amplitude`
function is specific to KSππ (it swaps the Dalitz variables). The model has no D̄0
couplings: the options file describes D0 only.

The natural definition is CP of the event: p̄ = (K+ ↔ K−, π+ ↔ π−, **p⃗ → −p⃗**), so
that Ā_f(p) = Σ c_k A_k(p̄) without direct CP violation. Checked numerically on 10⁵
events:

- **Parity:** under p⃗ → −p⃗ alone, every chain is invariant except the 6 `D0[P]` chains
  (ε-tensor), which change sign. **The parity flip cannot be dropped**: swapping the
  charges alone is not CP.
- **CP map:** for most chains, A_k(p̄) = η_k A_{π(k)}(p), with η = ±1 exactly and a
  partner π(k). This includes self-conjugate chains such as φρ, K*K̄*, and the S-wave ×
  S-wave chains, and mirror pairs such as K₁⁺K⁻ ↔ K₁⁻K⁺, K*(1680)⁰KPi4 ↔
  K̄*(1680)⁰KPi5, the P-wave K*(1680)K̄* ↔ K̄*(1680)K*, and KPi10[Kpi]KPi00[I32] ↔
  KPi10[I32]KPi00[Kpi]. In the chains with a CP mirror the AmpGen conjugation phases
  are built into the options file, for example `_Im = … − PI`.
- **Four chains have no exact partner in the basis:**
  - `D0[D]{K*(1680)0, K*(892)bar0}` and `D0{K*(1680)0, K*(892)bar0}`: the
    K̄*(1680)K*(892) S- and D-wave chains are not in the model; only the P wave has
    both charge orderings.
  - `D0{K(2)*(1430)bar0, K*(892)0}`: there is no K₂*(1430)⁰K̄*(892) chain.
  - `D0{a1+[GSpline]{φπ+},π−}` ↔ `D0{a1−…,π+}`: the ratio is −1 only to about 1.3e-3,
    whereas the other pairs agree to 1e-6. The parameters of the two chains are
    identical. The cause is not yet understood; Phase 2 will pin it with a test.

  The CP map of Phase 2 therefore covers most of the 55 columns. For the remaining 3–4
  chains, A_k(p̄) has to be evaluated explicitly, or the missing mirror chains have to be
  added to the basis with zero D0 coupling.
- **F+ of the model** (*checked*, no uncertainty evaluated): F+ ≈ 0.731, with
  Im⟨A Ā*⟩/√(⟨|A|²⟩⟨|Ā|²⟩) ≈ 3e-4. This is consistent with the CLEO-c measurement of
  0.75 ± 0.04 (Malde et al.). ⟨|Ā|²⟩/⟨|A|²⟩ = 1.006 on this sample; the exact value is 1,
  because the map is measure-preserving, so the difference is a sampling fluctuation.

## 5. Toy generation

- **Phase space:** `aman_examples.phasespace.NBodyPhaseSpaceGenerator`, a NumPy port of
  ROOT's `TGenPhaseSpace` (Raubold–Lynch). Its weights are turned into **unweighted**
  events by accept-reject against the **analytic** maximum weight (`max_weight`, as in
  `TGenPhaseSpace::SetDecay`). This is an exact bound, so it never fails. Random
  numbers come from the TensorWaves `RealNumberGenerator`, for example
  `NumpyUniformRNG(seed)`.
- **|A|² sampling:** TensorWaves' `IntensityDistributionGenerator` does hit-or-miss in
  bunches (100k in the notebook). Each bunch is accepted against **its own maximum**
  (`_generate_bunch`), so the event at the bunch maximum is always kept. This
  over-represents the highest-|A|² points by about 1 event per bunch, or roughly
  1/(accepted per bunch) ≈ 0.1% for 100k bunches at this model's efficiency. A running
  maximum, 1.05 × the largest bunch maximum so far, is used only to decide when to
  discard everything and **restart**; it is never the accept-reject ceiling. There is
  no a-priori estimate and no option to raise an error instead. The notebook's executed
  output shows one restart. The generator returns only the momenta, so the amplitudes
  computed for each bunch are thrown away.
- **Efficiency** (*checked*): max/mean of |A|² on 10⁵ flat events is about 129, so the
  hit-or-miss efficiency from flat phase space is **below 0.8%**, worse than the 3%
  assumed in the plan. The notebook's 20k events take 71 s (on GPU, including JIT).
- **Mixing toys:** they exist only for KSππ (`docs/mixing.ipynb`, `mixing-cpv.ipynb`).
  Those notebooks use a uniform Dalitz plot plus `DalitzDecayTimeGenerator` (τ from
  e^{−t/τ}) and hit-or-miss on the rate relative to e^{−t/τ}. That is "accept-reject
  from flat", the approach the plan wants to avoid for KKππ.

## 6. Normalisation

- `tensorwaves.estimator.UnbinnedNLL(f, data, phsp)` computes
  −ln L = N ln⟨f⟩_phsp − Σ ln f(x_i): the normalisation is the **plain mean** over an
  unweighted sample. There are no per-event weights; `phsp_volume` only adds a
  constant.
- The notebook uses 10⁶ flat phase-space events. A large sample is needed because of
  the narrow φ(1020).
- `create_cached_function(expr, defaults, free_parameters=…)` splits the expression
  into the parts that depend on the free couplings and the precomputed data
  "variables". The data and phase-space samples are transformed once, and only the
  coupling algebra is evaluated in the fit (25 complex couplings: 4166 calls, 121 s).
- For KSππ mixing, `estimator.SimultaneousEstimator` sums the D0 and D̄0 NLLs and their
  gradients. `mixing.MixingModel` writes the rate as a small expression of the data
  variables `A_f`, `Ā_f` and `t`, which are precomputed by a `SympyDataTransformer`.
  This is the same trick that Phase 6 needs.

## 7. Backend

- SymPy expressions are lambdified by TensorWaves to **JAX** (the notebook uses
  `backend="jax"`). NumPy is also possible. There is no TensorFlow and no C++.
- **GPU:** works. JAX sees the CUDA device, and the optional extra
  `aman-examples[cuda]` installs `jax[cuda12]`. Float64 requires
  `jax.config.update("jax_enable_x64", True)`; TensorWaves does not set it.
- **Cost** (*checked*): lambdifying the 55 chains as separate functions takes about
  30 s, and the first call (JIT) takes about 40 s. After that, evaluating all 55 chains
  on 10⁵ events takes **0.24 s** on the GPU. Batch evaluation is the normal mode, since
  every function is vectorised over events.
- The optimiser is `tensorwaves.optimizer.Minuit2` (iminuit). It uses numerical
  gradients by default; JAX gradients are available through
  `use_analytic_gradient=True`.

## 8. Gaps for this project

| Needed | In the example? | Note |
|---|---|---|
| K-column component amplitudes `(N, K)` | partly | `model.chains` gives them as expressions; we need a small wrapper that lambdifies each chain and stacks the results. `create_cached_function` does a similar thing internally, but not as a plain `(N, K)` array |
| CP conjugation of events, Ā_f | no | Add p̄ = swap + parity, and a CP map (partner, η). 3–4 chains are unpaired (§4) |
| D̄0 couplings / direct CPV | no | c̄_k = c_k by default; per-component direct CPV is a later extension |
| Decay time and exact mixing for KKππ | partly | `mixing.py` has the exact ψ±, ψ_i rate and the (x, y, \|q/p\|, φ) ↔ (x_CP, y_CP, Δx, Δy) conversions, tested in `test_mixing_cpv.py`. It is SymPy-based and imports the KSππ module. The plan's cosh/sinh form is equivalent and can be tested against it |
| Mixing toy generator from the \|A\|² proposal | no | Only flat accept-reject (KSππ) and TensorWaves' adaptive hit-or-miss (no mixing). Phase 4's proposal needs \|A_f\|² samples, which come from TensorWaves' generator. That generator has an adaptive maximum and restarts, so a "c exceeded → error" policy has to be added on our side |
| LHCb-like acceptance (production, IP, HLT1) | no | `acceptance.py` and `parametric_acceptance.py` are KSππ-specific: a histogram ε(Dalitz, t) and a logistic fit to MC. Only the decay-time turn-on functional form might be reused. The emulator is new code |
| Lab-frame kinematics (pT, η, boost) | no | `phasespace._boost` exists but is private |
| Likelihood with external, acceptance-shaped normalisation, per tag | partly | `UnbinnedNLL` + `SimultaneousEstimator` give the structure. The normalisation sample has to be unweighted, drawn from flat × e^{−τ} × ε_model; this is consistent with the plan |
| (τ, A_k) storage, multiprocessing, study driver | no | New. Multiprocessing with JAX on one GPU needs care: one process per GPU, or the CPU backend in the workers |
| Tests of conjugation, Bose symmetry and F+ | no | New (Phase 2) |

## Update: release `0.1.0` (`9c40613`)

The tag adds commit #7, "faster D0 → K+K−π+π− toys and fits". This changes §5 and §6:

- `d02kkpipi.phase_space_generator` now returns `PairPhaseSpaceGenerator`, which
  generates (K+K−)(π+π−) pair masses followed by isotropic decays. It is still flat
  phase space, but its maximum weight now comes from a grid scan with a 1.02 safety
  factor; it is no longer the analytic bound.
- `d02kkpipi.peaked_phase_space_generator` adds Cauchy peaks at φ, ρ and ω. Each event
  comes with `density_ratio(sample)` r(p), and `ImportanceSampledFunction` gives I/r.
  The release notes quote about 10× faster toys. **This is the domain to use for our own
  |A_f|² sampler** (decision 5): accept-reject on |A|²/r with a fixed, pre-estimated
  maximum.
- `AmpGenModel.linear_amplitude(free)` gives the basis B_k = ∂A/∂c_k per free coupling,
  plus a constant B_0. `estimator.LinearAmplitudeNLL` turns the normalisation into a
  Gram matrix, which makes fits of the couplings without mixing fast. For mixing, the
  normalisation depends on τ through the mixing parameters, so a single Gram matrix is
  not enough. With fixed couplings, though, only A_f and Ā_f are needed per event, and
  the per-event cost is a few complex operations.

## Proposed decisions (to confirm)

1. **Dependency:** ~~pin `c1bc005`~~ → **confirmed: pin the tag `0.1.0`**.
2. **Reuse as is:** `d02kkpipi.formulate_model`, `phase_space_generator`,
   `model.chains`/`couplings`, `mixing.zcp_from_mixing`/`mixing_from_zcp` (as the
   reference for our conversion and for tests), `SimultaneousEstimator`, and TensorWaves
   lambdification on the JAX backend with float64.
3. **New, in this repository:**
   - `model.py`: the component matrix (N, K), CP conjugation of events, and the CP map
     with an explicit fallback for unpaired chains.
   - `timeevolution.py`: NumPy/JAX cosh/sinh, tested against `aman_examples.mixing`.
   - Our own likelihood on the precomputed (τ, A_k) arrays. The alternative is to wrap
     `MixingModel` + `UnbinnedNLL`; our own version keeps the per-tag normalisation and
     the (τ, A_k) caching explicit.
   - Generator, acceptance emulator, and driver.
4. **Unpaired chains:** evaluate A_k(p̄) explicitly for the 3–4 unpaired chains
   (K + 4 columns), rather than adding mirror chains with zero coupling. This keeps the
   model identical to AmpGen's.
5. **Proposal generator for Phase 4 (confirmed: our own sampler):** accept-reject on
   |A_f|²/r over `peaked_phase_space_generator`, with a fixed maximum estimated on a
   large sample times a safety factor. It raises an error if an event exceeds the
   maximum, and it returns the momenta together with the A_k of the accepted events.
6. **Open item:** the 1.3e-3 a₁⁺ ↔ a₁⁻ mismatch. Investigate in Phase 2 before trusting
   the CP map for those two chains.
