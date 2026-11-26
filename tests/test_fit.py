"""Phase 6: the likelihood with an external normalisation sample and the acceptance."""

import numpy as np
import pytest

from conftest import BACKEND, K1680BAR_KPI5, PHI_PIPI_S
from d02hhhh_toys.acceptance import HLT1, AcceptanceConfig, BoxLine, LHCbAcceptance
from d02hhhh_toys.fit import CouplingMixingNLL, MixingNLL, RateSample, fit
from d02hhhh_toys.generate import AmplitudeSampler, MixingGenerator
from d02hhhh_toys.model import from_sample
from d02hhhh_toys.normalisation import normalisation_sample
from d02hhhh_toys.timeevolution import D0, D0BAR, MixingParameters

INFLATED = MixingParameters.from_mixing(x=0.03, y=0.025, q_over_p=0.8, phase=-0.3)


def loose_acceptance():
    """Depends on tau and on phase space, but keeps about half of the events (fast tests)."""
    config = AcceptanceConfig(
        hlt1=HLT1([BoxLine("loose", 1, 0.5, 4.0)])
    ).with_overrides(
        {
            "tracks": {"eta_min": -10.0, "eta_max": 10.0, "p_min": 0.0, "pt_min": 0.0},
            "tracking": {"plateau": 1.0, "threshold": 0.0},
            "decay_vertex.rho_max_um": None,
        }
    )
    return LHCbAcceptance(config)


@pytest.fixture(scope="module")
def generator(subset_model):
    sampler = AmplitudeSampler(subset_model, estimate_size=200_000, bunch_size=100_000)
    return MixingGenerator(sampler, INFLATED)


def m2(events, i, j):
    p = events[:, i] + events[:, j]
    return p[:, 3] ** 2 - np.sum(p[:, :3] ** 2, axis=1)


def test_loose_acceptance_is_not_trivial(subset_model):
    acc = loose_acceptance()
    events = from_sample(subset_model.phase_space(20_000, seed=1))
    short = acc.accept(
        events, np.full(len(events), 0.2), np.random.default_rng(1)
    ).mean()
    long = acc.accept(
        events, np.full(len(events), 3.0), np.random.default_rng(1)
    ).mean()
    assert 0.05 < short < 0.6 * long
    assert long > 0.3


@pytest.mark.slow
def test_acceptance_before_amplitudes_is_unbiased(generator):
    """Generating with the acceptance applied before the amplitudes gives the same sample
    as generating without it and applying it afterwards."""
    acc = loose_acceptance()
    inside = generator.generate(D0, 15_000, seed=2, acceptance=acc)
    full = generator.generate(D0, 40_000, seed=3)
    after = full.tau[acc.accept(full.events, full.tau, np.random.default_rng(4))]
    events_after = full.events[
        acc.accept(full.events, full.tau, np.random.default_rng(4))
    ]
    for x, y in (
        (inside.tau, after),
        (m2(inside.events, 0, 1), m2(events_after, 0, 1)),
    ):
        error = np.hypot(x.std() / np.sqrt(len(x)), y.std() / np.sqrt(len(y)))
        assert abs(x.mean() - y.mean()) < 4 * error


@pytest.mark.slow
def test_normalisation_methods_agree(subset_model, generator):
    """The three weighted samples estimate the same normalisation ratios."""
    acc = loose_acceptance()
    estimates = {}
    for method, size in (("flat", 200_000), ("peaked", 100_000), ("proposal", 20_000)):
        norm = normalisation_sample(
            subset_model,
            size,
            seed=5,
            acceptance=acc,
            method=method,
            generator=generator,
        )
        w = norm.weight
        den = w * (np.abs(norm.amplitude) ** 2 + np.abs(norm.amplitude_bar) ** 2) / 2
        num = w * (np.conj(norm.amplitude) * norm.amplitude_bar).real
        ratio = num.mean() / den.mean()
        error = np.std((num - ratio * den) / den.mean()) / np.sqrt(len(w))
        tau = np.average(norm.tau, weights=den)
        estimates[method] = (ratio, error, tau)
    for a in estimates:
        for b in estimates:
            (ra, ea, _), (rb, eb, _) = estimates[a], estimates[b]
            assert abs(ra - rb) < 4 * np.hypot(ea, eb), estimates


def test_normalisation_validation(subset_model):
    with pytest.raises(ValueError, match="method"):
        normalisation_sample(subset_model, 10, seed=0, method="uniform")
    with pytest.raises(ValueError, match="MixingGenerator"):
        normalisation_sample(subset_model, 10, seed=0, method="proposal")


def test_coupling_nll_matches_fixed_nll(subset_model, generator):
    data = {
        tag: RateSample.from_toy(
            generator.generate(tag, 2000, seed=10 + tag, keep_components=True)
        )
        for tag in (D0, D0BAR)
    }
    norm = normalisation_sample(subset_model, 20_000, seed=6, keep_components=True)
    fixed = MixingNLL(data, {D0: norm, D0BAR: norm})
    k = subset_model.names.index(K1680BAR_KPI5)
    free = CouplingMixingNLL(data, {D0: norm, D0BAR: norm}, subset_model.couplings, [k])
    values = [4.0, 6.0, 1.0, -0.5]
    assert free(values + free.initial(MixingParameters())[4:]) == pytest.approx(
        fixed(values), rel=1e-12
    )
    assert free.names[4:] == (f"c{k}_re", f"c{k}_im")
    with pytest.raises(ValueError, match="stay fixed"):
        CouplingMixingNLL(
            data, {D0: norm, D0BAR: norm}, subset_model.couplings, list(range(5))
        )
    no_components = {
        tag: RateSample(s.tau, s.amplitude, s.amplitude_bar) for tag, s in data.items()
    }
    with pytest.raises(ValueError, match="components"):
        CouplingMixingNLL(
            no_components, {D0: norm, D0BAR: norm}, subset_model.couplings, [k]
        )


def pulls(result, truth):
    return {n: (result.values[n] - truth[n]) / result.errors[n] for n in truth}


def mixing_truth():
    return {
        n: getattr(INFLATED, n) / 1e-3
        for n in ("x_cp_mix", "y_cp_mix", "dx_mix", "dy_mix")
    }


def closure_with_acceptance(model, generator, acceptance, size, n_norm, seed):
    """Data after the true acceptance; normalisation after the same acceptance model."""
    seeds = np.random.SeedSequence(seed).spawn(3)
    data = {
        tag: RateSample.from_toy(
            generator.generate(tag, size, s, acceptance=acceptance)
        )
        for tag, s in zip((D0, D0BAR), seeds[:2], strict=True)
    }
    norm = normalisation_sample(model, n_norm, seeds[2], acceptance=acceptance)
    result = fit(MixingNLL(data, {D0: norm, D0BAR: norm}))
    assert result.valid
    p = pulls(result, mixing_truth())
    assert all(abs(v) < 4 for v in p.values()), (p, result.values, result.errors)
    return result


@pytest.mark.slow
def test_single_toy_closure_with_acceptance(subset_model, generator):
    """The plan's Phase 6 test, with the default (toy) LHCb-like acceptance."""
    closure_with_acceptance(
        subset_model, generator, LHCbAcceptance(), 15_000, 150_000, seed=41
    )


@pytest.mark.slow
@pytest.mark.skipif(
    BACKEND != "jax", reason="too slow on CPU: run with D02HHHH_TEST_BACKEND=jax"
)
def test_single_toy_closure_with_acceptance_full_model(full_model):
    sampler = AmplitudeSampler(full_model, estimate_size=300_000)
    generator = MixingGenerator(sampler, INFLATED)
    closure_with_acceptance(
        full_model, generator, LHCbAcceptance(), 30_000, 300_000, seed=42
    )


@pytest.mark.slow
def test_single_toy_closure_with_free_couplings(subset_model, generator):
    """Mixing parameters and two complex couplings free, with the loose acceptance."""
    acc = loose_acceptance()
    seeds = np.random.SeedSequence(43).spawn(3)
    data = {
        tag: RateSample.from_toy(
            generator.generate(tag, 20_000, s, acceptance=acc, keep_components=True)
        )
        for tag, s in zip((D0, D0BAR), seeds[:2], strict=True)
    }
    norm = normalisation_sample(
        subset_model, 150_000, seeds[2], acceptance=acc, keep_components=True
    )
    free = [subset_model.names.index(n) for n in (K1680BAR_KPI5, PHI_PIPI_S)]
    nll = CouplingMixingNLL(data, {D0: norm, D0BAR: norm}, subset_model.couplings, free)
    shifted = MixingParameters()  # mixing starts at zero; couplings at the truth
    result = fit(nll, shifted)
    assert result.valid
    truth = mixing_truth()
    for k in free:
        truth[f"c{k}_re"] = subset_model.couplings[k].real
        truth[f"c{k}_im"] = subset_model.couplings[k].imag
    p = pulls(result, truth)
    assert all(abs(v) < 4 for v in p.values()), (p, result.values, result.errors)


def test_nll_is_finite_far_from_the_minimum(subset_model, generator):
    data = {
        tag: RateSample.from_toy(generator.generate(tag, 500, seed=20 + tag))
        for tag in (D0, D0BAR)
    }
    norm = normalisation_sample(subset_model, 5000, seed=21)
    nll = MixingNLL(data, {D0: norm, D0BAR: norm})
    with np.errstate(all="raise"):
        value = nll([1e5, -1e5, 1e5, 1e5])  # |w| tau/2 >> 700 at large tau
    assert np.isfinite(value)
