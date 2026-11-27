"""Phase 4: toy generation with mixing."""

import numpy as np
import pytest

from conftest import BACKEND, PHI_PIPI_S, PHI_RHO_S
from d02hhhh_toys.fit import MixingNLL, RateSample, fit
from d02hhhh_toys.generate import (
    AmplitudeSampler,
    MaximumExceededError,
    MixingGenerator,
)
from d02hhhh_toys.model import from_sample
from d02hhhh_toys.timeevolution import D0, D0BAR, MixingParameters

INFLATED = MixingParameters.from_mixing(x=0.03, y=0.025, q_over_p=0.8, phase=-0.3)


def m2(events, i, j):
    p = events[:, i] + events[:, j]
    return p[:, 3] ** 2 - np.sum(p[:, :3] ** 2, axis=1)


@pytest.fixture(scope="module")
def sampler(subset_model):
    return AmplitudeSampler(subset_model, estimate_size=200_000, bunch_size=100_000)


def with_couplings(model, couplings):
    """Copy of the couplings of ``model`` with only ``couplings`` non-zero."""
    values = np.zeros_like(model.couplings)
    for name, value in couplings.items():
        values[model.names.index(name)] = value
    return values


def test_sampler_follows_intensity(sampler, subset_model):
    """Means of m^2(K+K-), m^2(pi+pi-), m^2(K+pi-) agree with weighted flat phase space."""
    toy = sampler.generate(30_000, seed=1)
    flat = from_sample(subset_model.phase_space(300_000, seed=2))
    weight = (
        np.abs(subset_model.component_amplitudes(flat) @ subset_model.couplings) ** 2
    )
    for i, j in ((0, 1), (2, 3), (0, 3)):
        x, ref = m2(toy.events, i, j), m2(flat, i, j)
        mean_ref = np.average(ref, weights=weight)
        # error of a ratio estimator, and of the toy mean
        err_ref = np.sqrt(np.sum((weight * (ref - mean_ref)) ** 2)) / weight.sum()
        err = np.hypot(x.std() / np.sqrt(len(x)), err_ref)
        assert abs(x.mean() - mean_ref) < 4 * err


def test_sampler_stores_components(sampler, subset_model):
    toy = sampler.generate(1000, seed=3)
    np.testing.assert_allclose(
        toy.components, subset_model.component_amplitudes(toy.events)
    )
    conj = toy.conjugated()
    np.testing.assert_allclose(
        conj.components, subset_model.component_amplitudes(conj.events)
    )


def test_sampler_raises_above_maximum(sampler):
    saved = sampler.maximum
    sampler.maximum = 0.1 * sampler.estimated_maximum
    try:
        with pytest.raises(MaximumExceededError):
            sampler.generate(1000, seed=4)
    finally:
        sampler.maximum = saved


def test_seeds_are_reproducible(sampler):
    generator = MixingGenerator(sampler, INFLATED)
    a, b = (generator.generate(D0, 500, seed=9) for _ in range(2))
    np.testing.assert_array_equal(a.tau, b.tau)
    np.testing.assert_array_equal(a.events, b.events)
    c = generator.generate(D0, 500, seed=10)
    assert not np.array_equal(a.tau, c.tau)


def test_bound_is_close_to_two(sampler):
    generator = MixingGenerator(sampler, MixingParameters(0.004, 0.006, 0.0, 0.0))
    assert 2 < generator.bound[D0] < 2.05


@pytest.mark.slow
@pytest.mark.parametrize(
    ("couplings", "sign"),
    [({PHI_RHO_S: 1}, +1), ({PHI_PIPI_S: 1}, -1)],
    ids=["CP-even", "CP-odd"],
)
@pytest.mark.parametrize("tag", [D0, D0BAR])
def test_decay_time_slope_of_cp_eigenstate(sampler, couplings, sign, tag):
    """A CP-even (odd) final state decays with Gamma (1 + y_CP) (Gamma (1 - y_CP)).

    With z_CP = -(y_CP + i x_CP) and no CP violation, R = exp(-+w tau/2) A for
    Abar = +-A, so the rate is exp(-(1 +- y_CP) tau). The truncation at tau_max = 20 is
    negligible. y_CP is inflated, so that the slope differs from 1 by many sigma.
    """
    y_cp = 0.05  # 1 +- y_CP are 17 sigma apart with 30k events
    params = MixingParameters(x_cp_mix=0.03, y_cp_mix=y_cp)
    model = sampler.model
    saved = model.couplings
    model.couplings = with_couplings(model, couplings)
    try:
        cp_sampler = AmplitudeSampler(model, estimate_size=100_000, bunch_size=100_000)
        toy = MixingGenerator(cp_sampler, params).generate(tag, 30_000, seed=21)
    finally:
        model.couplings = saved
    gamma = 1 / toy.tau.mean()  # MLE of the exponential slope
    error = gamma / np.sqrt(len(toy))
    assert abs(gamma - (1 + sign * y_cp)) < 4 * error
    assert abs(gamma - (1 - sign * y_cp)) > 10 * error


def closure(model, size, n_norm, seed):
    sampler = AmplitudeSampler(model, estimate_size=200_000)
    generator = MixingGenerator(sampler, INFLATED)
    seeds = np.random.SeedSequence(seed).spawn(3)
    data = {
        tag: RateSample.from_toy(generator.generate(tag, size, s))
        for tag, s in zip((D0, D0BAR), seeds[:2], strict=True)
    }
    norm = RateSample.from_toy(
        generator.proposal(n_norm, seeds[2]), proposal_weights=True
    )
    result = fit(MixingNLL(data, {D0: norm, D0BAR: norm}))
    assert result.valid
    pulls = {
        n: (result.values[n] - getattr(INFLATED, n) / 1e-3) / result.errors[n]
        for n in result.values
    }
    assert all(abs(p) < 4 for p in pulls.values()), (
        pulls,
        result.values,
        result.errors,
    )
    return result


@pytest.mark.slow
def test_single_toy_closure_without_acceptance(subset_model):
    closure(subset_model, size=20_000, n_norm=100_000, seed=31)


@pytest.mark.slow
@pytest.mark.skipif(
    BACKEND != "jax", reason="too slow on CPU: run with D02HHHH_TEST_BACKEND=jax"
)
def test_single_toy_closure_full_model(full_model):
    closure(full_model, size=40_000, n_norm=200_000, seed=32)


def test_seed_sequence_is_not_consumed(sampler):
    """The same SeedSequence object gives the same toy twice (spawn() would not)."""
    generator = MixingGenerator(sampler, INFLATED)
    seed = np.random.SeedSequence(99)
    a = generator.generate(D0, 300, seed)
    b = generator.generate(D0, 300, seed)
    np.testing.assert_array_equal(a.tau, b.tau)


def test_fixed_maximum(sampler, subset_model):
    fixed = AmplitudeSampler(
        subset_model, estimate_size=20_000, maximum=2 * sampler.estimated_maximum
    )
    assert fixed.maximum == 2 * sampler.estimated_maximum
    with pytest.raises(ValueError, match="below the estimated"):
        AmplitudeSampler(subset_model, estimate_size=20_000, maximum=1e-6)
