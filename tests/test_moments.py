"""The normalisation as decay-time moments."""

import numpy as np
import pytest

from d02hhhh_toys.fit import CouplingMixingNLL, MixingNLL, RateSample
from d02hhhh_toys.moments import (
    TRUNCATION_TOLERANCE,
    NormMoments,
    decay_time_moments,
)
from d02hhhh_toys.normalisation import normalisation_moments, normalisation_sample
from d02hhhh_toys.timeevolution import D0, D0BAR, MixingParameters, rate

REALISTIC = MixingParameters(0.004, 0.006, 0.0, 0.0)
INFLATED = MixingParameters.from_mixing(x=0.03, y=0.025, q_over_p=0.8, phase=-0.3)
LARGE = MixingParameters(0.15, -0.1, 0.05, 0.08)


def random_sample(n=5000, k=3, seed=1, tau_max=20.0):
    rng = np.random.default_rng(seed)
    tau = rng.exponential(size=n)
    tau = tau[tau < tau_max]
    n = len(tau)
    a = rng.normal(size=(n, k)) + 1j * rng.normal(size=(n, k))
    abar = rng.normal(size=(n, k)) + 1j * rng.normal(size=(n, k))
    c = np.array([1.0, 0.4 - 0.3j, -0.2 + 0.7j])[:k]
    return RateSample(tau, a @ c, abar @ c, rng.uniform(0.5, 2, size=n), (a, abar)), c


@pytest.mark.parametrize("parameters", [REALISTIC, INFLATED, LARGE])
@pytest.mark.parametrize("tag", [D0, D0BAR])
def test_moments_reproduce_the_event_sum(parameters, tag):
    sample, c = random_sample()
    direct = np.average(
        rate(
            sample.tau,
            sample.amplitude,
            sample.amplitude_bar,
            parameters,
            tag,
            relative_to_decay=True,
        ),
        weights=sample.weight,
    )
    moments = NormMoments.from_sample(sample)
    assert moments.truncation(parameters) < TRUNCATION_TOLERANCE
    assert moments.mean(parameters, tag) == pytest.approx(direct, rel=1e-12)
    # the same from the moments of the component amplitudes
    components = NormMoments.from_sample(sample, components=True)
    assert components.mean(parameters, tag, c) == pytest.approx(direct, rel=1e-12)


def test_moments_add_up():
    sample, _ = random_sample(n=3000)
    whole = NormMoments.from_sample(sample)
    halves = [
        NormMoments.from_sample(
            RateSample(
                sample.tau[s],
                sample.amplitude[s],
                sample.amplitude_bar[s],
                sample.weight[s],
            )
        )
        for s in (slice(0, 1000), slice(1000, None))
    ]
    total = halves[0] + halves[1]
    for name in ("aa", "bb", "ab", "tau_powers"):
        np.testing.assert_allclose(
            getattr(total, name), getattr(whole, name), rtol=1e-12
        )
    assert total.count == whole.count
    assert total.weight_sum == pytest.approx(whole.weight_sum)
    assert total.tau_max == whole.tau_max


def test_truncation_bound():
    moments = NormMoments.from_sample(random_sample()[0])
    assert moments.truncation(REALISTIC) < 1e-20
    assert moments.truncation(INFLATED) < 1e-15
    huge = MixingParameters(1.0, 1.0, 0.0, 0.0)  # far from any minimum
    assert moments.truncation(huge) > TRUNCATION_TOLERANCE


@pytest.mark.parametrize("tau_max", [3.0, 20.0, 40.0])
def test_decay_time_moments(tau_max):
    tau = np.linspace(0, tau_max, 2_000_001)
    density = np.exp(-tau)
    expected = [
        np.trapezoid(tau**n * density, tau) / np.trapezoid(density, tau)
        for n in range(21)
    ]
    np.testing.assert_allclose(decay_time_moments(tau_max, 20), expected, rtol=1e-8)


def test_new_decay_times_are_the_limit_of_permutations():
    """Replacing the decay times by independent ones: the mean over permutations."""
    sample, _ = random_sample(n=2000)
    moments = NormMoments.from_sample(sample)
    replaced = moments.with_decay_time_moments(moments.mean_tau_powers, moments.tau_max)
    rng = np.random.default_rng(2)
    permuted = [
        NormMoments.from_sample(
            RateSample(
                sample.tau[rng.permutation(len(sample))],
                sample.amplitude,
                sample.amplitude_bar,
                sample.weight,
            )
        ).mean(INFLATED, D0)
        for _ in range(200)
    ]
    spread = np.std(permuted) / np.sqrt(len(permuted))
    assert abs(np.mean(permuted) - replaced.mean(INFLATED, D0)) < 4 * spread
    np.testing.assert_array_equal(replaced.aa[0], moments.aa[0])


def test_nll_with_moments_matches_event_sum(subset_model):
    norm = normalisation_sample(subset_model, 20_000, seed=6, keep_components=True)
    data = {
        tag: RateSample(norm.tau[i::7], norm.amplitude[i::7], norm.amplitude_bar[i::7])
        for i, tag in ((0, D0), (1, D0BAR))
    }
    data_c = {
        tag: RateSample(
            norm.tau[i::7],
            norm.amplitude[i::7],
            norm.amplitude_bar[i::7],
            None,
            (norm.components[0][i::7], norm.components[1][i::7]),
        )
        for i, tag in ((0, D0), (1, D0BAR))
    }
    norms = {D0: norm, D0BAR: norm}
    for values in ([4.0, 6.0, 0.0, 0.0], [30.0, 25.0, 5.0, -3.0]):
        assert MixingNLL(data, norms)(values) == pytest.approx(
            MixingNLL(data, norms, moments=False)(values), rel=1e-12
        )
    free = [3]
    with_moments = CouplingMixingNLL(data_c, norms, subset_model.couplings, free)
    events = CouplingMixingNLL(
        data_c, norms, subset_model.couplings, free, moments=False
    )
    values = [30.0, 25.0, 5.0, -3.0, 0.3, -0.8]
    assert with_moments(values) == pytest.approx(events(values), rel=1e-12)
    with pytest.raises(TypeError, match="events"):
        MixingNLL(data, {D0: NormMoments.from_sample(norm), D0BAR: norm}, moments=False)
    with pytest.raises(ValueError, match="component"):
        CouplingMixingNLL(
            data_c,
            {D0: NormMoments.from_sample(norm), D0BAR: norm},
            subset_model.couplings,
            free,
        )


def test_streamed_moments_match_the_sample(subset_model):
    """Accumulated bunch by bunch, the moments are those of the same events."""
    kwargs = {"seed": [7, 1], "bunch_size": 4000}
    sample = normalisation_sample(subset_model, 10_000, **kwargs)
    streamed = normalisation_moments(subset_model, 10_000, **kwargs)
    whole = NormMoments.from_sample(sample)
    assert streamed.count == 10_000
    for name in ("aa", "bb", "ab", "tau_powers"):
        np.testing.assert_allclose(
            getattr(streamed, name), getattr(whole, name), rtol=1e-10
        )


@pytest.mark.slow
def test_v_factor_predicts_the_spread_of_fits(subset_model):
    """The delta-method v against fits of the same data with independent normalisations."""
    from aman_examples import d02kkpipi
    from tensorwaves.data.rng import NumpyUniformRNG

    from d02hhhh_toys.acceptance import AcceptanceConfig, LHCbAcceptance
    from d02hhhh_toys.fit import PARAMETERS, fit
    from d02hhhh_toys.generate import AmplitudeSampler, MixingGenerator
    from d02hhhh_toys.model import from_sample
    from d02hhhh_toys.normalisation import _decay_times
    from d02hhhh_toys.variance import variance_budget

    acceptance = LHCbAcceptance(AcceptanceConfig())
    generator = MixingGenerator(
        AmplitudeSampler(subset_model, estimate_size=200_000), INFLATED
    )
    n_data, n_norm, n_samples = 3000, 6000, 40
    data = {
        tag: RateSample.from_toy(
            generator.generate(tag, n_data, seed=20 + tag, acceptance=acceptance)
        )
        for tag in (D0, D0BAR)
    }
    values, errors = [], []
    for i in range(n_samples):
        m = normalisation_moments(subset_model, n_norm, [31, i], acceptance=acceptance)
        result = fit(MixingNLL(data, {D0: m, D0BAR: m}), INFLATED)
        values.append([result.values[n] for n in PARAMETERS])
        errors.append([result.errors[n] for n in PARAMETERS])
    measured = (
        np.var(values, axis=0, ddof=1) / np.mean(errors, axis=0) ** 2 * n_norm / n_data
    )
    domain = d02kkpipi.peaked_phase_space_generator(subset_model.model)
    k, replicas = 50_000, 8
    sample = domain.generate(k, NumpyUniformRNG(seed=5))
    p = from_sample(sample)
    a, abar = subset_model.amplitudes(p)
    rng = np.random.default_rng(5)
    tau = _decay_times(k * replicas, 20.0, rng)
    accepted = acceptance.accept(np.repeat(p, replicas, axis=0), tau, rng)
    budget = variance_budget(
        tau.reshape(k, replicas),
        a,
        abar,
        1 / domain.density_ratio(sample),
        accepted.reshape(k, replicas),
        INFLATED,
    )
    predicted = budget.v["per_event"]
    for name in ("y_cp_mix", "dx_mix"):  # the parameters with a large v
        i = PARAMETERS.index(name)
        assert 0.55 < measured[i] / predicted[i] < 1.6, (measured, predicted)


@pytest.mark.parametrize("tag", [D0, D0BAR])
def test_analytic_moments_converge_far_from_the_truth(tag):
    """At the ignore_time/none minima (y_CP ~ -0.42, Δx ~ -0.1) order 80 is exact."""
    from scipy.integrate import quad

    from d02hhhh_toys.moments import ANALYTIC_ORDER

    tau_max = 20.0
    one = np.ones((1, 1, 1))
    base = NormMoments(
        aa=1.3 * one, bb=0.8 * one, ab=(0.2 - 0.3j) * one,
        weight_sum=1.0, count=1, tau_powers=np.ones(1), tau_max=tau_max,
    )  # fmt: skip
    moments = base.with_decay_time_moments(
        decay_time_moments(tau_max, ANALYTIC_ORDER), tau_max
    )
    p = MixingParameters(x_cp_mix=6e-3, y_cp_mix=-0.42, dx_mix=-0.1, dy_mix=0.012)
    assert moments.order == ANALYTIC_ORDER
    assert moments.truncation(p) < 1e-15
    w = np.sqrt(complex(p.z_cp**2 - p.delta_z**2))
    zeta = p.z_cp + tag * p.delta_z
    aa, bb, ab = (1.3, 0.8, 0.2 - 0.3j) if tag == D0 else (0.8, 1.3, 0.2 + 0.3j)

    def integrand(t, part):
        c, s = np.cosh(w * t / 2), np.sinh(w * t / 2) / w
        r = abs(c) ** 2 * aa + abs(zeta * s) ** 2 * bb
        r += 2 * np.real(zeta * np.conj(c) * s * ab)
        return r * np.exp(-t)

    exact = quad(integrand, 0, tau_max, args=(0,), limit=200)[0] / -np.expm1(-tau_max)
    assert moments.mean(p, tag) == pytest.approx(exact, rel=1e-12)
