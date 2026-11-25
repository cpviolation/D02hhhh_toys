"""Phase 2: the amplitude adapter and its CP conventions."""

import numpy as np
import pytest

from conftest import K1680_KPI4, K1680BAR_KPI5, KST_KST_P, PHI_PIPI_S, PHI_RHO_S
from d02hhhh_toys.model import cp_conjugate, from_sample, to_sample

K1400_PLUS = "D0{K(1)(1400)+{K*(892)0{K+,pi-},pi+},K-}"
K1400_MINUS = "D0{K(1)(1400)bar-{K*(892)bar0{K-,pi+},pi-},K+}"
#: chains without an exact CP partner in DtoKKpipi_v2.opt (see the model docstring)
MISSING_MIRROR = {
    "D0[D]{K*(1680)0{K+,pi-},K*(892)bar0{K-,pi+}}",
    "D0{K*(1680)0{K+,pi-},K*(892)bar0{K-,pi+}}",
    "D0{K(2)*(1430)bar0{K-,pi+},K*(892)0{K+,pi-}}",
}


def invariant_mass_squared(events, indices):
    p = events[:, list(indices)].sum(axis=1)
    return p[:, 3] ** 2 - np.sum(p[:, :3] ** 2, axis=1)


@pytest.fixture(scope="module")
def subset(subset_model):
    return subset_model


@pytest.fixture(scope="module")
def events(subset):
    return from_sample(subset.phase_space(2000, seed=11))


def index(model, name):
    return model.names.index(name)


# --- event-level conventions (no amplitudes)


def test_sample_round_trip(events):
    np.testing.assert_array_equal(from_sample(to_sample(events)), events)
    sample = to_sample(events)
    np.testing.assert_array_equal(sample["p2_E"], events[:, 2, 3])  # pi+ energy


def test_cp_conjugation_is_an_involution(events):
    np.testing.assert_array_equal(cp_conjugate(cp_conjugate(events)), events)


def test_cp_conjugation_maps_invariant_masses(events):
    """m(K+pi-) of the conjugate is m(K-pi+) of the event, and so on."""
    conjugate = cp_conjugate(events)
    for pair, image in (((0, 3), (1, 2)), ((0, 1), (0, 1)), ((0, 2), (1, 3))):
        np.testing.assert_allclose(
            invariant_mass_squared(conjugate, pair),
            invariant_mass_squared(events, image),
        )


def test_cp_conjugation_flips_momenta(events):
    conjugate = cp_conjugate(events)
    np.testing.assert_array_equal(conjugate[:, 0, :3], -events[:, 1, :3])  # K+ <- K-
    np.testing.assert_array_equal(conjugate[:, 3, 3], events[:, 2, 3])  # pi- <- pi+


# --- CP map on a subset of chains


def test_cp_eigenstate_components(subset, events):
    """phi rho (S wave) is CP-even; K*K*bar (P wave) and phi (pipi)_S are CP-odd."""
    a = subset.component_amplitudes(events)
    a_bar = subset.component_amplitudes(cp_conjugate(events))
    for name, eta in ((PHI_RHO_S, 1), (KST_KST_P, -1), (PHI_PIPI_S, -1)):
        k = index(subset, name)
        assert subset.cp_map.partner[k] == k
        assert subset.cp_map.sign[k] == eta
        np.testing.assert_allclose(a_bar[:, k], eta * a[:, k], rtol=1e-9, atol=0)


def test_cp_mirror_components(subset, events):
    a = subset.component_amplitudes(events)
    a_bar = subset.component_amplitudes(cp_conjugate(events))
    k, m = index(subset, K1680_KPI4), index(subset, K1680BAR_KPI5)
    assert subset.cp_map.partner[k] == m
    assert subset.cp_map.partner[m] == k
    np.testing.assert_allclose(a_bar[:, k], subset.cp_map.sign[k] * a[:, m], rtol=1e-9)


def test_parity_flip_is_needed(subset, events):
    """Swapping the charges without p -> -p is not CP for the P-wave chain."""
    k = index(subset, KST_KST_P)
    swapped = events[:, [1, 0, 3, 2], :]
    a = subset.component_amplitudes(events)[:, k]
    np.testing.assert_allclose(subset.component_amplitudes(swapped)[:, k], a, rtol=1e-9)
    np.testing.assert_allclose(
        subset.component_amplitudes(cp_conjugate(events))[:, k], -a, rtol=1e-9
    )


def test_cp_map_is_an_involution(subset):
    cp = subset.cp_map
    paired = np.flatnonzero(cp.paired)
    np.testing.assert_array_equal(cp.partner[cp.partner[paired]], paired)
    np.testing.assert_array_equal(cp.sign[paired] * cp.sign[cp.partner[paired]], 1)


def test_conjugate_components_match_explicit_evaluation(subset, events):
    _, conjugates = subset.component_pair(events)
    explicit = subset.component_amplitudes(cp_conjugate(events))
    np.testing.assert_allclose(conjugates, explicit, rtol=1e-9, atol=0)


def test_amplitude_bar_is_amplitude_of_conjugate(subset, events):
    a, a_bar = subset.amplitudes(events)
    a_of_conjugate, _ = subset.amplitudes(cp_conjugate(events))
    np.testing.assert_allclose(a_bar, a_of_conjugate, rtol=1e-9)
    np.testing.assert_allclose(
        a, subset.component_amplitudes(events) @ subset.couplings
    )


@pytest.mark.parametrize(
    ("couplings", "expected"),
    [
        ({PHI_RHO_S: 1}, 1.0),  # CP-even eigenstate
        ({KST_KST_P: 1, PHI_PIPI_S: 0.5j}, 0.0),  # CP-odd eigenstates
        # A_k + eta_k A_pi(k) is CP-even, A_k - eta_k A_pi(k) is CP-odd
        ({K1680_KPI4: 1, K1680BAR_KPI5: "+eta"}, 1.0),
        ({K1680_KPI4: 1, K1680BAR_KPI5: "-eta"}, 0.0),
        ({PHI_RHO_S: 1, K1680_KPI4: 0.3, K1680BAR_KPI5: "+eta"}, 1.0),
    ],
)
def test_cp_even_fraction_of_eigenstates(subset, couplings, expected):
    eta = subset.cp_map.sign[index(subset, K1680_KPI4)]
    values = dict.fromkeys(subset.names, 0.0)
    for name, value in couplings.items():
        if isinstance(value, str):
            value = couplings[K1680_KPI4] * eta * (1 if value[0] == "+" else -1)
        values[name] = value
    original = subset.couplings.copy()
    try:
        subset.set_couplings(values)
        f_plus, error = subset.cp_even_fraction(size=20_000, seed=3)
    finally:
        subset.couplings = original
    assert f_plus == pytest.approx(expected, abs=1e-10)
    assert error == pytest.approx(0, abs=1e-10)


# --- full model


@pytest.fixture(scope="module")
def full(full_model):
    return full_model


@pytest.mark.slow
def test_full_model_unpaired_chains(full):
    """Only the chains of the model docstring lack an exact CP partner."""
    unpaired = {full.names[k] for k in full.cp_map.unpaired}
    spline = {n for n in full.names if "K(1)(1270)" in n or "a(1)(1260)" in n}
    assert unpaired == MISSING_MIRROR | spline
    assert len(full.names) == 55


@pytest.mark.slow
def test_full_model_mirror_k1400(full):
    k, m = index(full, K1400_PLUS), index(full, K1400_MINUS)
    assert full.cp_map.partner[k] == m
    assert full.cp_map.sign[k] == 1


@pytest.mark.slow
def test_full_model_conjugate_components(full, events):
    _, conjugates = full.component_pair(events)
    explicit = full.component_amplitudes(cp_conjugate(events))
    np.testing.assert_allclose(conjugates, explicit, rtol=1e-9, atol=0)


@pytest.mark.slow
def test_full_model_cp_even_fraction(full):
    """F+ of the LHCb model (about 0.73; CLEO-c measured 0.75 +- 0.04)."""
    f_plus, error = full.cp_even_fraction(size=200_000, seed=5)
    assert error < 0.005
    assert 0.70 < f_plus < 0.76
