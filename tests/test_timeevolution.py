"""Phase 3: exact time evolution and the mixing-parameter conventions."""

import numpy as np
import pytest
import scipy.linalg
import sympy as sp
from aman_examples import mixing as upstream

from d02hhhh_toys.timeevolution import (
    D0,
    D0BAR,
    MixingParameters,
    evolved_amplitude,
    rate,
    time_functions,
)

#: non-trivial mixing and CP violation in mixing, much larger than in nature
X, Y, QOP, PHI = 0.03, 0.025, 0.8, -0.3


@pytest.fixture
def amplitudes():
    rng = np.random.default_rng(7)
    a = rng.normal(size=50) + 1j * rng.normal(size=50)
    a_bar = rng.normal(size=50) + 1j * rng.normal(size=50)
    tau = rng.exponential(size=50) * 3
    return tau, a, a_bar


def diagonalised_amplitudes(tau, a, a_bar, x, y, q_over_p, phase):
    """D0 and D0bar amplitudes from exp(-i H t) of the 2x2 effective Hamiltonian.

    |D_{1,2}> = p|D0> +- q|D0bar>, with m_{1,2} = +-x/2 and Gamma_{1,2} = 1 +- y, in units
    of Gamma (mean mass 0): the eigenvalues of H = M - i Gamma/2 are m_k - i Gamma_k/2.
    """
    qp = q_over_p * np.exp(1j * phase)
    p = 1 / np.sqrt(1 + abs(qp) ** 2)
    q = qp * p
    eigenvectors = np.array([[p, p], [q, -q]])  # columns |D1>, |D2> in (D0, D0bar)
    eigenvalues = np.array([x / 2 - 0.5j * (1 + y), -x / 2 - 0.5j * (1 - y)])
    hamiltonian = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
    out = {D0: [], D0BAR: []}
    for t, af, afb in zip(tau, a, a_bar, strict=True):
        u = scipy.linalg.expm(-1j * hamiltonian * t)
        # column 0: initial D0, column 1: initial D0bar; rows: D0, D0bar components
        out[D0].append(u[0, 0] * af + u[1, 0] * afb)
        out[D0BAR].append(u[0, 1] * af + u[1, 1] * afb)
    return {k: np.array(v) for k, v in out.items()}


@pytest.mark.parametrize("tag", [D0, D0BAR])
def test_explicit_diagonalisation(amplitudes, tag):
    tau, a, a_bar = amplitudes
    params = MixingParameters.from_mixing(X, Y, QOP, PHI)
    expected = diagonalised_amplitudes(tau, a, a_bar, X, Y, QOP, PHI)[tag]
    r = evolved_amplitude(tau, a, a_bar, params, tag)
    np.testing.assert_allclose(r * np.exp(-tau / 2), expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        rate(tau, a, a_bar, params, tag), np.abs(expected) ** 2, rtol=1e-12
    )


@pytest.mark.parametrize("tag", [D0, D0BAR])
def test_agrees_with_aman_examples(amplitudes, tag):
    """Same rate as aman_examples.mixing.formulate_mixing_model_zcp."""
    tau, a, a_bar = amplitudes
    params = MixingParameters.from_mixing(X, Y, QOP, PHI)
    model = upstream.formulate_mixing_model_zcp(
        params.x_cp_mix,
        params.y_cp_mix,
        params.dx_mix,
        params.dy_mix,
        initial_d0=tag == D0,
        relative_to_decay=False,
    )
    function = sp.lambdify(
        (upstream.t, upstream.amplitude_symbol, upstream.amplitude_bar_symbol),
        model.intensity.xreplace(model.parameter_defaults),
        "numpy",
    )
    np.testing.assert_allclose(
        rate(tau, a, a_bar, params, tag), function(tau, a, a_bar), rtol=1e-10
    )


def test_first_order_expansion(amplitudes):
    tau, a, a_bar = amplitudes
    tau = tau * 1e-3
    params = MixingParameters(0.04, 0.06, 0.01, -0.02)
    z, dz = params.z_cp, params.delta_z
    w2 = z**2 - dz**2
    for tag, direct, mixed, zz in ((D0, a, a_bar, z + dz), (D0BAR, a_bar, a, z - dz)):
        r = evolved_amplitude(tau, a, a_bar, params, tag)
        first = direct + tau / 2 * zz * mixed
        # the residual of each order is bounded by the next term of the series
        next_term = tau**2 / 8 * abs(w2) * np.abs(direct)
        assert np.all(np.abs(r - first) <= 1.01 * next_term)
        second = first + tau**2 / 8 * w2 * direct
        third = tau**3 / 48 * abs(w2) * np.abs(zz * mixed)
        assert np.all(np.abs(r - second) <= 1.01 * third + 4e-16 * np.abs(direct))


def test_no_mixing_is_identity(amplitudes):
    tau, a, a_bar = amplitudes
    none = MixingParameters()
    np.testing.assert_allclose(evolved_amplitude(tau, a, a_bar, none, D0), a)
    np.testing.assert_allclose(evolved_amplitude(tau, a, a_bar, none, D0BAR), a_bar)
    np.testing.assert_allclose(
        rate(tau, a, a_bar, none, D0), np.abs(a) ** 2 * np.exp(-tau)
    )


def test_series_matches_closed_form():
    tau = np.array([1e-6, 1e-4, 0.5, 2.0])
    z, dz = -(0.006 + 0.004j), 0.0005j
    cosh, sinhc = time_functions(tau, z, dz)
    w = np.sqrt(z**2 - dz**2 + 0j)
    np.testing.assert_allclose(cosh, np.cosh(w * tau / 2), rtol=1e-14)
    np.testing.assert_allclose(sinhc, np.sinh(w * tau / 2) / w, rtol=1e-12)


def test_conversion_without_cp_violation():
    params = MixingParameters.from_mixing(0.004, 0.006)
    assert params.x_cp_mix == pytest.approx(0.004)
    assert params.y_cp_mix == pytest.approx(0.006)
    assert params.dx_mix == pytest.approx(0, abs=1e-15)
    assert params.dy_mix == pytest.approx(0, abs=1e-15)


def test_conversion_matches_aman_examples_and_round_trips():
    params = MixingParameters.from_mixing(X, Y, QOP, PHI)
    ref = upstream.zcp_from_mixing(X, Y, QOP, PHI)
    assert params.x_cp_mix == pytest.approx(ref["x_CP"])
    assert params.y_cp_mix == pytest.approx(ref["y_CP"])
    assert params.dx_mix == pytest.approx(ref["Delta_x"])
    assert params.dy_mix == pytest.approx(ref["Delta_y"])
    back = params.to_mixing()
    for key, value in (("x", X), ("y", Y), ("q_over_p", QOP), ("phase", PHI)):
        assert back[key] == pytest.approx(value)


def test_jax_backend(amplitudes):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    tau, a, a_bar = amplitudes
    params = MixingParameters(0.004, 0.006, 0.001, -0.0005)

    @jax.jit
    def f(x_cp, y_cp, dx, dy):
        p = MixingParameters(x_cp, y_cp, dx, dy)
        return rate(jnp.asarray(tau), jnp.asarray(a), jnp.asarray(a_bar), p, D0, xp=jnp)

    np.testing.assert_allclose(
        np.asarray(f(0.004, 0.006, 0.001, -0.0005)),
        rate(tau, a, a_bar, params, D0),
        rtol=1e-12,
    )
