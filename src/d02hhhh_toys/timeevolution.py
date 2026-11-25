r"""Exact time evolution with :math:`D^0`-:math:`\bar{D}^0` mixing, in :math:`(x_{CP}, y_{CP}, \Delta x, \Delta y)`.

Conventions (HFLAV; as :mod:`aman_examples.mixing`): :math:`|D_{1,2}\rangle = p|D^0\rangle
\pm q|\bar{D}^0\rangle`, :math:`x = (m_1 - m_2)/\Gamma`, :math:`y = (\Gamma_1 -
\Gamma_2)/(2\Gamma)`, and

.. math::

    z_{CP} \pm \Delta z = -(q/p)^{\pm 1}(y + ix), \quad
    x_{CP} = -\mathrm{Im}\,z_{CP},\ y_{CP} = -\mathrm{Re}\,z_{CP},\quad
    \Delta x = -\mathrm{Im}\,\Delta z,\ \Delta y = -\mathrm{Re}\,\Delta z.

With :math:`w = y + ix`, :math:`w^2 = z_{CP}^2 - \Delta z^2` and :math:`\tau = \Gamma t`,
the amplitudes of an initial :math:`D^0` and :math:`\bar{D}^0` (without the common
factor :math:`e^{-\tau/2}e^{-im\tau/\Gamma}`) are

.. math::

    R_{D^0} = \cosh\frac{w\tau}{2} A_f + \frac{\sinh(w\tau/2)}{w}(z_{CP} + \Delta z)\bar{A}_f,
    \qquad
    R_{\bar{D}^0} = \cosh\frac{w\tau}{2} \bar{A}_f + \frac{\sinh(w\tau/2)}{w}(z_{CP} - \Delta z) A_f,

and the rate is :math:`e^{-\tau}|R|^2`. Both :math:`\cosh(w\tau/2)` and
:math:`\sinh(w\tau/2)/w` are even in :math:`w`, so only :math:`w^2` is needed and there
is no branch to choose.

The functions take an array module ``xp`` (NumPy by default, or ``jax.numpy``), so that
the same code is used in generation and in the likelihood. The mixing parameters are
plain numbers here (not in units of 1e-3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Tags: initial D0 and initial D0bar
D0, D0BAR = +1, -1
_SERIES_LIMIT = 1e-3  # |w tau/2| below which the series of sinh(u)/u is used


@dataclass(frozen=True)
class MixingParameters:
    """:math:`x_{CP}, y_{CP}, \\Delta x, \\Delta y` (plain numbers, not in units of 1e-3)."""

    x_cp_mix: float = 0.0
    y_cp_mix: float = 0.0
    dx_mix: float = 0.0
    dy_mix: float = 0.0

    # written as arithmetic, not complex(), so that the fields can be JAX tracers
    @property
    def z_cp(self) -> complex:
        return -(self.y_cp_mix + 1j * self.x_cp_mix)

    @property
    def delta_z(self) -> complex:
        return -(self.dy_mix + 1j * self.dx_mix)

    @classmethod
    def from_mixing(
        cls, x: float, y: float, q_over_p: float = 1.0, phase: float = 0.0
    ) -> MixingParameters:
        r"""From :math:`x, y, |q/p|, \phi = \arg(q/p)`."""
        qp = q_over_p * np.exp(1j * phase)
        w = complex(y, x)
        z_plus, z_minus = -qp * w, -w / qp  # z_CP + Delta z, z_CP - Delta z
        z, dz = (z_plus + z_minus) / 2, (z_plus - z_minus) / 2
        return cls(-z.imag, -z.real, -dz.imag, -dz.real)

    def to_mixing(self) -> dict[str, float]:
        r""":math:`x, y, |q/p|, \phi`, on the branch where :math:`y + ix \to y_{CP} + ix_{CP}` without CP violation."""
        z_plus, z_minus = self.z_cp + self.delta_z, self.z_cp - self.delta_z
        base = -self.z_cp
        w = base * np.sqrt(z_plus * z_minus / base**2)
        qp = -z_plus / w
        return {
            "x": w.imag,
            "y": w.real,
            "q_over_p": abs(qp),
            "phase": float(np.angle(qp)),
        }


def time_functions(
    tau: Any, z_cp: complex, delta_z: complex, xp: Any = np
) -> tuple[Any, Any]:
    r""":math:`\cosh(w\tau/2)` and :math:`\sinh(w\tau/2)/w`, with :math:`w^2 = z_{CP}^2 - \Delta z^2`."""
    w = xp.sqrt(xp.asarray(z_cp**2 - delta_z**2) + 0j)
    u = w * xp.asarray(tau) / 2
    u2 = u * u
    small = xp.abs(u) < _SERIES_LIMIT
    safe_u = xp.where(small, 1.0, u)
    sinhc = xp.where(small, 1 + u2 / 6 + u2 * u2 / 120, xp.sinh(safe_u) / safe_u)
    return xp.cosh(u), sinhc * xp.asarray(tau) / 2


def evolved_amplitude(
    tau: Any,
    amplitude: Any,
    amplitude_bar: Any,
    parameters: MixingParameters,
    tag: int,
    xp: Any = np,
) -> Any:
    r""":math:`R_{D^0}` (``tag = +1``) or :math:`R_{\bar{D}^0}` (``tag = -1``)."""
    z, dz = parameters.z_cp, parameters.delta_z
    g_cosh, g_sinh = time_functions(tau, z, dz, xp)
    if tag == D0:
        return g_cosh * amplitude + g_sinh * (z + dz) * amplitude_bar
    if tag == D0BAR:
        return g_cosh * amplitude_bar + g_sinh * (z - dz) * amplitude
    msg = f"tag must be {D0} or {D0BAR}, got {tag}"
    raise ValueError(msg)


def rate(
    tau: Any,
    amplitude: Any,
    amplitude_bar: Any,
    parameters: MixingParameters,
    tag: int,
    *,
    relative_to_decay: bool = False,
    xp: Any = np,
) -> Any:
    r""":math:`e^{-\tau}|R_\mathrm{tag}|^2`, or :math:`|R_\mathrm{tag}|^2` if ``relative_to_decay``."""
    r = evolved_amplitude(tau, amplitude, amplitude_bar, parameters, tag, xp)
    value = xp.real(r * xp.conj(r))
    return value if relative_to_decay else value * xp.exp(-xp.asarray(tau))
