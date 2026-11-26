r"""How much a finite normalisation sample moves the fit: the factor :math:`v`.

A finite normalisation sample of :math:`n` events shifts the fitted parameters of
:math:`N` events per tag by an amount common to all toys, with

.. math::

    \sigma_\mathrm{norm}^2 = v\, \frac{N}{n}\, \sigma_\mathrm{stat}^2 .

To first order (delta method), the sample enters the fit only through the gradient of
the normalisation term,
:math:`N \sum_\mathrm{tags} \partial_\theta \ln \hat{Z}_\mathrm{tag}`. Each event
:math:`j` contributes the influence

.. math::

    U_j = \sum_\mathrm{tags} \frac{w_j\left(\partial_\theta r_j
          - r_j\, \partial_\theta \ln Z_\mathrm{tag}\right)}{Z_\mathrm{tag}},
    \qquad r = |R_\mathrm{tag}|^2,\ Z_\mathrm{tag} = \langle w r \rangle ,

which is summed over the tags because both tags share the sample. With the Fisher
information per event :math:`I = \sum_\mathrm{tags} E_\mathrm{tag}[s s^T]`, where
:math:`s = \partial_\theta \ln r - \partial_\theta \ln Z_\mathrm{tag}`, the shift is
:math:`\delta\theta = -I^{-1}\bar{U}`, with :math:`\bar{U}` the mean over the sample, and

.. math::

    v_i = \frac{(I^{-1}\, \mathrm{Cov}(U)\, I^{-1})_{ii}}{(I^{-1})_{ii}} .

Everything is computed from one weighted sample, without toys.
:func:`variance_budget` splits :math:`\mathrm{Cov}(U)` into a phase-space part and a
part from the decay times and the acceptance. It also predicts :math:`v` for
alternative designs of the sample:

- with the average over :math:`\tau` and the acceptance taken per phase-space point;
- with a phase-space density proportional to :math:`|A_f|^2 + |\bar{A}_f|^2`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from d02hhhh_toys.fit import PARAMETERS
from d02hhhh_toys.timeevolution import D0, D0BAR, MixingParameters, rate


def _rate_and_gradient(
    tau: np.ndarray,
    a: np.ndarray,
    abar: np.ndarray,
    parameters: MixingParameters,
    tag: int,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    r = rate(tau, a, abar, parameters, tag, relative_to_decay=True)
    gradient = np.empty((len(tau), len(PARAMETERS)))
    for i, name in enumerate(PARAMETERS):
        value = getattr(parameters, name)
        up = replace(parameters, **{name: value + step})
        down = replace(parameters, **{name: value - step})
        gradient[:, i] = (
            rate(tau, a, abar, up, tag, relative_to_decay=True)
            - rate(tau, a, abar, down, tag, relative_to_decay=True)
        ) / (2 * step)
    return r, gradient


def influence(
    tau: np.ndarray,
    amplitude: np.ndarray,
    amplitude_bar: np.ndarray,
    weight: np.ndarray,
    parameters: MixingParameters,
    *,
    step: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """The influence :math:`U_j` ``(n, 4)`` of each event, and the information per event :math:`I` ``(4, 4)``.

    ``weight`` may be zero (e.g. a candidate rejected by the acceptance): such an event
    counts in :math:`n` but contributes nothing.
    """
    total_u = np.zeros((len(tau), len(PARAMETERS)))
    information = np.zeros((len(PARAMETERS), len(PARAMETERS)))
    for tag in (D0, D0BAR):
        r, dr = _rate_and_gradient(tau, amplitude, amplitude_bar, parameters, tag, step)
        z = np.mean(weight * r)
        d = np.mean(weight[:, None] * dr, axis=0) / z  # gradient of ln Z
        total_u += weight[:, None] * (dr - r[:, None] * d) / z
        keep = weight > 0
        s = dr[keep] / r[keep, None] - d
        information += (weight[keep] * r[keep] * s.T) @ s / (z * len(tau))
    return total_u, information


def v_factor(covariance: np.ndarray, information: np.ndarray) -> np.ndarray:
    """:math:`v_i` for each parameter, from :math:`\\mathrm{Cov}(U)` and :math:`I`."""
    inverse = np.linalg.inv(information)
    return np.diag(inverse @ covariance @ inverse) / np.diag(inverse)


@dataclass
class VarianceBudget:
    """:math:`v` per parameter (dict of arrays, in the order of :data:`.fit.PARAMETERS`).

    - ``per_event``: the sample as it is built now, per accepted event.
    - ``phase_space`` and ``time_acceptance``: the two parts of ``per_event``. The first
      is the variance of the influence averaged over :math:`\\tau` and the acceptance at
      fixed phase-space point, the second the remainder.
    - ``averaged``: the average over :math:`\\tau` and the acceptance taken for each
      phase-space point, per phase-space point (one amplitude evaluation each), in the
      limit of infinitely many replicas. With :math:`M` replicas per point it is
      ``averaged + within / M``.
    - ``within``: the variance between the replicas of a point, per replica.
    - ``*_amplitude_density``: the same with phase-space points drawn from
      :math:`|A_f|^2 + |\\bar{A}_f|^2` instead of the current density.
    - ``efficiency``: fraction of candidates accepted; ``points_accepted``: fraction of
      phase-space points with at least one accepted replica.
    """

    v: dict[str, np.ndarray]
    efficiency: float
    points_accepted: float
    n_points: int
    n_replicas: int


def variance_budget(
    tau: np.ndarray,
    amplitude: np.ndarray,
    amplitude_bar: np.ndarray,
    weight: np.ndarray,
    accepted: np.ndarray,
    parameters: MixingParameters,
) -> VarianceBudget:
    """:math:`v` of the current and alternative normalisation samples (:class:`VarianceBudget`).

    Args:
        tau: Decay times ``(K, M)``: :math:`M` replicas for each of :math:`K`
            phase-space points.

        amplitude: :math:`A_f` of each point ``(K,)``.

        amplitude_bar: :math:`\\bar{A}_f` of each point ``(K,)``.

        weight: Weight :math:`1/g` of each point ``(K,)``.

        accepted: Acceptance decision of each replica ``(K, M)``.

        parameters: Mixing parameters at which the fit is linearised.
    """
    k, m = tau.shape
    flat_w = (np.repeat(weight, m) * accepted.ravel()).astype(float)
    u, information = influence(
        tau.ravel(),
        np.repeat(amplitude, m),
        np.repeat(amplitude_bar, m),
        flat_w,
        parameters,
    )
    u = u.reshape(k, m, -1)
    efficiency = float(accepted.mean())
    # candidates: Cov(U) per candidate; per accepted event it is efficiency times that
    mean = u.mean(axis=1)  # (K, 4): average over replicas
    within = np.einsum("kmi,kmj->kij", u - mean[:, None], u - mean[:, None]) / (m - 1)
    # Var(E[U|p]) without the noise of the finite number of replicas
    between = mean.T @ mean / k - within.mean(axis=0) / m
    total = np.einsum("kmi,kmj->ij", u, u) / (k * m)
    # density |A|^2 + |Abar|^2: relative to the current one, q = g'/g, E_g[q] = 1
    q = weight * (np.abs(amplitude) ** 2 + np.abs(amplitude_bar) ** 2)
    q = q / q.mean()
    total_q = np.einsum("kmi,kmj,k->ij", u, u, 1 / q) / (k * m)
    between_q = (mean.T / q) @ mean / k - (within / q[:, None, None]).mean(axis=0) / m
    efficiency_q = float(np.mean(q[:, None] * accepted))
    v = {
        "per_event": v_factor(efficiency * total, information),
        "phase_space": v_factor(efficiency * between, information),
        "time_acceptance": v_factor(efficiency * (total - between), information),
        "averaged": v_factor(between, information),
        "within": v_factor(within.mean(axis=0), information),
        "per_event_amplitude_density": v_factor(efficiency_q * total_q, information),
        "averaged_amplitude_density": v_factor(between_q, information),
    }
    return VarianceBudget(
        v=v,
        efficiency=efficiency,
        points_accepted=float(accepted.any(axis=1).mean()),
        n_points=k,
        n_replicas=m,
    )
