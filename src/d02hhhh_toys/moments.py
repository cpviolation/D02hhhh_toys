r"""The normalisation of the likelihood as decay-time moments of a normalisation sample.

With :math:`c(\tau) = \cosh(w\tau/2)` and :math:`s(\tau) = \sinh(w\tau/2)/w`
(:func:`.timeevolution.time_functions`), the rate of a :math:`D^0` relative to
:math:`e^{-\tau}` is

.. math::

    |R|^2 = |c|^2 |A_f|^2 + |s|^2 |\zeta|^2 |\bar{A}_f|^2
        + 2\,\mathrm{Re}\left(\bar{c} s\, \zeta\, A_f^* \bar{A}_f\right),
    \qquad \zeta = z_{CP} + \Delta z,

and the same with :math:`A_f \leftrightarrow \bar{A}_f` and :math:`\zeta = z_{CP} -
\Delta z` for a :math:`\bar{D}^0`. The functions :math:`|c|^2`, :math:`|s|^2` and
:math:`\bar{c}s` are power series in :math:`\tau` whose coefficients depend only on the
mixing parameters. The weighted sum over a normalisation sample is therefore a
combination of the **moments**

.. math::

    M^{AA}_n = \sum_j w_j \tau_j^n A_{f,j}^* A_{f,j}, \quad
    M^{\bar{A}\bar{A}}_n = \sum_j w_j \tau_j^n \bar{A}_{f,j}^* \bar{A}_{f,j}, \quad
    M^{A\bar{A}}_n = \sum_j w_j \tau_j^n A_{f,j}^* \bar{A}_{f,j},

for :math:`n \le` ``order``. With free couplings the moments are :math:`K \times K`
matrices of the component amplitudes, and :math:`A_f^* A_f \to c^\dagger M c`.

- **Exact up to the truncation of the series.** The terms fall as
  :math:`(|w|\tau_\max/2)^n/n!`. With ``order`` = 20 and :math:`\tau_\max = 20` the
  truncation is below 1e-15 for :math:`|w| \lesssim 0.1`, i.e. for realistic and inflated
  mixing values. :meth:`NormMoments.truncation` bounds it for given parameters; the
  likelihood treats a point where it exceeds :data:`TRUNCATION_TOLERANCE` as a bad step.
- **Cost:** a fit no longer depends on the size of the normalisation sample, and the
  sample can be accumulated in chunks without being stored.
- **Acceptance modes** of :mod:`.study`: replacing the decay times of a sample by
  independent ones (``factorised``, ``ignore_time``, ``ignore_ps``) multiplies the
  zeroth moments by the moments :math:`\langle\tau^n\rangle` of the new decay times
  (:meth:`NormMoments.with_decay_time_moments`). This is the limit of infinitely many
  permutations, without their random noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from d02hhhh_toys.timeevolution import D0, D0BAR, MixingParameters

if TYPE_CHECKING:
    from d02hhhh_toys.fit import RateSample

ORDER = 20
#: maximum relative truncation error of the series accepted by the likelihood
TRUNCATION_TOLERANCE = 1e-10


def _series(parameters: MixingParameters, order: int) -> tuple[np.ndarray, ...]:
    r"""Coefficients of :math:`\tau^n` in :math:`|c|^2`, :math:`|s|^2`, :math:`\bar{c}s` (``n <= order``)."""
    u2 = (parameters.z_cp**2 - parameters.delta_z**2) / 4  # (w/2)^2
    c = np.zeros(order + 1, dtype=complex)
    s = np.zeros(order + 1, dtype=complex)
    for k in range(order // 2 + 1):
        c[2 * k] = u2**k / math.factorial(2 * k)
        if 2 * k + 1 <= order:
            s[2 * k + 1] = u2**k / (2 * math.factorial(2 * k + 1))
    cc = np.convolve(c, c.conj())[: order + 1].real
    ss = np.convolve(s, s.conj())[: order + 1].real
    cs = np.convolve(c.conj(), s)[: order + 1]
    return cc, ss, cs


def decay_time_moments(tau_max: float, order: int) -> np.ndarray:
    r""":math:`\langle\tau^n\rangle` of :math:`e^{-\tau}` on :math:`[0, \tau_\max)`, for ``n <= order``."""
    # I_n = n I_{n-1} - T^n e^{-T}: stable for n < T, with a series otherwise
    t = float(tau_max)
    integrals = np.empty(order + 1)
    integrals[0] = -math.expm1(-t)
    for n in range(1, order + 1):
        integrals[n] = n * integrals[n - 1] - t**n * math.exp(-t)
        if n >= t:  # lower incomplete gamma from its series, always stable
            terms, term, k = 0.0, t ** (n + 1) / (n + 1), 0
            while term > 1e-17 * max(terms, 1e-300):
                terms += term
                k += 1
                term *= t / (n + 1 + k)
            integrals[n] = math.exp(-t) * terms
    return integrals / integrals[0]


@dataclass
class NormMoments:
    """Decay-time moments of a normalisation sample (module docstring).

    ``aa``, ``bb``, ``ab``: arrays ``(order + 1, K, K)`` of :math:`\\sum_j w_j
    \\tau_j^n\\, X_{k,j}^* Y_{l,j}` with :math:`(X, Y) = (A, A), (\\bar{A}, \\bar{A}), (A,
    \\bar{A})`; :math:`K = 1` for the fixed-coupling amplitudes :math:`A_f`,
    :math:`\\bar{A}_f`. ``tau_powers``: the unweighted :math:`\\sum_j \\tau_j^n`.
    """

    aa: np.ndarray
    bb: np.ndarray
    ab: np.ndarray
    weight_sum: float
    count: int
    tau_powers: np.ndarray
    tau_max: float

    @property
    def order(self) -> int:
        return len(self.tau_powers) - 1

    @property
    def n_components(self) -> int:
        return self.aa.shape[1]

    @classmethod
    def from_sample(
        cls,
        sample: RateSample,
        *,
        order: int = ORDER,
        components: bool = False,
    ) -> NormMoments:
        """The moments of a :class:`.fit.RateSample`; with ``components``, of its :math:`A_k`."""
        tau = np.asarray(sample.tau, dtype=float)
        weight = np.ones(len(tau)) if sample.weight is None else sample.weight
        if components:
            if sample.components is None:
                msg = "The sample has no component amplitudes (keep_components=True)"
                raise ValueError(msg)
            a, abar = sample.components
        else:
            a, abar = sample.amplitude[:, None], sample.amplitude_bar[:, None]
        powers = tau[:, None] ** np.arange(order + 1)  # (N, n)
        wp = powers * weight[:, None]
        k = a.shape[1]
        aa = np.empty((order + 1, k, k), dtype=complex)
        bb = np.empty_like(aa)
        ab = np.empty_like(aa)
        for n in range(order + 1):
            wa = a.conj() * wp[:, n, None]
            aa[n] = wa.T @ a
            ab[n] = wa.T @ abar
            bb[n] = (abar.conj() * wp[:, n, None]).T @ abar
        return cls(
            aa=aa,
            bb=bb,
            ab=ab,
            weight_sum=float(np.sum(weight)),
            count=len(tau),
            tau_powers=powers.sum(axis=0),
            tau_max=float(tau.max(initial=0.0)),
        )

    def __add__(self, other: NormMoments) -> NormMoments:
        if self.aa.shape != other.aa.shape:
            msg = f"Moments of different shapes: {self.aa.shape}, {other.aa.shape}"
            raise ValueError(msg)
        return NormMoments(
            aa=self.aa + other.aa,
            bb=self.bb + other.bb,
            ab=self.ab + other.ab,
            weight_sum=self.weight_sum + other.weight_sum,
            count=self.count + other.count,
            tau_powers=self.tau_powers + other.tau_powers,
            tau_max=max(self.tau_max, other.tau_max),
        )

    @property
    def mean_tau_powers(self) -> np.ndarray:
        r"""Unweighted :math:`\langle\tau^n\rangle` of the sample's decay times."""
        return self.tau_powers / self.count

    def with_decay_time_moments(
        self, moments: np.ndarray, tau_max: float
    ) -> NormMoments:
        r"""The moments if each event's :math:`\tau` were replaced by an independent one with moments :math:`\langle\tau^n\rangle` = ``moments``."""
        moments = np.asarray(moments, dtype=float)[: self.order + 1]
        return replace(
            self,
            aa=self.aa[:1] * moments[:, None, None],
            bb=self.bb[:1] * moments[:, None, None],
            ab=self.ab[:1] * moments[:, None, None],
            tau_powers=self.count * moments,
            tau_max=float(tau_max),
        )

    def integrals(
        self, couplings: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r""":math:`c^\dagger M_n c` for the three moments (``couplings`` = None for :math:`K = 1`)."""
        if couplings is None:
            if self.n_components != 1:
                msg = "Moments of the component amplitudes need the couplings"
                raise ValueError(msg)
            return self.aa[:, 0, 0].real, self.bb[:, 0, 0].real, self.ab[:, 0, 0]
        c = np.asarray(couplings, dtype=complex)

        def form(m: np.ndarray) -> np.ndarray:
            return np.einsum("k,nkl,l->n", c.conj(), m, c)

        return form(self.aa).real, form(self.bb).real, form(self.ab)

    def mean(
        self,
        parameters: MixingParameters,
        tag: int,
        couplings: np.ndarray | None = None,
    ) -> float:
        r""":math:`\sum_j w_j |R_\mathrm{tag}(\tau_j, p_j)|^2 / \sum_j w_j` (rate relative to :math:`e^{-\tau}`)."""
        cc, ss, cs = _series(parameters, self.order)
        aa, bb, ab = self.integrals(couplings)
        if tag == D0:
            zeta, primary, other, cross = (
                parameters.z_cp + parameters.delta_z,
                aa,
                bb,
                ab,
            )
        elif tag == D0BAR:
            zeta, primary, other, cross = (
                parameters.z_cp - parameters.delta_z,
                bb,
                aa,
                ab.conj(),
            )
        else:
            msg = f"tag must be {D0} or {D0BAR}, got {tag}"
            raise ValueError(msg)
        total = (
            cc @ primary
            + abs(zeta) ** 2 * (ss @ other)
            + 2 * np.real(zeta * (cs @ cross))
        )
        return float(total / self.weight_sum)

    def truncation(self, parameters: MixingParameters) -> float:
        r"""Bound on the relative error from truncating the series at ``order``, for :math:`\tau \le` ``tau_max``."""
        x = abs(parameters.z_cp**2 - parameters.delta_z**2) ** 0.5 / 2 * self.tau_max
        zeta = abs(parameters.z_cp) + abs(parameters.delta_z)
        # majorants: |c| <= cosh(x), |s| <= (tau/2) sinh(x)/x; the tail of the product
        # series starts at order + 1 (in logarithms: x is large far from the minimum)
        if x == 0:
            return 0.0
        n = self.order + 1
        log_tail = (
            n * math.log(2 * x)
            - math.lgamma(n + 1)
            + 2 * x
            + 2 * math.log1p(zeta * self.tau_max)
        )
        return math.exp(min(log_tail, 700.0))
