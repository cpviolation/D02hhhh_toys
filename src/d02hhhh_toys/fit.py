r"""Time-dependent amplitude likelihood with an external normalisation sample.

.. math::

    -\ln L = -\sum_\mathrm{tags}\left[\sum_i \ln|R_\mathrm{tag}(\tau_i, p_i)|^2
        - N_\mathrm{tag} \ln \frac{\sum_j w_j |R_\mathrm{tag}(\tau_j, p_j)|^2}{\sum_j w_j}\right]

The rates are relative to :math:`e^{-\tau}` (:func:`.timeevolution.rate` with
``relative_to_decay=True``). The factor :math:`e^{-\tau_i}` of the data does not depend
on the parameters. The normalisation events must be distributed as
:math:`e^{-\tau} \times g(p) \times \varepsilon_\mathrm{model}(\tau, p)`, with weights
:math:`w_j \propto 1/g(p_j)` for any known phase-space density :math:`g`. The
normalisation is estimated up to a constant factor, which does not change the fit.

- **Default:** :math:`g = (|A_f|^2 + |\bar{A}_f|^2)/2`, the proposal of
  :meth:`.generate.MixingGenerator.proposal`, with :math:`w = 2/(|A_f|^2 +
  |\bar{A}_f|^2)` (``RateSample.from_toy(..., proposal_weights=True)``). It makes
  :math:`w|R|^2` nearly constant, so a much smaller sample gives the same precision.
- Cross-check: :math:`g = 1` (flat phase space, :math:`w = 1`).

Each tag has its own normalisation (the same sample may serve both). The mixing
parameters are fitted in units of :data:`SCALE` = 1e-3.

- :class:`MixingNLL`: couplings fixed (the default), so only :math:`A_f` and
  :math:`\bar{A}_f` are needed per event, precomputed once per toy.
- :class:`CouplingMixingNLL`: some couplings :math:`c_k` free as well, from the
  precomputed component amplitudes :math:`A_k(p)` and :math:`A_k(\bar{p})`; each call
  costs :math:`N K` operations. At least one coupling must stay fixed, to set the scale
  and phase of the amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import iminuit
import numpy as np

from d02hhhh_toys.timeevolution import MixingParameters, rate

if TYPE_CHECKING:
    from collections.abc import Mapping

    from d02hhhh_toys.generate import ToySample

SCALE = 1e-3
PARAMETERS = ("x_cp_mix", "y_cp_mix", "dx_mix", "dy_mix")
#: value of -ln L where it is not finite
NON_FINITE_PENALTY = 1e30


@dataclass
class RateSample:
    r"""What the likelihood needs per event: :math:`\tau`, :math:`A_f`, :math:`\bar{A}_f`, an optional weight, and optionally :math:`A_k(p)`, :math:`A_k(\bar{p})` for free couplings."""

    tau: np.ndarray
    amplitude: np.ndarray
    amplitude_bar: np.ndarray
    weight: np.ndarray | None = None
    components: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.tau)

    @classmethod
    def from_toy(cls, toy: ToySample, *, proposal_weights: bool = False) -> RateSample:
        r"""From a toy sample; with ``proposal_weights``, :math:`w = 2/(|A_f|^2 + |\bar{A}_f|^2)`."""
        weight = None
        if proposal_weights:
            weight = 2 / (np.abs(toy.amplitude) ** 2 + np.abs(toy.amplitude_bar) ** 2)
        return cls(toy.tau, toy.amplitude, toy.amplitude_bar, weight, toy.components)


def _parameters(values: np.ndarray) -> MixingParameters:
    return MixingParameters(*(np.asarray(values[:4], dtype=float) * SCALE))


def _check_tags(
    data: Mapping[int, RateSample], normalisation: Mapping[int, RateSample]
) -> None:
    if set(data) != set(normalisation):
        msg = f"Tags of the data {set(data)} and normalisation {set(normalisation)} differ"
        raise ValueError(msg)


def _tag_nll(
    p: MixingParameters,
    tag: int,
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
    norm: tuple[np.ndarray, np.ndarray, np.ndarray],
    weight: np.ndarray | None,
) -> float:
    # far from the minimum (e.g. a Migrad line search) cosh/sinh can overflow: return a
    # large finite value instead of NaN, which Migrad handles as a bad step
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        r_data = rate(*data, p, tag, relative_to_decay=True)
        r_norm = rate(*norm, p, tag, relative_to_decay=True)
        mean = np.average(r_norm, weights=weight)
        value = -(np.sum(np.log(r_data)) - len(data[0]) * np.log(mean))
    return float(value) if np.isfinite(value) else NON_FINITE_PENALTY


class MixingNLL:
    """:math:`-\\ln L` with fixed couplings, as a function of the mixing parameters in units of 1e-3.

    Args:
        data: Data sample of each tag (``+1``, ``-1``).

        normalisation: Normalisation sample of each tag.
    """

    names: tuple[str, ...] = PARAMETERS

    def __init__(
        self, data: Mapping[int, RateSample], normalisation: Mapping[int, RateSample]
    ) -> None:
        _check_tags(data, normalisation)
        self.data = dict(data)
        self.normalisation = dict(normalisation)

    def initial(self, start: MixingParameters) -> list[float]:
        return [getattr(start, n) / SCALE for n in PARAMETERS]

    def __call__(self, values: np.ndarray) -> float:
        p = _parameters(values)
        total = 0.0
        for tag, d in self.data.items():
            n = self.normalisation[tag]
            total += _tag_nll(
                p,
                tag,
                (d.tau, d.amplitude, d.amplitude_bar),
                (n.tau, n.amplitude, n.amplitude_bar),
                n.weight,
            )
        return float(total)


class CouplingMixingNLL:
    """:math:`-\\ln L` with the mixing parameters (units of 1e-3) and the real and imaginary parts of the ``free`` couplings.

    Args:
        data: Data sample of each tag, with ``components``.

        normalisation: Normalisation sample of each tag, with ``components``.

        couplings: Values of all couplings (the fixed ones stay at these values).

        free: Indices of the free couplings.

        labels: Names of the couplings, for the parameter names (default ``c<k>``).
    """

    def __init__(
        self,
        data: Mapping[int, RateSample],
        normalisation: Mapping[int, RateSample],
        couplings: np.ndarray,
        free: list[int],
        labels: list[str] | None = None,
    ) -> None:
        _check_tags(data, normalisation)
        for sample in [*data.values(), *normalisation.values()]:
            if sample.components is None:
                msg = "Free couplings need the component amplitudes (keep_components=True)"
                raise ValueError(msg)
        self.couplings = np.asarray(couplings, dtype=complex).copy()
        self.free = list(free)
        fixed = np.setdiff1d(np.flatnonzero(self.couplings != 0), self.free)
        if len(fixed) == 0:
            msg = "At least one non-zero coupling must stay fixed (scale and phase)"
            raise ValueError(msg)
        labels = labels or [f"c{k}" for k in range(len(self.couplings))]
        self.names = (
            *PARAMETERS,
            *(f"{labels[k]}_{part}" for k in self.free for part in ("re", "im")),
        )
        self.data = dict(data)
        self.normalisation = dict(normalisation)

    def initial(self, start: MixingParameters) -> list[float]:
        values = [getattr(start, n) / SCALE for n in PARAMETERS]
        for k in self.free:
            values += [self.couplings[k].real, self.couplings[k].imag]
        return values

    def coupling_values(self, values: np.ndarray) -> np.ndarray:
        c = self.couplings.copy()
        pairs = np.asarray(values[4:], dtype=float).reshape(-1, 2)
        c[self.free] = pairs[:, 0] + 1j * pairs[:, 1]
        return c

    def __call__(self, values: np.ndarray) -> float:
        p = _parameters(values)
        c = self.coupling_values(values)
        total = 0.0
        for tag, d in self.data.items():
            n = self.normalisation[tag]
            total += _tag_nll(
                p,
                tag,
                (d.tau, d.components[0] @ c, d.components[1] @ c),
                (n.tau, n.components[0] @ c, n.components[1] @ c),
                n.weight,
            )
        return float(total)


@dataclass
class FitResult:
    """Fitted values and uncertainties (mixing parameters in units of 1e-3), and the covariance."""

    values: dict[str, float]
    errors: dict[str, float]
    covariance: np.ndarray
    fval: float
    valid: bool
    nfcn: int
    minuit: iminuit.Minuit | None = field(default=None, repr=False)


def fit(
    nll: MixingNLL | CouplingMixingNLL,
    start: MixingParameters | None = None,
    *,
    fixed: Mapping[str, float] | None = None,
) -> FitResult:
    """Minimise with Migrad and compute the Hesse errors.

    Args:
        nll: The negative log-likelihood.

        start: Starting values of the mixing parameters (default: no mixing); free
            couplings start at their values in ``nll``.

        fixed: Parameters to fix, with their values (mixing parameters in units of 1e-3).
    """
    start = start or MixingParameters()
    minuit = iminuit.Minuit(nll, nll.initial(start), name=nll.names)
    minuit.errordef = iminuit.Minuit.LIKELIHOOD
    minuit.strategy = 1
    for name, value in (fixed or {}).items():
        minuit.values[name] = value
        minuit.fixed[name] = True
    minuit.migrad()
    minuit.hesse()
    return FitResult(
        values=dict(zip(nll.names, minuit.values, strict=True)),
        errors=dict(zip(nll.names, minuit.errors, strict=True)),
        covariance=np.array(minuit.covariance),
        fval=float(minuit.fval),
        valid=bool(minuit.valid),
        nfcn=int(minuit.nfcn),
        minuit=minuit,
    )
