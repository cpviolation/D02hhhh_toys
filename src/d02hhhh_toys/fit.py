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

Each tag has its own normalisation sample. The couplings are fixed, so only
:math:`A_f` and :math:`\bar{A}_f` are needed per event. The mixing parameters are
fitted in units of :data:`SCALE` = 1e-3.
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


@dataclass
class RateSample:
    r"""What the likelihood needs per event: :math:`\tau`, :math:`A_f`, :math:`\bar{A}_f`, and an optional weight."""

    tau: np.ndarray
    amplitude: np.ndarray
    amplitude_bar: np.ndarray
    weight: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.tau)

    @classmethod
    def from_toy(cls, toy: ToySample, *, proposal_weights: bool = False) -> RateSample:
        r"""From a toy sample; with ``proposal_weights``, :math:`w = 2/(|A_f|^2 + |\bar{A}_f|^2)`."""
        weight = None
        if proposal_weights:
            weight = 2 / (np.abs(toy.amplitude) ** 2 + np.abs(toy.amplitude_bar) ** 2)
        return cls(toy.tau, toy.amplitude, toy.amplitude_bar, weight)


def _parameters(values: np.ndarray) -> MixingParameters:
    return MixingParameters(*(np.asarray(values, dtype=float) * SCALE))


class MixingNLL:
    """:math:`-\\ln L` of the module docstring, as a function of the parameters in units of 1e-3.

    Args:
        data: Data sample of each tag (``+1``, ``-1``).

        normalisation: Normalisation sample of each tag.
    """

    def __init__(
        self, data: Mapping[int, RateSample], normalisation: Mapping[int, RateSample]
    ) -> None:
        if set(data) != set(normalisation):
            msg = f"Tags of the data {set(data)} and normalisation {set(normalisation)} differ"
            raise ValueError(msg)
        self.data = dict(data)
        self.normalisation = dict(normalisation)

    def __call__(self, values: np.ndarray) -> float:
        p = _parameters(values)
        total = 0.0
        for tag, sample in self.data.items():
            norm = self.normalisation[tag]
            r_data = rate(
                sample.tau,
                sample.amplitude,
                sample.amplitude_bar,
                p,
                tag,
                relative_to_decay=True,
            )
            r_norm = rate(
                norm.tau,
                norm.amplitude,
                norm.amplitude_bar,
                p,
                tag,
                relative_to_decay=True,
            )
            mean = np.average(r_norm, weights=norm.weight)
            total -= np.sum(np.log(r_data)) - len(sample) * np.log(mean)
        return float(total)


@dataclass
class FitResult:
    """Fitted values and uncertainties in units of 1e-3, and the covariance."""

    values: dict[str, float]
    errors: dict[str, float]
    covariance: np.ndarray
    fval: float
    valid: bool
    nfcn: int
    minuit: iminuit.Minuit | None = field(default=None, repr=False)


def fit(
    nll: MixingNLL,
    start: MixingParameters | None = None,
    *,
    fixed: Mapping[str, float] | None = None,
) -> FitResult:
    """Minimise with Migrad and compute the Hesse errors.

    Args:
        nll: The negative log-likelihood.

        start: Starting values (default: no mixing).

        fixed: Parameters to fix, with their values in units of 1e-3.
    """
    start = start or MixingParameters()
    initial = [getattr(start, n) / SCALE for n in PARAMETERS]
    minuit = iminuit.Minuit(nll, initial, name=PARAMETERS)
    minuit.errordef = iminuit.Minuit.LIKELIHOOD
    minuit.strategy = 1
    for name, value in (fixed or {}).items():
        minuit.values[name] = value
        minuit.fixed[name] = True
    minuit.migrad()
    minuit.hesse()
    return FitResult(
        values=dict(zip(PARAMETERS, minuit.values, strict=True)),
        errors=dict(zip(PARAMETERS, minuit.errors, strict=True)),
        covariance=np.array(minuit.covariance),
        fval=float(minuit.fval),
        valid=bool(minuit.valid),
        nfcn=int(minuit.nfcn),
        minuit=minuit,
    )
