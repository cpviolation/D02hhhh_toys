r"""Toy generation of tagged :math:`D^0` and :math:`\bar{D}^0` decays with mixing.

Two steps:

1. :class:`AmplitudeSampler` draws events from :math:`|A_f(p)|^2` without mixing. The
   candidates come from ``aman_examples.d02kkpipi.peaked_phase_space_generator``, which
   is phase space times a known density ratio :math:`r(p)` peaked at the
   :math:`\phi`, :math:`\rho` and :math:`\omega`. They are accepted with probability
   :math:`|A_f|^2 / (r M)`, where the maximum :math:`M` is estimated once on a large
   sample and multiplied by a safety factor. An event above :math:`M` raises
   :class:`MaximumExceededError`; the maximum is never raised silently.

2. :class:`MixingGenerator` uses the proposal

   .. math::

       q(\tau, p) \propto \tfrac{1}{2}\left(|A_f(p)|^2 + |\bar{A}_f(p)|^2\right) e^{-\tau},
       \qquad 0 \le \tau < \tau_\mathrm{max}.

   Half of the candidates are :math:`|A_f|^2` events. The other half are the CP
   conjugates of :math:`|A_f|^2` events, which are distributed as :math:`|\bar{A}_f|^2`
   because :math:`\bar{A}_f(p) = A_f(\bar{p})` and CP conjugation preserves phase space.
   Each candidate is accepted with probability :math:`|R_\mathrm{tag}|^2 / (c\,q)`.
   Cauchy-Schwarz gives the bound

   .. math::

       \frac{|R_\mathrm{tag}|^2}{\tfrac{1}{2}(|A_f|^2 + |\bar{A}_f|^2)}
       \le 2\left(|\cosh(w\tau/2)|^2 + |\sinh(w\tau/2)/w|^2 |z_{CP} \pm \Delta z|^2\right),

   so :math:`c` is the maximum of the right-hand side on :math:`[0, \tau_\mathrm{max})`.
   It is exact, not estimated, and close to 2, which makes the efficiency about 50%.

:math:`D^0` and :math:`\bar{D}^0` samples are generated independently, each with its
own seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from aman_examples import d02kkpipi
from tensorwaves.data.rng import NumpyUniformRNG

from d02hhhh_toys.model import cp_conjugate, from_sample
from d02hhhh_toys.timeevolution import D0, D0BAR, MixingParameters, rate, time_functions

if TYPE_CHECKING:
    from d02hhhh_toys.model import AmplitudeModel


class MaximumExceededError(RuntimeError):
    """An event's weight is above the accept-reject maximum."""


def _seeds(seed: int | np.random.SeedSequence, n: int) -> list[int]:
    sequence = (
        seed
        if isinstance(seed, np.random.SeedSequence)
        else np.random.SeedSequence(seed)
    )
    return [int(s.generate_state(1)[0]) for s in sequence.spawn(n)]


@dataclass
class AmplitudeSample:
    r"""Events with :math:`A_k(p)` and :math:`A_k(\bar{p})` of each event (``(N, K)`` each)."""

    events: np.ndarray
    components: np.ndarray
    conjugates: np.ndarray

    def __len__(self) -> int:
        return len(self.events)

    def conjugated(self) -> AmplitudeSample:
        r"""The CP-conjugate events, :math:`p \to \bar{p}`, whose components are swapped."""
        return AmplitudeSample(
            cp_conjugate(self.events), self.conjugates, self.components
        )

    def select(self, mask: np.ndarray) -> AmplitudeSample:
        return AmplitudeSample(
            self.events[mask], self.components[mask], self.conjugates[mask]
        )

    @staticmethod
    def concatenate(samples: list[AmplitudeSample]) -> AmplitudeSample:
        return AmplitudeSample(
            np.concatenate([s.events for s in samples]),
            np.concatenate([s.components for s in samples]),
            np.concatenate([s.conjugates for s in samples]),
        )


class AmplitudeSampler:
    r"""Unweighted events from :math:`|A_f|^2` (no mixing), with their component amplitudes.

    Args:
        model: The amplitude adapter.

        safety: Factor applied to the estimated maximum of :math:`|A_f|^2 / r`.

        estimate_size: Number of candidates used to estimate the maximum.

        seed: Seed of the estimate.

        bunch_size: Number of candidates evaluated at a time.

        peak_fractions: Fractions of the :math:`\phi`, :math:`\rho`, :math:`\omega` peaks
            of the candidate generator.
    """

    def __init__(
        self,
        model: AmplitudeModel,
        *,
        safety: float = 1.5,
        estimate_size: int = 500_000,
        seed: int = 12345,
        bunch_size: int = 200_000,
        peak_fractions: tuple[float, float, float] = (0.2, 0.2, 0.1),
    ) -> None:
        self.model = model
        self.domain = d02kkpipi.peaked_phase_space_generator(
            model.model, peak_fractions
        )
        self.bunch_size = int(bunch_size)
        self.safety = float(safety)
        _, weights, _ = self._candidates(estimate_size, NumpyUniformRNG(seed=seed))
        self.estimated_maximum = float(weights.max())
        self.maximum = self.safety * self.estimated_maximum
        self.estimated_efficiency = float(weights.mean() / self.maximum)

    def _candidates(
        self, size: int, rng: NumpyUniformRNG
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Candidate events, their components and their weights :math:`|A_f|^2/r`."""
        sample = self.domain.generate(size, rng)
        events = from_sample(sample)
        components = self.model.component_amplitudes(events)
        amplitude = components @ self.model.couplings
        weights = np.abs(amplitude) ** 2 / self.domain.density_ratio(sample)
        return events, weights, components

    def generate(
        self, size: int, seed: int | np.random.SeedSequence
    ) -> AmplitudeSample:
        rng = NumpyUniformRNG(seed=_seeds(seed, 1)[0])
        accepted: list[AmplitudeSample] = []
        count = 0
        while count < size:
            events, weights, components = self._candidates(self.bunch_size, rng)
            if weights.max() > self.maximum:
                msg = (
                    f"|A|^2/r = {weights.max():.4g} exceeds the maximum {self.maximum:.4g}"
                    f" (estimated {self.estimated_maximum:.4g} x safety {self.safety}):"
                    " increase the safety factor or the estimate sample"
                )
                raise MaximumExceededError(msg)
            keep = rng(size=len(weights), max_value=self.maximum) < weights
            if keep.any():
                conjugates = self.model.conjugate_components(
                    events[keep], components[keep]
                )
                accepted.append(
                    AmplitudeSample(events[keep], components[keep], conjugates)
                )
                count += int(keep.sum())
        return AmplitudeSample.concatenate(accepted).select(slice(0, size))


@dataclass
class ToySample:
    r"""A tagged toy sample: events, decay times :math:`\tau = \Gamma t`, and amplitudes.

    ``amplitude`` and ``amplitude_bar`` are :math:`A_f(p)` and :math:`\bar{A}_f(p)` with
    the couplings of the generation. ``components`` (:math:`A_k(p)`, :math:`A_k(\bar{p})`)
    are kept only on request, because they take :math:`2K` complex numbers per event.
    """

    tag: int
    events: np.ndarray
    tau: np.ndarray
    amplitude: np.ndarray
    amplitude_bar: np.ndarray
    components: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.tau)


class MixingGenerator:
    r"""Tagged toys from the exact time-dependent rate, via the proposal of the module docstring.

    Args:
        sampler: The :math:`|A_f|^2` sampler.

        parameters: True mixing parameters.

        tau_max: Upper limit of the generated decay times, in units of the lifetime.
    """

    def __init__(
        self,
        sampler: AmplitudeSampler,
        parameters: MixingParameters,
        *,
        tau_max: float = 20.0,
    ) -> None:
        self.sampler = sampler
        self.parameters = parameters
        self.tau_max = float(tau_max)
        self.bound = {tag: self._bound(tag) for tag in (D0, D0BAR)}

    def _bound(self, tag: int) -> float:
        z, dz = self.parameters.z_cp, self.parameters.delta_z
        grid = np.linspace(0, self.tau_max, 2001)
        cosh, sinhc = time_functions(grid, z, dz)
        mixed = abs(z + dz) if tag == D0 else abs(z - dz)
        return float(np.max(2 * (np.abs(cosh) ** 2 + (np.abs(sinhc) * mixed) ** 2)))

    def _decay_times(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """:math:`e^{-\\tau}` truncated to :math:`[0, \\tau_\\mathrm{max})`."""
        u = rng.uniform(size=size)
        return -np.log1p(-u * -np.expm1(-self.tau_max))

    def _propose(
        self, n: int, seed: np.random.SeedSequence, rng: np.random.Generator
    ) -> tuple[AmplitudeSample, np.ndarray]:
        """``n`` events of the proposal: half |A|^2, half CP-conjugated |A|^2, with e^-tau."""
        candidates = self.sampler.generate(n, seed)
        conjugate = rng.uniform(size=n) < 0.5
        candidates = AmplitudeSample.concatenate(
            [
                candidates.select(~conjugate),
                candidates.select(conjugate).conjugated(),
            ]
        ).select(rng.permutation(n))
        return candidates, self._decay_times(n, rng)

    def _amplitudes(self, sample: AmplitudeSample) -> tuple[np.ndarray, np.ndarray]:
        couplings = self.sampler.model.couplings
        return sample.components @ couplings, sample.conjugates @ couplings

    def _toy(
        self, tag: int, sample: AmplitudeSample, tau: np.ndarray, keep_components: bool
    ) -> ToySample:
        a, a_bar = self._amplitudes(sample)
        return ToySample(
            tag=tag,
            events=sample.events,
            tau=tau,
            amplitude=a,
            amplitude_bar=a_bar,
            components=(sample.components, sample.conjugates)
            if keep_components
            else None,
        )

    def proposal(
        self,
        size: int,
        seed: int | np.random.SeedSequence,
        *,
        keep_components: bool = False,
    ) -> ToySample:
        r"""Events of the proposal :math:`q`, without the mixing accept-reject (``tag = 0``).

        With weights :math:`2/(|A_f|^2 + |\bar{A}_f|^2)` they are a normalisation sample of
        either tag (:class:`.fit.RateSample`).
        """
        seed_sampler, seed_rng = _seeds(seed, 2)
        rng = np.random.default_rng(seed_rng)
        sample, tau = self._propose(size, np.random.SeedSequence(seed_sampler), rng)
        return self._toy(0, sample, tau, keep_components)

    def generate(
        self,
        tag: int,
        size: int,
        seed: int | np.random.SeedSequence,
        *,
        keep_components: bool = False,
    ) -> ToySample:
        """``size`` events of initial flavour ``tag`` (``+1`` for D0, ``-1`` for D0bar)."""
        if tag not in (D0, D0BAR):
            msg = f"tag must be {D0} or {D0BAR}, got {tag}"
            raise ValueError(msg)
        seed_sampler, seed_rng = _seeds(seed, 2)
        sequence = np.random.SeedSequence(seed_sampler)
        rng = np.random.default_rng(seed_rng)
        bound = self.bound[tag]
        accepted: list[tuple[AmplitudeSample, np.ndarray]] = []
        count = 0
        while count < size:
            n = int(2.2 * (size - count)) + 1000  # about 2 candidates per event
            candidates, tau = self._propose(n, sequence.spawn(1)[0], rng)
            a, a_bar = self._amplitudes(candidates)
            proposal = (np.abs(a) ** 2 + np.abs(a_bar) ** 2) / 2
            ratio = (
                rate(tau, a, a_bar, self.parameters, tag, relative_to_decay=True)
                / proposal
            )
            if ratio.max() > bound * (1 + 1e-9):
                msg = f"Mixing weight {ratio.max():.6g} above its analytic bound {bound:.6g}"
                raise MaximumExceededError(msg)
            keep = rng.uniform(size=n) * bound < ratio
            accepted.append((candidates.select(keep), tau[keep]))
            count += int(keep.sum())
        sample = AmplitudeSample.concatenate([s for s, _ in accepted]).select(
            slice(0, size)
        )
        tau = np.concatenate([t for _, t in accepted])[:size]
        return self._toy(tag, sample, tau, keep_components)
