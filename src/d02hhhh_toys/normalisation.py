r"""Normalisation samples for the likelihood of :mod:`.fit`.

A normalisation sample is distributed as :math:`g(p)\, e^{-\tau}\, \varepsilon(\tau, p)`
on :math:`0 \le \tau < \tau_\max`, where :math:`\varepsilon` is the analyst's model of
the acceptance, applied by accept-reject. Each event carries a weight
:math:`w \propto 1/g(p)`, so that :math:`\sum_j w_j |R(\tau_j, p_j)|^2 / \sum_j w_j`
estimates the normalisation integral up to a constant factor. Three densities are
available:

``"peaked"`` (default)
    The peaked phase space of ``aman_examples`` (:math:`\phi`, :math:`\rho`,
    :math:`\omega`), :math:`g = r(p)`, with :math:`w = 1/r`. The acceptance is applied
    before any amplitude is evaluated.

``"flat"``
    Flat phase space, :math:`w = 1`, as in the original plan.

``"proposal"``
    The generation proposal :math:`\tfrac{1}{2}(|A_f|^2 + |\bar{A}_f|^2)`, with
    :math:`w = 2/(|A_f|^2 + |\bar{A}_f|^2)`. Its weights vary least, but each event costs
    an accept-reject on :math:`|A_f|^2/r`.

On the full model, the variance per event of the mixing-dependent terms of the
normalisation is about 0.5–0.7 (peaked), 0.9–2.3 (flat) and 0.2–0.3 (proposal), while
the proposal needs about 40 amplitude evaluations per event against 1 for the others
(``docs/notes/normalisation.md``).

The same sample serves both tags: neither the proposal nor the emulated acceptance
depends on the flavour, and the likelihood normalises each tag separately.

:func:`normalisation_moments` accumulates the decay-time moments of
:mod:`.moments` bunch by bunch, so that a large sample is never held in memory. It
sees the same events as :func:`normalisation_sample` with the same arguments.
:func:`shared_normalisation_moments` does the same for several acceptance models at
once: the candidates are generated once, and the amplitudes are evaluated once for the
union of the accepted events.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
from aman_examples import d02kkpipi
from tensorwaves.data.rng import NumpyUniformRNG

from d02hhhh_toys.fit import RateSample
from d02hhhh_toys.generate import MixingGenerator, _seeds
from d02hhhh_toys.model import from_sample
from d02hhhh_toys.moments import ORDER, NormMoments

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator, Mapping

    from d02hhhh_toys.acceptance import LHCbAcceptance
    from d02hhhh_toys.model import AmplitudeModel

METHODS = ("peaked", "flat", "proposal")


def _decay_times(size: int, tau_max: float, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(size=size)
    return -np.log1p(-u * -np.expm1(-tau_max))


def normalisation_sample(
    model: AmplitudeModel,
    size: int,
    seed: int | np.random.SeedSequence,
    *,
    acceptance: LHCbAcceptance | None = None,
    method: str = "peaked",
    generator: MixingGenerator | None = None,
    tau_max: float = 20.0,
    bunch_size: int = 200_000,
    keep_components: bool = False,
) -> RateSample:
    """``size`` weighted events for the normalisation, after the ``acceptance`` (if any).

    Args:
        model: The amplitude adapter; its couplings give :math:`A_f`, :math:`\\bar{A}_f`.

        size: Number of events after the acceptance.

        seed: Seed of the sample.

        acceptance: The analyst's model of the acceptance (None: no acceptance).

        method: ``"peaked"``, ``"flat"`` or ``"proposal"`` (module docstring).

        generator: For ``"proposal"``, the generator whose proposal is used (its
            ``tau_max`` is used instead of ``tau_max``).

        tau_max: Upper limit of the decay times; must match the generation.

        bunch_size: Phase-space candidates per iteration.

        keep_components: Also store :math:`A_k(p)`, :math:`A_k(\\bar{p})`, for fits with
            free couplings.
    """
    _check_method(method, generator)
    if method == "proposal":
        toy = generator.proposal(
            size, seed, acceptance=acceptance, keep_components=keep_components
        )
        return RateSample.from_toy(toy, proposal_weights=True)
    chunks = list(
        _peaked_or_flat_chunks(
            model, size, seed, acceptance, method, tau_max, bunch_size, keep_components
        )
    )
    components = None
    if keep_components:
        components = tuple(
            np.concatenate([c.components[i] for c in chunks]) for i in (0, 1)
        )
    return RateSample(
        tau=np.concatenate([c.tau for c in chunks]),
        amplitude=np.concatenate([c.amplitude for c in chunks]),
        amplitude_bar=np.concatenate([c.amplitude_bar for c in chunks]),
        weight=np.concatenate([c.weight for c in chunks]),
        components=components,
    )


def _check_method(method: str, generator: MixingGenerator | None) -> None:
    if method not in METHODS:
        msg = f"Unknown normalisation method {method!r} (known: {METHODS})"
        raise ValueError(msg)
    if method == "proposal" and generator is None:
        msg = "The 'proposal' normalisation needs the MixingGenerator"
        raise ValueError(msg)


def _peaked_or_flat_chunks(
    model: AmplitudeModel,
    size: int,
    seed: int | np.random.SeedSequence,
    acceptance: LHCbAcceptance | None,
    method: str,
    tau_max: float,
    bunch_size: int,
    keep_components: bool,
) -> Iterator[RateSample]:
    """The accepted events of each bunch, until ``size`` events (the last bunch is cut)."""
    for bunch in _shared_chunks(
        model,
        size,
        seed,
        {None: acceptance},
        method,
        tau_max,
        bunch_size,
        keep_components,
    ):
        if None in bunch:
            yield bunch[None]


def _shared_chunks(
    model: AmplitudeModel,
    size: int,
    seed: int | np.random.SeedSequence,
    acceptances: Mapping[Hashable, LHCbAcceptance | None],
    method: str,
    tau_max: float,
    bunch_size: int,
    keep_components: bool,
) -> Iterator[dict[Hashable, RateSample]]:
    """Accepted events of each bunch for several acceptance models, until each has ``size`` events.

    All models see the same candidates, and the same random numbers in each bunch
    (common random numbers). The amplitudes are evaluated once per bunch, for the
    union of the events accepted by any model.
    """
    if method == "peaked":
        domain = d02kkpipi.peaked_phase_space_generator(model.model)
    else:
        domain = d02kkpipi.phase_space_generator(model.model)
    seed_domain, seed_rng = _seeds(seed, 2)
    rng_domain, rng = NumpyUniformRNG(seed=seed_domain), np.random.default_rng(seed_rng)
    counts = dict.fromkeys(acceptances, 0)
    while min(counts.values()) < size:
        sample = domain.generate(bunch_size, rng_domain)
        p = from_sample(sample)
        tau = _decay_times(bunch_size, tau_max, rng)
        seed_acceptance = rng.integers(2**63)  # the same for every model
        indices = {}
        for key, acceptance in acceptances.items():
            if counts[key] >= size:
                continue
            keep = (
                np.ones(bunch_size, dtype=bool)
                if acceptance is None
                else acceptance.accept(p, tau, np.random.default_rng(seed_acceptance))
            )
            index = np.flatnonzero(keep)[: size - counts[key]]
            counts[key] += len(index)
            if len(index):
                indices[key] = index
        if not indices:
            continue
        union = functools.reduce(np.union1d, indices.values())
        w = (
            1 / domain.density_ratio(sample)
            if method == "peaked"
            else np.ones(bunch_size)
        )
        components, conjugates = model.component_pair(p[union])
        amplitude = components @ model.couplings
        amplitude_bar = conjugates @ model.couplings
        bunch = {}
        for key, index in indices.items():
            at = np.searchsorted(union, index)
            bunch[key] = RateSample(
                tau=tau[index],
                amplitude=amplitude[at],
                amplitude_bar=amplitude_bar[at],
                weight=w[index],
                components=(components[at], conjugates[at])
                if keep_components
                else None,
            )
        yield bunch


def shared_normalisation_moments(
    model: AmplitudeModel,
    size: int,
    seed: int | np.random.SeedSequence,
    acceptances: Mapping[Hashable, LHCbAcceptance | None],
    *,
    method: str = "peaked",
    tau_max: float = 20.0,
    bunch_size: int = 200_000,
    order: int = ORDER,
    components: bool = False,
) -> dict[Hashable, NormMoments]:
    """:func:`normalisation_moments` for several acceptance models at once, sharing the candidates and the amplitudes.

    Each model gets exactly the moments of ``normalisation_moments(...,
    acceptance=acceptances[key])`` with the same arguments.
    """
    if method not in {"peaked", "flat"}:
        msg = f"Shared normalisation samples need 'peaked' or 'flat', not {method!r}"
        raise ValueError(msg)
    totals: dict[Hashable, NormMoments] = {}
    for bunch in _shared_chunks(
        model, size, seed, acceptances, method, tau_max, bunch_size, components
    ):
        for key, sample in bunch.items():
            moments = NormMoments.from_sample(
                sample, order=order, components=components
            )
            totals[key] = totals[key] + moments if key in totals else moments
    return totals


def normalisation_moments(
    model: AmplitudeModel,
    size: int,
    seed: int | np.random.SeedSequence,
    *,
    acceptance: LHCbAcceptance | None = None,
    method: str = "peaked",
    generator: MixingGenerator | None = None,
    tau_max: float = 20.0,
    bunch_size: int = 200_000,
    order: int = ORDER,
    components: bool = False,
) -> NormMoments:
    """The moments (:class:`.moments.NormMoments`) of :func:`normalisation_sample`, accumulated bunch by bunch.

    With ``components``, the moments of the component amplitudes, for free couplings.
    """
    _check_method(method, generator)
    if method == "proposal":
        sample = normalisation_sample(
            model,
            size,
            seed,
            acceptance=acceptance,
            method=method,
            generator=generator,
            keep_components=components,
        )
        return NormMoments.from_sample(sample, order=order, components=components)
    return shared_normalisation_moments(
        model,
        size,
        seed,
        {None: acceptance},
        method=method,
        tau_max=tau_max,
        bunch_size=bunch_size,
        order=order,
        components=components,
    )[None]
