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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from aman_examples import d02kkpipi
from tensorwaves.data.rng import NumpyUniformRNG

from d02hhhh_toys.fit import RateSample
from d02hhhh_toys.generate import MixingGenerator, _seeds
from d02hhhh_toys.model import from_sample

if TYPE_CHECKING:
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
    if method not in METHODS:
        msg = f"Unknown normalisation method {method!r} (known: {METHODS})"
        raise ValueError(msg)
    if method == "proposal":
        if generator is None:
            msg = "The 'proposal' normalisation needs the MixingGenerator"
            raise ValueError(msg)
        toy = generator.proposal(
            size, seed, acceptance=acceptance, keep_components=keep_components
        )
        return RateSample.from_toy(toy, proposal_weights=True)
    if method == "peaked":
        domain = d02kkpipi.peaked_phase_space_generator(model.model)
    else:
        domain = d02kkpipi.phase_space_generator(model.model)
    seed_domain, seed_rng = _seeds(seed, 2)
    rng_domain, rng = NumpyUniformRNG(seed=seed_domain), np.random.default_rng(seed_rng)
    events, taus, weights = [], [], []
    count = 0
    while count < size:
        sample = domain.generate(bunch_size, rng_domain)
        p = from_sample(sample)
        tau = _decay_times(bunch_size, tau_max, rng)
        keep = (
            np.ones(bunch_size, dtype=bool)
            if acceptance is None
            else acceptance.accept(p, tau, rng)
        )
        w = (
            1 / domain.density_ratio(sample)
            if method == "peaked"
            else np.ones(bunch_size)
        )
        events.append(p[keep])
        taus.append(tau[keep])
        weights.append(w[keep])
        count += int(keep.sum())
    p = np.concatenate(events)[:size]
    components, conjugates = model.component_pair(p)
    return RateSample(
        tau=np.concatenate(taus)[:size],
        amplitude=components @ model.couplings,
        amplitude_bar=conjugates @ model.couplings,
        weight=np.concatenate(weights)[:size],
        components=(components, conjugates) if keep_components else None,
    )
