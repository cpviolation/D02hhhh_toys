r"""Adapter around the :math:`D^0 \to K^+K^-\pi^+\pi^-` model of ``aman_examples``.

The amplitude is :math:`A_f(p) = \sum_k c_k A_k(p)`, with the 55 decay chains
:math:`A_k` of :func:`aman_examples.d02kkpipi.formulate_model` (AmpGen's
``DtoKKpipi_v2.opt``) and their couplings :math:`c_k`. Without direct CP violation the
:math:`\bar{D}^0` amplitude is

.. math::

    \bar{A}_f(p) = \sum_k \bar{c}_k A_k(\bar{p}), \qquad \bar{c}_k = c_k,

where :math:`\bar{p}` is the CP conjugate of the event: :math:`K^+ \leftrightarrow K^-`,
:math:`\pi^+ \leftrightarrow \pi^-` **and** :math:`\vec{p} \to -\vec{p}`
(:func:`cp_conjugate`). The parity flip matters: the P-wave vector-vector chains contain
a Levi-Civita tensor and change sign under it.

**CP map.** For most chains, :math:`A_k(\bar{p}) = \eta_k A_{\pi(k)}(p)` exactly, with
:math:`\eta_k = \pm 1` and a partner chain :math:`\pi(k)`, possibly :math:`k` itself.
:class:`CPMap` finds these relations numerically, on a probe sample, with a strict
tolerance. A chain without an exact partner is evaluated explicitly at :math:`\bar{p}`.
In ``DtoKKpipi_v2.opt`` these chains are:

- the chains whose CP mirror is not in the model:
  :math:`K^*(1680)^0\bar{K}^*(892)^0` in the S and D waves, and
  :math:`\bar{K}_2^*(1430)^0 K^*(892)^0`;
- the :math:`a_1(1260)^\pm` and :math:`K_1(1270)^\pm` chains, because their spline
  running widths differ between the two charges in the options file: knot 26 of the
  :math:`a_1` spline is 1.2 for the :math:`a_1^+` and 1.20331 for the :math:`a_1^-`, and
  three :math:`K_1` knots differ at the 1e-6 level.

The model is used as it stands. :math:`\bar{A}_f` is always :math:`\sum_k c_k
A_k(\bar{p})`, so the :math:`D^0` and :math:`\bar{D}^0` amplitudes are exact CP images
of each other, whatever the asymmetries between mirror chains within :math:`A_f`.

Events are arrays of shape ``(N, 4, 4)``: particle :math:`(K^+, K^-, \pi^+, \pi^-)` ×
component :math:`(x, y, z, E)`, in GeV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp
from aman_examples import d02kkpipi
from tensorwaves.data.rng import NumpyUniformRNG
from tensorwaves.function.sympy import create_function

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from aman_examples.ampgen.model import AmpGenModel

FINAL_STATE = ("K+", "K-", "pi+", "pi-")
#: Particle permutation of CP conjugation: (K+, K-, pi+, pi-) -> (K-, K+, pi-, pi+)
CP_PERMUTATION = (1, 0, 3, 2)
COMPONENTS = ("x", "y", "z", "E")


def cp_conjugate(events: np.ndarray) -> np.ndarray:
    r"""CP conjugate of events ``(N, 4, 4)``: swap the charges and flip :math:`\vec{p}`."""
    events = np.asarray(events, dtype=float)
    conjugate = events[:, CP_PERMUTATION, :].copy()
    conjugate[..., :3] *= -1
    return conjugate


def to_sample(events: np.ndarray) -> dict[str, np.ndarray]:
    """TensorWaves data sample (``p<i>_x``, ...) from events ``(N, 4, 4)``."""
    events = np.asarray(events, dtype=float)
    return {
        f"p{i}_{c}": events[:, i, j]
        for i in range(len(FINAL_STATE))
        for j, c in enumerate(COMPONENTS)
    }


def from_sample(sample: Mapping[str, np.ndarray]) -> np.ndarray:
    """Events ``(N, 4, 4)`` from a TensorWaves data sample."""
    return np.stack(
        [
            np.column_stack([np.asarray(sample[f"p{i}_{c}"]) for c in COMPONENTS])
            for i in range(len(FINAL_STATE))
        ],
        axis=1,
    )


@dataclass(frozen=True)
class CPMap:
    r""":math:`A_k(\bar{p}) = \eta_k A_{\pi(k)}(p)`, where ``partner[k]`` is :math:`\pi(k)`, or -1 when chain :math:`k` has no exact partner."""

    partner: np.ndarray
    sign: np.ndarray

    @property
    def paired(self) -> np.ndarray:
        return self.partner >= 0

    @property
    def unpaired(self) -> np.ndarray:
        return np.flatnonzero(~self.paired)

    @classmethod
    def find(
        cls, amplitudes: np.ndarray, conjugates: np.ndarray, tolerance: float = 1e-9
    ) -> CPMap:
        r"""Find the map from :math:`A_k(p)` and :math:`A_k(\bar{p})`, both ``(n, K)``, on a probe sample.

        Chain :math:`k` is paired with :math:`m` if :math:`A_k(\bar{p}) = \eta A_m(p)`
        with :math:`\eta = \pm 1` on every event, to a relative ``tolerance`` of the rms.
        """
        n_chains = amplitudes.shape[1]
        rms = np.sqrt(np.mean(np.abs(amplitudes) ** 2, axis=0))
        partner = np.full(n_chains, -1)
        sign = np.zeros(n_chains)
        for k in range(n_chains):
            for m in range(n_chains):
                for eta in (1.0, -1.0):
                    residual = conjugates[:, k] - eta * amplitudes[:, m]
                    if np.max(np.abs(residual)) <= tolerance * max(rms[k], rms[m]):
                        partner[k], sign[k] = m, eta
                        break
                if partner[k] >= 0:
                    break
        return cls(partner, sign)


class AmplitudeModel:
    r"""Component amplitudes :math:`A_k(p)`, :math:`A_f` and :math:`\bar{A}_f` of the AmpGen :math:`K^+K^-\pi^+\pi^-` model.

    Args:
        model: The AmpGen model (default: :func:`aman_examples.d02kkpipi.formulate_model`).

        chains: Names of the chains to keep (default: all). Fewer chains are faster to
            build, which helps in tests.

        couplings: Couplings :math:`c_k` by chain name (default: those of the options
            file). Chains that are left out keep their default coupling.

        backend: ``"numpy"`` (default, no compilation) or ``"jax"`` (GPU). With JAX,
            events are evaluated in padded chunks of ``chunk_size``, so that every call
            reuses the same compiled function.

        cp_tolerance: Tolerance of :meth:`CPMap.find`.

        probe_size: Number of phase-space events (fixed seed) used to find the CP map.
    """

    def __init__(
        self,
        model: AmpGenModel | None = None,
        chains: Sequence[str] | None = None,
        couplings: Mapping[str, complex] | None = None,
        *,
        backend: str = "numpy",
        chunk_size: int = 65_536,
        cp_tolerance: float = 1e-9,
        probe_size: int = 512,
    ) -> None:
        self.model = model or d02kkpipi.formulate_model()
        if tuple(self.model.final_state) != FINAL_STATE:
            msg = f"Expected final state {FINAL_STATE}, got {self.model.final_state}"
            raise ValueError(msg)
        self.names = list(self.model.chains if chains is None else chains)
        unknown = set(self.names) - set(self.model.chains)
        if unknown:
            msg = f"Unknown chains: {sorted(unknown)}"
            raise KeyError(msg)
        defaults = self.model.parameter_defaults
        self.default_couplings = np.array(
            [
                complex(sp.N(self.model.couplings[n].xreplace(defaults)))
                for n in self.names
            ]
        )
        self.couplings = self.default_couplings.copy()
        if couplings:
            self.set_couplings(couplings)
        self.backend = backend
        self.chunk_size = int(chunk_size)
        if backend == "jax":
            import jax

            jax.config.update("jax_enable_x64", True)
        self._functions: list[Callable] = [
            create_function(
                self.model.chains[n].doit().xreplace(defaults), backend=backend
            )
            for n in self.names
        ]
        probe = from_sample(self.phase_space(probe_size, seed=0))
        self.cp_map = CPMap.find(
            self._evaluate(probe), self._evaluate(cp_conjugate(probe)), cp_tolerance
        )

    # --- phase space
    def phase_space(self, size: int, seed: int) -> dict[str, np.ndarray]:
        """Flat four-body phase space (``aman_examples``), as a TensorWaves sample."""
        generator = d02kkpipi.phase_space_generator(self.model)
        return generator.generate(size, NumpyUniformRNG(seed=seed))

    # --- couplings
    def set_couplings(self, couplings: Mapping[str, complex]) -> None:
        index = {n: k for k, n in enumerate(self.names)}
        for name, value in couplings.items():
            self.couplings[index[name]] = complex(value)

    # --- amplitudes
    def _evaluate(
        self, events: np.ndarray, columns: Sequence[int] | None = None
    ) -> np.ndarray:
        """:math:`A_k(p)` for the chains ``columns`` (default: all), shape ``(N, len(columns))``."""
        # the lambdified complex square roots evaluate both branches of a select
        with np.errstate(invalid="ignore", divide="ignore"):
            out = self._evaluate_unchecked(events, columns)
        if not np.all(np.isfinite(out)):
            msg = "Non-finite component amplitudes"
            raise FloatingPointError(msg)
        return out

    def _evaluate_unchecked(
        self, events: np.ndarray, columns: Sequence[int] | None
    ) -> np.ndarray:
        columns = range(len(self.names)) if columns is None else columns
        events = np.asarray(events, dtype=float)
        size = len(events)
        out = np.empty((size, len(columns)), dtype=complex)
        if size == 0 or len(columns) == 0:
            return out
        if self.backend != "jax":
            sample = to_sample(events)
            for j, k in enumerate(columns):
                out[:, j] = np.broadcast_to(self._functions[k](sample), size)
            return out
        for start in range(0, size, self.chunk_size):
            chunk = events[start : start + self.chunk_size]
            n = len(chunk)
            padded = np.concatenate(
                [chunk, np.broadcast_to(events[:1], (self.chunk_size - n, 4, 4))]
            )
            sample = to_sample(padded)
            for j, k in enumerate(columns):
                values = np.asarray(self._functions[k](sample))
                out[start : start + n, j] = np.broadcast_to(values, self.chunk_size)[:n]
        return out

    def component_amplitudes(self, events: np.ndarray) -> np.ndarray:
        r""":math:`A_k(p)`, shape ``(N, K)``."""
        return self._evaluate(events)

    def conjugate_components(
        self, events: np.ndarray, components: np.ndarray | None = None
    ) -> np.ndarray:
        r""":math:`A_k(\bar{p})`, shape ``(N, K)``, from ``components`` :math:`= A_k(p)` with the CP map, and explicit evaluation for the unpaired chains."""
        if components is None:
            components = self.component_amplitudes(events)
        cp = self.cp_map
        out = np.empty_like(components)
        paired = np.flatnonzero(cp.paired)
        out[:, paired] = cp.sign[paired] * components[:, cp.partner[paired]]
        unpaired = cp.unpaired
        out[:, unpaired] = self._evaluate(cp_conjugate(events), unpaired)
        return out

    def component_pair(self, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r""":math:`A_k(p)` and :math:`A_k(\bar{p})`, each ``(N, K)``: what fits need to precompute."""
        components = self.component_amplitudes(events)
        return components, self.conjugate_components(events, components)

    def amplitudes(
        self, events: np.ndarray, couplings: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        r""":math:`A_f(p) = \sum_k c_k A_k(p)` and :math:`\bar{A}_f(p) = \sum_k c_k A_k(\bar{p})`."""
        c = self.couplings if couplings is None else np.asarray(couplings)
        components, conjugates = self.component_pair(events)
        return components @ c, conjugates @ c

    # --- diagnostics
    def cp_even_fraction(
        self, size: int = 200_000, seed: int = 1
    ) -> tuple[float, float]:
        r""":math:`F_+ = \int|A_f + \bar{A}_f|^2 / (2\int(|A_f|^2 + |\bar{A}_f|^2))` and its Monte Carlo uncertainty.

        The integrals are over flat phase space (``size`` events, ``seed``). The
        uncertainty is that of a ratio of correlated means (delta method).
        """
        events = from_sample(self.phase_space(size, seed))
        a, a_bar = self.amplitudes(events)
        numerator = np.abs(a + a_bar) ** 2
        denominator = 2 * (np.abs(a) ** 2 + np.abs(a_bar) ** 2)
        ratio = numerator.mean() / denominator.mean()
        residual = (numerator - ratio * denominator) / denominator.mean()
        return float(ratio), float(residual.std(ddof=1) / np.sqrt(size))
