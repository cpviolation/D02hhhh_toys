r"""Parametric LHCb-like acceptance emulator for :math:`D^0 \to h^+h^-h^+h^-` toys.

The emulator acts on events in the :math:`D^0` rest frame and their decay times
:math:`\tau = \Gamma t`. For each event:

1. **Orientation:** the event is rotated at random, so that the acceptance depends only
   on the invariants of the decay and on :math:`\tau`, not on how the generator
   oriented it.
2. **Production:** the :math:`D^0` gets a transverse momentum from
   :math:`dN/dp_T \propto p_T / (1 + (p_T/p_0)^2)^n` on :math:`[0, p_{T,\max}]`, a
   pseudorapidity uniform in :math:`[\eta_\min, \eta_\max]` and a uniform azimuth. The
   daughters are boosted to the laboratory.
3. **Flight and IP:** the decay vertex is at :math:`\vec{L} = (\vec{p}/m)\,c\tau_{D^0}\,\tau`
   from the primary vertex, which is at the origin. Each daughter's impact parameter is
   :math:`|\vec{L} \times \hat{u}|`. It is smeared with a Gaussian of
   :math:`\sigma_\mathrm{IP} = s\,(a + b/p_T)`, and :math:`\mathrm{IP}\chi^2 =
   (\mathrm{IP}_\mathrm{meas}/\sigma_\mathrm{IP})^2`.
4. **Tracks:** each daughter must satisfy the :math:`\eta`, :math:`p` and :math:`p_T`
   requirements, and is reconstructed with probability :math:`\varepsilon(p) =
   \varepsilon_\max (1 - e^{-(p - p_\mathrm{thr})/w})` (0 below :math:`p_\mathrm{thr}`).
5. **Selection:** all four daughters reconstructed; offline, at least
   :math:`N_\mathrm{off}` daughters with :math:`\mathrm{IP}\chi^2` above a cut
   (:class:`Offline`; :math:`N_\mathrm{off} = 0` by default, so that the
   :math:`\mathrm{IP}\chi^2` requirement enters only through the 1- or 2-track HLT1
   lines; 4 is a cut on every daughter); and the HLT1 decision.

**HLT1** is a list of lines, combined with ``mode = "any"`` (the event passes if any
line fires, as in the real HLT1) or ``mode = "mixture"`` (each event is assigned one line
at random with ``fractions``, e.g. for data-taking periods with different trigger
settings). Only reconstructed daughters count. Three kinds of line exist:

- :class:`OneTrackLine` (``kind: one_track``): a daughter with
  :math:`p_T > p_\mathrm{wall}` and

  .. math::

      \ln\frac{\mathrm{IP}\chi^2}{\mathrm{IP}\chi^2_\mathrm{floor}} > \frac{k}{p_T - p_\mathrm{wall}},

  a hyperbola in the :math:`(p_T, \ln\mathrm{IP}\chi^2)` plane. The required
  :math:`\mathrm{IP}\chi^2` diverges at the wall and tends to the floor at high
  :math:`p_T`, so that soft tracks must be very displaced and hard ones only slightly.
- :class:`TwoTrackLine` (``kind: two_track``): a pair of daughters, each with
  :math:`p_T` and :math:`\mathrm{IP}\chi^2` above loose cuts, forming a vertex that
  passes a preselection (vertex :math:`p_T`, corrected mass, DOCA, vertex
  :math:`\chi^2`, :math:`\eta` of the flight direction) and a linear score

  .. math::

      w_\mathrm{FD}\ln\mathrm{FD}\chi^2 + w_{p_T}\textstyle\sum p_T
      + w_\mathrm{IP}\ln\min\mathrm{IP}\chi^2 - w_{\chi^2}\chi^2_\mathrm{vtx} > s_\min.

  The vertex is emulated: the DOCA is :math:`|g|\sqrt{\sigma_{\mathrm{IP},1}^2 +
  \sigma_{\mathrm{IP},2}^2}` with a standard normal :math:`g`, so that
  :math:`\chi^2_\mathrm{vtx} = g^2`, and the flight distance is smeared with the
  resolution :class:`VertexResolution` along the flight direction.
- :class:`BoxLine` (``kind: box``): at least :math:`N` daughters with :math:`p_T` and
  :math:`\mathrm{IP}\chi^2` above fixed cuts (a cross-check).

These lines follow the ideas of the LHCb HLT1 inclusive hadron lines (a
:math:`p_T`-dependent displacement requirement for one track, a multivariate
two-track vertex selection), with their own functional forms and toy parameters.

Nothing depends on the charges, so the acceptance is CP-symmetric:
:math:`\varepsilon(\tau, \bar{p}) = \varepsilon(\tau, p)`.

The numbers in :class:`AcceptanceConfig` are toy values, to be tuned on Run 3
simulation. Units: GeV, µm.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

D0_MASS = 1.86484  # GeV, as in the AmpGen particle table of the model
D0_CTAU_UM = 122.9  # µm, PDG


@dataclass
class Production:
    pt_scale: float = 2.5  # GeV, p0
    pt_power: float = 3.0  # n
    pt_max: float = 30.0  # GeV
    eta_min: float = 2.0
    eta_max: float = 5.0


@dataclass
class IPResolution:
    a_um: float = 12.0
    b_um_gev: float = 24.0
    scale: float = 1.0


@dataclass
class TrackRequirements:
    eta_min: float = 2.0
    eta_max: float = 5.0
    p_min: float = 2.0  # GeV
    pt_min: float = 0.25  # GeV


@dataclass
class TrackingEfficiency:
    plateau: float = 0.97
    threshold: float = 1.5  # GeV, p_thr
    width: float = 1.5  # GeV, w


@dataclass
class Offline:
    n_tracks: int = 0  # daughters required with IPchi2 > ipchi2_min (0: no cut, 4: all)
    ipchi2_min: float = 4.0

    def __post_init__(self) -> None:
        if not 0 <= self.n_tracks <= 4:
            msg = f"offline.n_tracks must be between 0 and 4, got {self.n_tracks}"
            raise ValueError(msg)


@dataclass
class VertexResolution:
    """Resolution of the flight distance (PV and decay vertex combined), µm."""

    sigma_xy_um: float = 20.0
    sigma_z_um: float = 100.0


@dataclass
class BoxLine:
    name: str
    n_tracks: int
    pt_min: float  # GeV
    ipchi2_min: float
    kind: str = "box"

    def decide(
        self, lab: LabEvent, config: AcceptanceConfig, rng: np.random.Generator
    ) -> np.ndarray:
        del config, rng
        good = (
            lab.reconstructed & (lab.pt > self.pt_min) & (lab.ipchi2 > self.ipchi2_min)
        )
        return good.sum(axis=1) >= self.n_tracks


@dataclass
class OneTrackLine:
    name: str = "one_track"
    pt_wall: float = 1.5  # GeV
    k: float = 2.0  # GeV
    ipchi2_floor: float = 8.0
    kind: str = "one_track"

    def ipchi2_threshold(self, pt: np.ndarray) -> np.ndarray:
        """Minimum IPχ² as a function of pT (infinite at or below the wall)."""
        pt = np.asarray(pt, dtype=float)
        excess = pt - self.pt_wall
        with np.errstate(divide="ignore", over="ignore"):
            return np.where(
                excess > 0,
                self.ipchi2_floor * np.exp(self.k / np.where(excess > 0, excess, 1)),
                np.inf,
            )

    def decide(
        self, lab: LabEvent, config: AcceptanceConfig, rng: np.random.Generator
    ) -> np.ndarray:
        del config, rng
        good = lab.reconstructed & (lab.ipchi2 > self.ipchi2_threshold(lab.pt))
        return good.any(axis=1)


#: the six pairs of daughters
PAIRS = np.array([(i, j) for i in range(4) for j in range(i + 1, 4)])


@dataclass
class TwoTrackLine:
    name: str = "two_track"
    track_pt_min: float = 0.2  # GeV
    track_ipchi2_min: float = 4.0
    vertex_pt_min: float = 1.0  # GeV
    mcorr_min: float = 1.0  # GeV
    doca_max_um: float = 200.0
    vertex_chi2_max: float = 20.0
    eta_min: float = 2.0  # of the flight direction
    eta_max: float = 5.0
    w_fd: float = 1.0
    w_pt: float = 0.5  # 1/GeV
    w_ip: float = 0.5
    w_chi2: float = 0.05
    score_min: float = 7.0
    kind: str = "two_track"

    def pair_quantities(
        self, lab: LabEvent, config: AcceptanceConfig, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """Emulated vertex quantities of the six pairs, each ``(N, 6)``."""
        i, j = PAIRS.T
        pair = lab.momenta[:, i] + lab.momenta[:, j]
        p3 = pair[..., :3]
        mass2 = np.clip(pair[..., 3] ** 2 - np.sum(p3**2, axis=-1), 0, None)
        flight = lab.flight
        length = np.linalg.norm(flight, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            n = flight / length[:, None]
        n = np.nan_to_num(n, nan=0.0)
        n[length == 0, 2] = 1.0
        p_along = np.einsum("npk,nk->np", p3, n)
        p_perp = np.sqrt(np.clip(np.sum(p3**2, axis=-1) - p_along**2, 0, None))
        res = config.vertex_resolution
        sigma_fd = np.sqrt(
            res.sigma_xy_um**2 * (n[:, 0] ** 2 + n[:, 1] ** 2)
            + res.sigma_z_um**2 * n[:, 2] ** 2
        )
        fd = length[:, None] + sigma_fd[:, None] * rng.normal(
            size=(len(length), len(PAIRS))
        )
        g = rng.normal(size=(len(length), len(PAIRS)))
        sigma_doca = np.hypot(lab.ip_sigma[:, i], lab.ip_sigma[:, j])
        flight_eta = np.arcsinh(n[:, 2] / np.maximum(np.hypot(n[:, 0], n[:, 1]), 1e-12))
        return {
            "track_ok": lab.reconstructed[:, i]
            & lab.reconstructed[:, j]
            & (np.minimum(lab.pt[:, i], lab.pt[:, j]) > self.track_pt_min)
            & (np.minimum(lab.ipchi2[:, i], lab.ipchi2[:, j]) > self.track_ipchi2_min),
            "vertex_pt": np.hypot(p3[..., 0], p3[..., 1]),
            "mcorr": np.sqrt(mass2 + p_perp**2) + p_perp,
            "doca": np.abs(g) * sigma_doca,
            "vertex_chi2": g**2,
            "fdchi2": (fd / sigma_fd[:, None]) ** 2,
            "sum_pt": lab.pt[:, i] + lab.pt[:, j],
            "min_ipchi2": np.minimum(lab.ipchi2[:, i], lab.ipchi2[:, j]),
            "flight_eta": np.broadcast_to(flight_eta[:, None], g.shape),
        }

    def score(self, q: dict[str, np.ndarray]) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return (
                self.w_fd * np.log(q["fdchi2"])
                + self.w_pt * q["sum_pt"]
                + self.w_ip * np.log(q["min_ipchi2"])
                - self.w_chi2 * q["vertex_chi2"]
            )

    def decide(
        self, lab: LabEvent, config: AcceptanceConfig, rng: np.random.Generator
    ) -> np.ndarray:
        q = self.pair_quantities(lab, config, rng)
        presel = (
            q["track_ok"]
            & (q["vertex_pt"] > self.vertex_pt_min)
            & (q["mcorr"] > self.mcorr_min)
            & (q["doca"] < self.doca_max_um)
            & (q["vertex_chi2"] < self.vertex_chi2_max)
            & (q["flight_eta"] > self.eta_min)
            & (q["flight_eta"] < self.eta_max)
        )
        return np.any(presel & (self.score(q) > self.score_min), axis=1)


LINE_KINDS = {"box": BoxLine, "one_track": OneTrackLine, "two_track": TwoTrackLine}
HLT1Line = BoxLine | OneTrackLine | TwoTrackLine


def make_line(values: dict[str, Any] | HLT1Line) -> HLT1Line:
    """An HLT1 line from its YAML dict; ``kind`` selects the class (default ``box``)."""
    if not isinstance(values, dict):
        return values
    values = dict(values)
    kind = values.pop("kind", "box")
    if kind not in LINE_KINDS:
        msg = f"Unknown HLT1 line kind {kind!r} (known: {sorted(LINE_KINDS)})"
        raise ValueError(msg)
    return LINE_KINDS[kind](**values)


@dataclass
class HLT1:
    lines: list[HLT1Line] = field(
        default_factory=lambda: [OneTrackLine(), TwoTrackLine()]
    )
    mode: str = "any"  # "any" or "mixture"
    fractions: list[float] | None = None  # for "mixture", one per line

    def __post_init__(self) -> None:
        self.lines = [make_line(line) for line in self.lines]
        if not self.lines:
            msg = "HLT1 needs at least one line"
            raise ValueError(msg)
        if self.mode not in {"any", "mixture"}:
            msg = f"HLT1 mode must be 'any' or 'mixture', got {self.mode!r}"
            raise ValueError(msg)
        if self.mode == "mixture":
            if self.fractions is None or len(self.fractions) != len(self.lines):
                msg = "HLT1 mode 'mixture' needs one fraction per line"
                raise ValueError(msg)
            if not np.isclose(sum(self.fractions), 1) or min(self.fractions) < 0:
                msg = f"HLT1 fractions must be non-negative and sum to 1, got {self.fractions}"
                raise ValueError(msg)


@dataclass
class AcceptanceConfig:
    """All parameters of the emulator (toy values)."""

    production: Production = field(default_factory=Production)
    ip_resolution: IPResolution = field(default_factory=IPResolution)
    tracks: TrackRequirements = field(default_factory=TrackRequirements)
    tracking: TrackingEfficiency = field(default_factory=TrackingEfficiency)
    offline: Offline = field(default_factory=Offline)
    vertex_resolution: VertexResolution = field(default_factory=VertexResolution)
    hlt1: HLT1 = field(default_factory=HLT1)
    ctau_um: float = D0_CTAU_UM
    d0_mass: float = D0_MASS

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> AcceptanceConfig:
        return _build(cls, values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AcceptanceConfig:
        """Read a YAML file. A ``base`` key names another file (relative to this one) whose values are overridden."""
        return cls.from_dict(_read_yaml(Path(path)))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def with_overrides(self, overrides: dict[str, Any]) -> AcceptanceConfig:
        """A copy with ``overrides`` applied: nested dicts, or dotted keys such as ``"ip_resolution.scale"``."""
        values = self.to_dict()
        for key, value in overrides.items():
            if isinstance(value, dict) and "." not in key:
                values[key] = _merge(values.get(key, {}), value)
                continue
            *parents, last = key.split(".")
            target = values
            for parent in parents:
                if parent not in target:
                    msg = f"Unknown acceptance parameter {key!r}"
                    raise KeyError(msg)
                target = target[parent]
            if last not in target:
                msg = f"Unknown acceptance parameter {key!r}"
                raise KeyError(msg)
            target[last] = value
        return AcceptanceConfig.from_dict(values)


def _merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(path.read_text()) or {}
    base = values.pop("base", None)
    if base is not None:
        values = _merge(_read_yaml(path.parent / base), values)
    return values


def _build(cls: type, values: dict[str, Any]) -> Any:
    """Build a (nested) dataclass, rejecting unknown keys."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(fields)
    if unknown:
        msg = f"Unknown {cls.__name__} parameters: {sorted(unknown)}"
        raise KeyError(msg)
    kwargs = {}
    for name, value in values.items():
        default = cls.__dataclass_fields__[name].default_factory  # type: ignore[attr-defined]
        nested = default() if default is not dataclasses.MISSING else None
        if dataclasses.is_dataclass(nested) and isinstance(value, dict):
            if isinstance(nested, HLT1):
                value = HLT1(**value)
            else:
                value = _build(type(nested), value)
        kwargs[name] = value
    return cls(**kwargs)


# --- kinematics


def random_rotations(size: int, rng: np.random.Generator) -> np.ndarray:
    """Rotation matrices ``(N, 3, 3)`` uniform in SO(3), from random unit quaternions."""
    q = rng.normal(size=(size, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1
            ),
            np.stack(
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1
            ),
            np.stack(
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1
            ),
        ],
        axis=1,
    )


def boost(p: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Boost four-vectors ``p`` (N, ..., 4) with components (x, y, z, E) by velocities ``beta`` (N, 3)."""
    beta = beta.reshape(beta.shape[0], *([1] * (p.ndim - 2)), 3)
    b2 = np.sum(beta**2, axis=-1)
    gamma = 1 / np.sqrt(1 - b2)
    bp = np.sum(beta * p[..., :3], axis=-1)
    factor = np.where(b2 > 0, (gamma - 1) * bp / np.where(b2 > 0, b2, 1), 0.0)
    space = p[..., :3] + (factor + gamma * p[..., 3])[..., None] * beta
    return np.concatenate([space, (gamma * (p[..., 3] + bp))[..., None]], axis=-1)


def sample_pt(
    production: Production, size: int, rng: np.random.Generator
) -> np.ndarray:
    r""":math:`p_T` from :math:`p_T/(1 + (p_T/p_0)^2)^n` on :math:`[0, p_{T,\max}]`, by inverting its CDF."""
    p0, n = production.pt_scale, production.pt_power
    u_max = (production.pt_max / p0) ** 2
    tail = 1 - (1 + u_max) ** (1 - n)
    f = rng.uniform(size=size)
    u = (1 - f * tail) ** (1 / (1 - n)) - 1
    return p0 * np.sqrt(u)


@dataclass
class LabEvent:
    """Laboratory quantities of the daughters, each of shape ``(N, 4)`` (one column per daughter) unless noted."""

    momenta: np.ndarray  # (N, 4, 4): lab four-momenta
    d_momentum: np.ndarray  # (N, 4): lab four-momentum of the D0
    flight: np.ndarray  # (N, 3): decay vertex, µm
    p: np.ndarray
    pt: np.ndarray
    eta: np.ndarray
    ip_true: np.ndarray  # µm
    ip_sigma: np.ndarray  # µm
    ipchi2: np.ndarray
    reconstructed: np.ndarray  # bool: in acceptance and found by the tracking


class LHCbAcceptance:
    """The emulator of the module docstring."""

    def __init__(self, config: AcceptanceConfig | None = None) -> None:
        self.config = config or AcceptanceConfig()

    def lab_event(
        self, events: np.ndarray, tau: np.ndarray, rng: np.random.Generator
    ) -> LabEvent:
        """Steps 1-4: orientation, production, flight, IP and tracking."""
        cfg = self.config
        events = np.asarray(events, dtype=float)
        size = len(events)
        rotation = random_rotations(size, rng)
        rest = events.copy()
        rest[..., :3] = np.einsum("nij,nkj->nki", rotation, events[..., :3])
        # production
        pt_d = sample_pt(cfg.production, size, rng)
        eta_d = rng.uniform(cfg.production.eta_min, cfg.production.eta_max, size)
        phi_d = rng.uniform(0, 2 * np.pi, size)
        p3_d = np.column_stack(
            [pt_d * np.cos(phi_d), pt_d * np.sin(phi_d), pt_d * np.sinh(eta_d)]
        )
        e_d = np.sqrt(np.sum(p3_d**2, axis=1) + cfg.d0_mass**2)
        lab = boost(rest, p3_d / e_d[:, None])
        # flight and impact parameters
        flight = p3_d / cfg.d0_mass * cfg.ctau_um * np.asarray(tau)[:, None]
        p3 = lab[..., :3]
        p = np.linalg.norm(p3, axis=-1)
        pt = np.hypot(p3[..., 0], p3[..., 1])
        eta = np.arcsinh(p3[..., 2] / pt)
        direction = p3 / p[..., None]
        ip_true = np.linalg.norm(np.cross(flight[:, None, :], direction), axis=-1)
        res = cfg.ip_resolution
        ip_sigma = res.scale * (res.a_um + res.b_um_gev / pt)
        ip_measured = np.abs(ip_true + ip_sigma * rng.normal(size=ip_true.shape))
        ipchi2 = (ip_measured / ip_sigma) ** 2
        # tracks
        trk = cfg.tracks
        in_acceptance = (
            (eta > trk.eta_min)
            & (eta < trk.eta_max)
            & (p > trk.p_min)
            & (pt > trk.pt_min)
        )
        eff = cfg.tracking
        efficiency = eff.plateau * -np.expm1(
            -np.clip(p - eff.threshold, 0, None) / eff.width
        )
        found = rng.uniform(size=p.shape) < efficiency
        return LabEvent(
            momenta=lab,
            d_momentum=np.column_stack([p3_d, e_d]),
            flight=flight,
            p=p,
            pt=pt,
            eta=eta,
            ip_true=ip_true,
            ip_sigma=ip_sigma,
            ipchi2=ipchi2,
            reconstructed=in_acceptance & found,
        )

    def hlt1_lines(self, lab: LabEvent, rng: np.random.Generator) -> np.ndarray:
        """Decision of each HLT1 line, ``(N, n_lines)``. Only reconstructed daughters count."""
        decisions = [
            line.decide(lab, self.config, rng) for line in self.config.hlt1.lines
        ]
        return np.column_stack(decisions)

    def hlt1(self, lab: LabEvent, rng: np.random.Generator) -> np.ndarray:
        lines = self.hlt1_lines(lab, rng)
        config = self.config.hlt1
        if config.mode == "any":
            return lines.any(axis=1)
        chosen = rng.choice(len(config.lines), size=len(lines), p=config.fractions)
        return lines[np.arange(len(lines)), chosen]

    def offline(self, lab: LabEvent) -> np.ndarray:
        cut = self.config.offline
        passed = (lab.reconstructed & (lab.ipchi2 > cut.ipchi2_min)).sum(axis=1)
        return lab.reconstructed.all(axis=1) & (passed >= cut.n_tracks)

    def accept(
        self, events: np.ndarray, tau: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Whether each event passes: all daughters reconstructed, the offline IPχ² requirement and HLT1."""
        lab = self.lab_event(events, tau, rng)
        return self.offline(lab) & self.hlt1(lab, rng)
