r"""Toy-study driver: generate each toy once, fit it with every scenario, summarise.

A **scenario** is the analyst's description of the acceptance, used to build the
normalisation sample. The data are always generated with the true acceptance. A
scenario has:

- a ``mode``, the assumption on the acceptance (:data:`MODES`):

  ================  ==============================================================
  ``full``          correct :math:`\varepsilon(\tau, p)`
  ``factorised``    :math:`\varepsilon_\mathrm{PS}(p)\,\varepsilon_t(\tau)`: the decay
                    times are permuted among the accepted normalisation events
  ``ignore_time``   :math:`\varepsilon_\mathrm{PS}(p)` only: new decay times from
                    :math:`e^{-\tau}` for the accepted events
  ``ignore_ps``     :math:`\varepsilon_t(\tau)` only: the accepted decay times, with
                    phase-space points drawn without acceptance
  ``none``          no acceptance
  ================  ==============================================================

- ``overrides`` of the analyst's acceptance model with respect to the truth, which
  mimic data/simulation differences (:meth:`.AcceptanceConfig.with_overrides`).

The scenario named ``closure`` must be ``full`` without overrides.

**Common random numbers.** Every scenario's normalisation sample uses the same seed,
so the scenarios differ only by their acceptance model. The emulator draws its random
numbers per event and per batch, whatever passes. Each base sample (one per distinct
acceptance model, and one without acceptance) is generated once per study, in chunks
of ``norm_chunk`` events with one seed per chunk. In each chunk all base samples
share the phase-space candidates and the amplitude evaluations, and the chunks run in
parallel. The scenarios are derived from the base samples as decay-time
moments (:class:`.moments.NormMoments`), stored in ``norm/<scenario>.npz``. The
couplings are fixed, so :math:`A_f` and :math:`\bar{A}_f` replace the :math:`A_k`.

**Outputs** in the study directory:

- ``toys.jsonl``: one line per toy, with each scenario's fitted values, errors and
  status. A study is resumed by skipping the toys already there.
- ``summary.json``: the closure pulls (mean and width per parameter), and for each
  scenario the paired shift :math:`(v_s - v_\mathrm{closure})/\sigma_\mathrm{closure}`
  (mean ± error, toy by toy).

**Finite normalisation sample.** All toys share the normalisation samples, so their
finite size gives an offset common to all toys, of order
:math:`\sigma_\mathrm{stat}\sqrt{v N_\mathrm{data}/N_\mathrm{norm}}`. The factor
:math:`v` depends on the model, the acceptance and the parameter, and can be well above
1 (a 0.8 :math:`\sigma` offset of :math:`y_{CP}` was seen with
:math:`N_\mathrm{norm}/N_\mathrm{data} = 20`). It is therefore measured: each toy is
also fitted with each of ``norm_check_parts`` = :math:`K` disjoint parts of the
``closure`` normalisation sample (chunk index modulo :math:`K`). The shift of part :math:`k`, averaged over the toys,
:math:`m_k`, is one draw of the offset of a sample :math:`K` times smaller, so the
summary reports :math:`\sigma_\mathrm{norm} = \mathrm{sd}_k(m_k)/\sqrt{K}`, in units
of :math:`\sigma_\mathrm{stat}`: the expected size of the common offset of the closure.
The spread over toys of a difference between parts does *not* measure it, because the
offset is the same for every toy.
The paired shifts are much less affected, because the common random numbers make the
offsets of the scenarios nearly the same.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import multiprocessing
import operator
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from d02hhhh_toys.acceptance import AcceptanceConfig, LHCbAcceptance
from d02hhhh_toys.fit import PARAMETERS, SCALE, MixingNLL, RateSample, fit
from d02hhhh_toys.generate import AmplitudeSampler, MixingGenerator, ToySample
from d02hhhh_toys.model import AmplitudeModel
from d02hhhh_toys.moments import ANALYTIC_ORDER, NormMoments, decay_time_moments
from d02hhhh_toys.normalisation import (
    normalisation_moments,
    shared_normalisation_moments,
)
from d02hhhh_toys.timeevolution import D0, D0BAR, MixingParameters

MODES = ("full", "factorised", "ignore_time", "ignore_ps", "none")
CLOSURE = "closure"


@dataclass
class Scenario:
    name: str
    mode: str = "full"
    overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            msg = f"Scenario {self.name!r}: unknown mode {self.mode!r} (known: {MODES})"
            raise ValueError(msg)


@dataclass
class StudyConfig:
    """A toy study. Paths are relative to the YAML file that defines them."""

    name: str
    truth: dict[str, float]  # x_cp_mix, y_cp_mix, dx_mix, dy_mix (plain numbers)
    n_per_tag: int
    n_norm: int
    n_toys: int
    seed: int
    acceptance: str  # YAML of the true acceptance
    scenarios: list[Scenario]
    chains: list[str] | None = None  # None: the full model
    backend: str = "numpy"
    norm_method: str = "peaked"
    tau_max: float = 20.0
    sampler_safety: float = 1.2
    sampler_maximum: float | None = None  # fixed |A|^2/r maximum (None: estimate)
    norm_check_parts: int = 4  # K of the normalisation-offset check (0: none)
    norm_chunk: int = 250_000  # accepted normalisation events per chunk (parallel unit)
    base_dir: str = "."

    def __post_init__(self) -> None:
        self.scenarios = [
            s if isinstance(s, Scenario) else Scenario(**s) for s in self.scenarios
        ]
        names = [s.name for s in self.scenarios]
        if len(set(names)) != len(names):
            msg = f"Duplicate scenario names: {names}"
            raise ValueError(msg)
        closure = [s for s in self.scenarios if s.name == CLOSURE]
        if not closure or closure[0].mode != "full" or closure[0].overrides:
            msg = f"The study needs a {CLOSURE!r} scenario with mode 'full' and no overrides"
            raise ValueError(msg)
        unknown = set(self.truth) - set(PARAMETERS)
        if unknown:
            msg = f"Unknown truth parameters: {sorted(unknown)}"
            raise KeyError(msg)

    @classmethod
    def from_yaml(cls, path: str | Path) -> StudyConfig:
        path = Path(path)
        values = yaml.safe_load(path.read_text())
        values.setdefault("base_dir", str(path.parent))
        return cls(**values)

    @property
    def parameters(self) -> MixingParameters:
        return MixingParameters(**self.truth)

    def true_acceptance(self) -> AcceptanceConfig:
        return AcceptanceConfig.from_yaml(Path(self.base_dir) / self.acceptance)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --- normalisation samples, as decay-time moments


#: key of the base sample without acceptance
NO_ACCEPTANCE = "none"


def acceptance_key(overrides: dict[str, Any]) -> str:
    """Key of the base sample after the acceptance with these overrides."""
    return json.dumps(overrides, sort_keys=True)


def base_keys(scenario: Scenario) -> list[str]:
    """The base samples a scenario is derived from."""
    if scenario.mode == "none":
        return [NO_ACCEPTANCE]
    keys = [acceptance_key(scenario.overrides)]
    if scenario.mode == "ignore_ps":
        keys.append(NO_ACCEPTANCE)
    return keys


def norm_chunks(config: StudyConfig) -> list[int]:
    """Number of accepted events of each chunk of a base sample.

    The number of chunks is a multiple of the ``norm_check_parts``, so that each part
    is a set of whole chunks; the sizes add up to ``n_norm``.
    """
    k = max(config.norm_check_parts, 1)
    n = k * math.ceil(config.n_norm / (k * config.norm_chunk))
    size, extra = divmod(config.n_norm, n)
    return [size + (i < extra) for i in range(n)]


def base_chunks(
    model: AmplitudeModel,
    config: StudyConfig,
    keys: list[str],
    chunk: int,
    *,
    generator: MixingGenerator | None = None,
) -> dict[str, NormMoments]:
    """Moments of chunk ``chunk`` of the base samples ``keys``: without acceptance, or after the analyst's acceptance ``key``.

    A chunk has the same seed for every key (common random numbers). With the
    ``peaked`` and ``flat`` methods all keys share the candidates and the amplitude
    evaluations (:func:`.normalisation.shared_normalisation_moments`).
    """
    # entropy, not a SeedSequence object, so that every call sees the same numbers
    seed = [config.seed, 1, chunk]
    size = norm_chunks(config)[chunk]
    true = config.true_acceptance()
    acceptances = {
        key: None
        if key == NO_ACCEPTANCE
        else LHCbAcceptance(true.with_overrides(json.loads(key)))
        for key in keys
    }
    if config.norm_method == "proposal":
        return {
            key: normalisation_moments(
                model,
                size,
                seed,
                acceptance=acceptance,
                method="proposal",
                generator=generator,
                tau_max=config.tau_max,
            )
            for key, acceptance in acceptances.items()
        }
    return shared_normalisation_moments(
        model,
        size,
        seed,
        acceptances,
        method=config.norm_method,
        tau_max=config.tau_max,
    )


def base_moments(
    chunks: dict[int, NormMoments], parts: int
) -> tuple[NormMoments, list[NormMoments]]:
    """The whole base sample, and its ``parts`` disjoint parts (chunk index modulo ``parts``)."""
    ordered = [chunks[i] for i in sorted(chunks)]
    total = functools.reduce(operator.add, ordered)
    split = [functools.reduce(operator.add, ordered[k::parts]) for k in range(parts)]
    return total, split


def derive_normalisation(
    config: StudyConfig, scenario: Scenario, bases: dict[str, NormMoments]
) -> NormMoments:
    r"""The normalisation moments of ``scenario`` from its base samples (:func:`base_keys`).

    Each new decay time is independent of its event, so a mode that replaces the decay
    times multiplies the zeroth moments by those of the new decay times
    (:meth:`.NormMoments.with_decay_time_moments`):

    - ``factorised``: the decay times of the accepted events;
    - ``ignore_time``: :math:`e^{-\tau}` on :math:`[0, \tau_\max)`, to order
      ``ANALYTIC_ORDER`` (its fit ends far from the truth, where order 20 does not
      converge);
    - ``none``: the points without acceptance, with :math:`e^{-\tau}` on
      :math:`[0, \tau_\max)` as for ``ignore_time`` (without acceptance the decay time
      is independent of the point, so this is the exact limit);
    - ``ignore_ps``: the points without acceptance, with the decay times of the accepted
      events.
    """
    if scenario.mode == "none":
        moments = decay_time_moments(config.tau_max, ANALYTIC_ORDER)
        return bases[NO_ACCEPTANCE].with_decay_time_moments(moments, config.tau_max)
    sample = bases[acceptance_key(scenario.overrides)]
    if scenario.mode == "full":
        return sample
    if scenario.mode == "factorised":
        return sample.with_decay_time_moments(sample.mean_tau_powers, sample.tau_max)
    if scenario.mode == "ignore_time":
        moments = decay_time_moments(config.tau_max, ANALYTIC_ORDER)
        return sample.with_decay_time_moments(moments, config.tau_max)
    flat = bases[NO_ACCEPTANCE]
    return flat.with_decay_time_moments(sample.mean_tau_powers, sample.tau_max)


def scenario_normalisation(
    model: AmplitudeModel,
    config: StudyConfig,
    scenario: Scenario,
    *,
    generator: MixingGenerator | None = None,
) -> NormMoments:
    """The normalisation moments of ``scenario`` (same seeds for every scenario)."""
    keys = base_keys(scenario)
    done = [
        base_chunks(model, config, keys, i, generator=generator)
        for i in range(len(norm_chunks(config)))
    ]
    bases = {
        key: functools.reduce(operator.add, (d[key] for d in done)) for key in keys
    }
    return derive_normalisation(config, scenario, bases)


def _base_worker(task: tuple[list[str], int]) -> tuple[int, dict[str, NormMoments]]:
    keys, chunk = task
    state = _STATE
    moments = base_chunks(
        state["model"], state["config"], keys, chunk, generator=state["generator"]
    )
    return chunk, moments


_MOMENT_FIELDS = ("aa", "bb", "ab", "tau_powers")


def save_moments(moments: NormMoments, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{name: getattr(moments, name) for name in _MOMENT_FIELDS},
        scalars=np.array([moments.weight_sum, moments.count, moments.tau_max]),
    )


def load_moments(path: Path) -> NormMoments:
    with np.load(path) as f:
        if "aa" not in f:
            msg = (
                f"{path} is a per-event normalisation sample from an earlier version;"
                " delete the norm/ directory to rebuild it as moments"
            )
            raise ValueError(msg)
        weight_sum, count, tau_max = f["scalars"]
        return NormMoments(
            **{name: f[name] for name in _MOMENT_FIELDS},
            weight_sum=float(weight_sum),
            count=int(count),
            tau_max=float(tau_max),
        )


# --- toys

_STATE: dict[str, Any] = {}


def _fit_record(result) -> dict[str, Any]:
    return {
        "values": result.values,
        "errors": result.errors,
        "valid": result.valid,
        "fval": result.fval,
        "nfcn": result.nfcn,
        "edm": float(result.minuit.fmin.edm),
    }


# --- stored toys

TOYS_DIR = "toys"


def generation_key(config: StudyConfig) -> dict[str, Any]:
    """Everything that determines the toys of ``config``: toys with equal keys are equal."""
    return {
        "seed": config.seed,
        "n_per_tag": config.n_per_tag,
        "truth": dict(config.truth),
        "tau_max": config.tau_max,
        "chains": config.chains,
        "sampler_safety": config.sampler_safety,
        "sampler_maximum": config.sampler_maximum,
        "acceptance": config.true_acceptance().to_dict(),
    }


def toy_path(directory: str | Path, index: int) -> Path:
    return Path(directory) / f"toy_{index:05d}.npz"


def save_toy(
    path: str | Path, index: int, key: dict[str, Any], toys: dict[int, ToySample]
) -> None:
    """Write the events, decay times and amplitudes of both tags of toy ``index``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for tag, toy in toys.items():
        name = _TAG_NAMES[tag]
        arrays |= {
            f"{name}_events": toy.events,
            f"{name}_tau": toy.tau,
            f"{name}_amplitude": toy.amplitude,
            f"{name}_amplitude_bar": toy.amplitude_bar,
        }
    meta = json.dumps({"toy": index, "generation": key})
    tmp = path.with_name(path.name + ".tmp.npz")  # atomic: never a half-written toy
    np.savez(tmp, meta=np.array(meta), **arrays)
    tmp.replace(path)


def load_toy(path: str | Path, index: int, key: dict[str, Any]) -> dict[int, ToySample]:
    """Read toy ``index``; raise ``ValueError`` if it was generated with another key."""
    with np.load(path) as f:
        meta = json.loads(str(f["meta"]))
        # through JSON, so that tuples and lists compare equal
        if meta["toy"] != index or meta["generation"] != json.loads(json.dumps(key)):
            msg = (
                f"{path} was generated with other settings than this study"
                " (seed, size, truth, acceptance or sampler)"
            )
            raise ValueError(msg)
        return {
            tag: ToySample(
                tag,
                f[f"{name}_events"],
                f[f"{name}_tau"],
                f[f"{name}_amplitude"],
                f[f"{name}_amplitude_bar"],
            )
            for tag, name in _TAG_NAMES.items()
        }


_TAG_NAMES = {D0: "d0", D0BAR: "d0bar"}


def run_toy(index: int) -> dict[str, Any]:
    """Generate (or read) toy ``index`` and fit it with every scenario (uses the state set by :func:`run_study`)."""
    config: StudyConfig = _STATE["config"]
    generator: MixingGenerator = _STATE["generator"]
    acceptance: LHCbAcceptance = _STATE["acceptance"]
    norms: dict[str, NormMoments] = _STATE["norms"]
    key = _STATE["generation_key"]
    start = time.time()
    source = None
    for directory in _STATE["toy_sources"]:
        if toy_path(directory, index).exists():
            source = toy_path(directory, index)
            break
    if source is not None:
        toys = load_toy(source, index, key)
    else:
        seeds = np.random.SeedSequence([config.seed, 0, index]).spawn(2)
        toys = {
            tag: generator.generate(tag, config.n_per_tag, seed, acceptance=acceptance)
            for tag, seed in zip((D0, D0BAR), seeds, strict=True)
        }
    store = _STATE["store_toys"]
    if store is not None and (source is None or source.parent != store):
        save_toy(toy_path(store, index), index, key, toys)
    data = {tag: RateSample.from_toy(toy) for tag, toy in toys.items()}
    generated = time.time()
    results = {}
    for scenario in config.scenarios:
        norm = norms[scenario.name]
        nll = MixingNLL(data, {D0: norm, D0BAR: norm})
        results[scenario.name] = _fit_record(fit(nll, config.parameters))
    parts = []
    for norm in _STATE["closure_parts"]:
        nll = MixingNLL(data, {D0: norm, D0BAR: norm})
        parts.append(_fit_record(fit(nll, config.parameters)))
    return {
        "toy": index,
        "seed": [config.seed, 0, index],
        "n_per_tag": config.n_per_tag,
        "toy_source": "generated" if source is None else str(source),
        "time_generate": generated - start,
        "time_fit": time.time() - generated,
        "results": results,
        "closure_parts": parts,
    }


def _done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {json.loads(line)["toy"] for line in path.read_text().splitlines() if line}


def run_study(
    config: StudyConfig,
    outdir: str | Path,
    *,
    workers: int = 1,
    store_toys: bool = False,
    toys_from: str | Path | None = None,
    log=print,
) -> dict[str, Any]:
    """Run (or resume) the study in ``outdir``; returns the summary.

    - ``store_toys``: write each toy (events, decay times, amplitudes of both tags,
      about 170 bytes per event) to ``outdir/toys/``.
    - ``toys_from``: read the toys from another study directory's ``toys/`` instead of
      generating them, e.g. to fit them with new scenarios. Its toys must have been
      generated with the same seed, size, truth, acceptance and sampler
      (:func:`generation_key`); missing toys are generated.

    A toy already in ``outdir/toys/`` is always read rather than generated.

    With ``workers > 1`` the toys run in forked processes that share the model and the
    normalisation samples. This needs the NumPy backend: JAX does not survive a fork.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if workers > 1 and config.backend != "numpy":
        msg = "Parallel toys need backend 'numpy' (JAX cannot be forked); use workers=1"
        raise ValueError(msg)
    (outdir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    log(f"Building the model ({config.backend})")
    model = AmplitudeModel(chains=config.chains, backend=config.backend)
    sampler = AmplitudeSampler(
        model, safety=config.sampler_safety, maximum=config.sampler_maximum
    )
    generator = MixingGenerator(sampler, config.parameters, tau_max=config.tau_max)
    _STATE.update(
        model=model,
        config=config,
        generator=generator,
        acceptance=LHCbAcceptance(config.true_acceptance()),
        generation_key=generation_key(config),
        store_toys=outdir / TOYS_DIR if store_toys else None,
        toy_sources=[
            *([Path(toys_from) / TOYS_DIR] if toys_from is not None else []),
            outdir / TOYS_DIR,
        ],
    )
    norm_dir = outdir / "norm"
    k = config.norm_check_parts
    names = [s.name for s in config.scenarios]
    part_names = [f"{CLOSURE}.part{i}" for i in range(k if k > 1 else 0)]
    paths = {name: norm_dir / f"{name}.npz" for name in [*names, *part_names]}
    if all(path.exists() for path in paths.values()):
        log("Normalisation: reusing the moments in norm/")
        norms = {name: load_moments(paths[name]) for name in names}
        parts = [load_moments(paths[name]) for name in part_names]
    else:
        # every chunk of every distinct base sample, in parallel; the scenarios are
        # derived from the sums
        keys = sorted({key for s in config.scenarios for key in base_keys(s)})
        # one task per chunk, for all base samples: they share the candidates
        tasks = [(keys, i) for i in range(len(norm_chunks(config)))]
        log(
            f"Normalisation: {len(keys)} base samples of {config.n_norm} events"
            f" in {len(tasks)} shared chunks, for {len(names)} scenarios"
        )
        if workers > 1:
            context = multiprocessing.get_context("fork")
            with context.Pool(min(workers, len(tasks))) as pool:
                done = pool.map(_base_worker, tasks)
        else:
            done = list(map(_base_worker, tasks))
        chunks: dict[str, dict[int, NormMoments]] = {key: {} for key in keys}
        for i, moments in done:
            for key in keys:
                chunks[key][i] = moments[key]
        bases = {}
        for key in keys:
            bases[key], split = base_moments(chunks[key], max(k, 1))
            if key == acceptance_key({}):
                parts = split if k > 1 else []
        norms = {}
        for scenario in config.scenarios:
            norms[scenario.name] = derive_normalisation(config, scenario, bases)
            log(
                f"Normalisation: {scenario.name} ({scenario.mode},"
                f" {scenario.overrides or 'no overrides'}),"
                f" {norms[scenario.name].count} events"
            )
        for name, moments in [*norms.items(), *zip(part_names, parts, strict=True)]:
            save_moments(moments, paths[name])
    _STATE["norms"] = norms
    _STATE["closure_parts"] = parts
    toys_path = outdir / "toys.jsonl"
    todo = [i for i in range(config.n_toys) if i not in _done(toys_path)]
    log(f"{len(todo)} toys to run, {config.n_toys - len(todo)} already done")
    with toys_path.open("a") as out:
        if workers > 1:
            if threading.active_count() > 1:
                log(
                    "Warning: forking a multi-threaded process (was JAX imported?);"
                    " this can deadlock"
                )
            context = multiprocessing.get_context("fork")
            with context.Pool(workers) as pool:
                for record in pool.imap_unordered(run_toy, todo):
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    log(
                        f"toy {record['toy']} done ({record['time_generate']:.0f} s generation, {record['time_fit']:.0f} s fits)"
                    )
        else:
            for index in todo:
                record = run_toy(index)
                out.write(json.dumps(record) + "\n")
                out.flush()
                log(
                    f"toy {index} done ({record['time_generate']:.0f} s generation, {record['time_fit']:.0f} s fits)"
                )
    summary = summarise(outdir)
    log(f"Summary written to {outdir / 'summary.json'}")
    return summary


# --- summary


def _mean_and_error(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return {
            "mean": float("nan"),
            "error": float("nan"),
            "std": float("nan"),
            "n": 0,
        }
    std = float(x.std(ddof=1)) if n > 1 else float("nan")
    return {"mean": float(x.mean()), "error": std / np.sqrt(n), "std": std, "n": n}


def summarise_records(
    records: list[dict[str, Any]], truth: dict[str, float]
) -> dict[str, Any]:
    """Closure pulls and paired shifts from the toy records (see the module docstring)."""
    scenarios = list(records[0]["results"]) if records else []
    closure_ok = [r for r in records if r["results"][CLOSURE]["valid"]]
    summary: dict[str, Any] = {"n_toys": len(records), "closure_valid": len(closure_ok)}
    pulls = {}
    for name in PARAMETERS:
        true = truth.get(name, 0.0) / SCALE
        values = np.array([r["results"][CLOSURE]["values"][name] for r in closure_ok])
        errors = np.array([r["results"][CLOSURE]["errors"][name] for r in closure_ok])
        pull = (values - true) / errors
        pulls[name] = {
            **_mean_and_error(pull),
            "width_error": float(pull.std(ddof=1) / np.sqrt(2 * (len(pull) - 1)))
            if len(pull) > 1
            else float("nan"),
            "mean_error_1e3": float(np.mean(errors)) if len(errors) else float("nan"),
        }
    summary["closure_pulls"] = pulls
    summary["norm_offset_sigma"] = norm_offset(closure_ok)
    shifts = {}
    for scenario in scenarios:
        if scenario == CLOSURE:
            continue
        both = [r for r in closure_ok if r["results"][scenario]["valid"]]
        shifts[scenario] = {"valid": len(both)}
        for name in PARAMETERS:
            delta = np.array(
                [
                    (
                        r["results"][scenario]["values"][name]
                        - r["results"][CLOSURE]["values"][name]
                    )
                    / r["results"][CLOSURE]["errors"][name]
                    for r in both
                ]
            )
            absolute = np.array(
                [
                    r["results"][scenario]["values"][name]
                    - r["results"][CLOSURE]["values"][name]
                    for r in both
                ]
            )
            shifts[scenario][name] = {
                **_mean_and_error(delta),
                "mean_1e3": float(absolute.mean()) if len(absolute) else float("nan"),
            }
    summary["shifts"] = shifts
    return summary


def norm_offset(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Expected size of the common offset from the normalisation sample, in units of σ_stat (module docstring)."""
    usable = [
        r
        for r in records
        if r.get("closure_parts") and all(p["valid"] for p in r["closure_parts"])
    ]
    k = len(usable[0]["closure_parts"]) if usable else 0
    offset: dict[str, Any] = {"n_toys": len(usable), "parts": k}
    for name in PARAMETERS:
        if k < 2:
            offset[name] = float("nan")
            continue
        shifts = np.array(
            [
                [
                    (p["values"][name] - r["results"][CLOSURE]["values"][name])
                    / r["results"][CLOSURE]["errors"][name]
                    for p in r["closure_parts"]
                ]
                for r in usable
            ]
        )  # (toys, parts)
        means = shifts.mean(axis=0)
        offset[name] = float(means.std(ddof=1) / np.sqrt(k))
        offset[f"{name}_part_means"] = means.tolist()
    return offset


def read_records(outdir: str | Path) -> list[dict[str, Any]]:
    path = Path(outdir) / "toys.jsonl"
    return sorted(
        (json.loads(line) for line in path.read_text().splitlines() if line),
        key=lambda r: r["toy"],
    )


def summarise(outdir: str | Path, *, plots: bool = True) -> dict[str, Any]:
    """Write ``summary.json`` (and the plots) from ``toys.jsonl``."""
    outdir = Path(outdir)
    config = json.loads((outdir / "config.json").read_text())
    records = read_records(outdir)
    summary = summarise_records(records, config["truth"])
    summary["study"] = config["name"]
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    if plots and records:
        from d02hhhh_toys import plotting

        plotting.plot_shifts(summary, outdir / "shifts.png")
        plotting.plot_closure_pulls(
            records, config["truth"], outdir / "closure_pulls.png"
        )
    return summary
