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
numbers per event and per batch, whatever passes. The samples are built once per study
and stored as :math:`(\tau, A_f, \bar{A}_f, w)` in ``norm/<scenario>.npz``. The
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
``closure`` normalisation sample. The shift of part :math:`k`, averaged over the toys,
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
import json
import multiprocessing
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from d02hhhh_toys.acceptance import AcceptanceConfig, LHCbAcceptance
from d02hhhh_toys.fit import PARAMETERS, SCALE, MixingNLL, RateSample, fit
from d02hhhh_toys.generate import AmplitudeSampler, MixingGenerator
from d02hhhh_toys.model import AmplitudeModel
from d02hhhh_toys.normalisation import normalisation_sample
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
    norm_check_parts: int = 4  # K of the normalisation-offset check (0: none)
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


# --- normalisation samples


def _decay_times(size: int, tau_max: float, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(size=size)
    return -np.log1p(-u * -np.expm1(-tau_max))


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


def base_sample(
    model: AmplitudeModel,
    config: StudyConfig,
    key: str,
    *,
    generator: MixingGenerator | None = None,
) -> RateSample:
    """A base normalisation sample: without acceptance, or after the analyst's acceptance ``key``.

    Every base sample uses the same seed (common random numbers).
    """
    # entropy, not a SeedSequence object, so that every call sees the same numbers
    seed = [config.seed, 1]
    acceptance = None
    if key != NO_ACCEPTANCE:
        overrides = json.loads(key)
        acceptance = LHCbAcceptance(config.true_acceptance().with_overrides(overrides))
    return normalisation_sample(
        model,
        config.n_norm,
        seed,
        acceptance=acceptance,
        method=config.norm_method,
        generator=generator,
        tau_max=config.tau_max,
    )


def derive_normalisation(
    config: StudyConfig, scenario: Scenario, bases: dict[str, RateSample]
) -> RateSample:
    """The normalisation sample of ``scenario`` from its base samples (:func:`base_keys`)."""
    if scenario.mode == "none":
        return bases[NO_ACCEPTANCE]
    sample = bases[acceptance_key(scenario.overrides)]
    rng = np.random.default_rng([config.seed, 2])
    if scenario.mode == "full":
        return sample
    if scenario.mode == "factorised":
        tau = sample.tau[rng.permutation(len(sample))]
        return dataclasses.replace(sample, tau=tau)
    if scenario.mode == "ignore_time":
        return dataclasses.replace(
            sample, tau=_decay_times(len(sample), config.tau_max, rng)
        )
    # ignore_ps: the accepted decay times, with phase-space points without acceptance
    flat = bases[NO_ACCEPTANCE]
    return dataclasses.replace(flat, tau=sample.tau[rng.permutation(len(sample))])


def scenario_normalisation(
    model: AmplitudeModel,
    config: StudyConfig,
    scenario: Scenario,
    *,
    generator: MixingGenerator | None = None,
) -> RateSample:
    """The normalisation sample of ``scenario`` (same seed for every scenario)."""
    bases = {
        key: base_sample(model, config, key, generator=generator)
        for key in base_keys(scenario)
    }
    return derive_normalisation(config, scenario, bases)


def _base_worker(key: str) -> tuple[str, RateSample]:
    state = _STATE
    return key, base_sample(
        state["model"], state["config"], key, generator=state["generator"]
    )


def save_rate_sample(sample: RateSample, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    weight = np.ones(len(sample)) if sample.weight is None else sample.weight
    np.savez(
        path,
        tau=sample.tau,
        amplitude=sample.amplitude,
        amplitude_bar=sample.amplitude_bar,
        weight=weight,
    )


def load_rate_sample(path: Path) -> RateSample:
    with np.load(path) as f:
        return RateSample(f["tau"], f["amplitude"], f["amplitude_bar"], f["weight"])


# --- toys

_STATE: dict[str, Any] = {}


def _fit_record(result) -> dict[str, Any]:
    return {
        "values": result.values,
        "errors": result.errors,
        "valid": result.valid,
        "fval": result.fval,
        "nfcn": result.nfcn,
    }


def run_toy(index: int) -> dict[str, Any]:
    """Generate toy ``index`` and fit it with every scenario (uses the state set by :func:`run_study`)."""
    config: StudyConfig = _STATE["config"]
    generator: MixingGenerator = _STATE["generator"]
    acceptance: LHCbAcceptance = _STATE["acceptance"]
    norms: dict[str, RateSample] = _STATE["norms"]
    start = time.time()
    seeds = np.random.SeedSequence([config.seed, 0, index]).spawn(2)
    data = {
        tag: RateSample.from_toy(
            generator.generate(tag, config.n_per_tag, seed, acceptance=acceptance)
        )
        for tag, seed in zip((D0, D0BAR), seeds, strict=True)
    }
    generated = time.time()
    results = {}
    for scenario in config.scenarios:
        norm = norms[scenario.name]
        nll = MixingNLL(data, {D0: norm, D0BAR: norm})
        results[scenario.name] = _fit_record(fit(nll, config.parameters))
    parts = []
    for norm in _parts(norms[CLOSURE], config.norm_check_parts):
        nll = MixingNLL(data, {D0: norm, D0BAR: norm})
        parts.append(_fit_record(fit(nll, config.parameters)))
    return {
        "toy": index,
        "seed": [config.seed, 0, index],
        "n_per_tag": config.n_per_tag,
        "time_generate": generated - start,
        "time_fit": time.time() - generated,
        "results": results,
        "closure_parts": parts,
    }


def _parts(sample: RateSample, k: int) -> list[RateSample]:
    """``k`` disjoint, equal parts of a normalisation sample (independent events)."""
    n = len(sample) // k if k else 0

    def part(sl: slice) -> RateSample:
        weight = None if sample.weight is None else sample.weight[sl]
        return RateSample(
            sample.tau[sl], sample.amplitude[sl], sample.amplitude_bar[sl], weight
        )

    return [part(slice(i * n, (i + 1) * n)) for i in range(k)]


def _done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {json.loads(line)["toy"] for line in path.read_text().splitlines() if line}


def run_study(
    config: StudyConfig,
    outdir: str | Path,
    *,
    workers: int = 1,
    log=print,
) -> dict[str, Any]:
    """Run (or resume) the study in ``outdir``; returns the summary.

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
    sampler = AmplitudeSampler(model, safety=config.sampler_safety)
    generator = MixingGenerator(sampler, config.parameters, tau_max=config.tau_max)
    _STATE.update(
        model=model,
        config=config,
        generator=generator,
        acceptance=LHCbAcceptance(config.true_acceptance()),
    )
    norms = {}
    missing = []
    for scenario in config.scenarios:
        path = outdir / "norm" / f"{scenario.name}.npz"
        if path.exists():
            norms[scenario.name] = load_rate_sample(path)
        else:
            missing.append(scenario)
    # each distinct base sample once, in parallel; the scenarios are derived from them
    keys = sorted({key for scenario in missing for key in base_keys(scenario)})
    if keys:
        log(f"Normalisation: {len(keys)} base samples for {len(missing)} scenarios")
    if keys and workers > 1:
        context = multiprocessing.get_context("fork")
        with context.Pool(min(workers, len(keys))) as pool:
            bases = dict(pool.map(_base_worker, keys))
    else:
        bases = dict(map(_base_worker, keys))
    for scenario in missing:
        norms[scenario.name] = derive_normalisation(config, scenario, bases)
        save_rate_sample(norms[scenario.name], outdir / "norm" / f"{scenario.name}.npz")
        log(
            f"Normalisation sample: {scenario.name} ({scenario.mode},"
            f" {scenario.overrides or 'no overrides'}), {len(norms[scenario.name])} events"
        )
    del bases
    _STATE["norms"] = norms
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
