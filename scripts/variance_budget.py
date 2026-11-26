"""The factor v of the normalisation offset, and how it would change with other sample designs.

    .venv/bin/python scripts/variance_budget.py configs/study/smoke.yaml --points 200000 --replicas 16

Draws phase-space points from the study's normalisation density, M replicas of
(decay time, acceptance) per point with the study's true acceptance, and prints
:class:`d02hhhh_toys.variance.VarianceBudget` at the true mixing parameters, with the
cost of each step.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from aman_examples import d02kkpipi
from tensorwaves.data.rng import NumpyUniformRNG

from d02hhhh_toys.acceptance import LHCbAcceptance
from d02hhhh_toys.fit import PARAMETERS
from d02hhhh_toys.model import AmplitudeModel, from_sample
from d02hhhh_toys.normalisation import _decay_times
from d02hhhh_toys.study import StudyConfig
from d02hhhh_toys.variance import variance_budget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--points", type=int, default=200_000)
    parser.add_argument("--replicas", type=int, default=16)
    parser.add_argument("--bunch", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", help="also write the result to this file")
    args = parser.parse_args()

    config = StudyConfig.from_yaml(args.config)
    if config.norm_method not in {"peaked", "flat"}:
        msg = "The budget supports the 'peaked' and 'flat' normalisation densities"
        raise ValueError(msg)
    model = AmplitudeModel(chains=config.chains, backend=config.backend)
    acceptance = LHCbAcceptance(config.true_acceptance())
    if config.norm_method == "peaked":
        domain = d02kkpipi.peaked_phase_space_generator(model.model)
    else:
        domain = d02kkpipi.phase_space_generator(model.model)
    rng_domain = NumpyUniformRNG(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    m = args.replicas
    parts = {"tau": [], "a": [], "abar": [], "w": [], "acc": []}
    cost = {"amplitudes": 0.0, "acceptance": 0.0}
    done = 0
    while done < args.points:
        n = min(args.bunch, args.points - done)
        sample = domain.generate(n, rng_domain)
        p = from_sample(sample)
        w = (
            1 / domain.density_ratio(sample)
            if config.norm_method == "peaked"
            else np.ones(n)
        )
        t0 = time.perf_counter()
        a, abar = model.amplitudes(p)
        t1 = time.perf_counter()
        tau = _decay_times(n * m, config.tau_max, rng)
        accepted = acceptance.accept(np.repeat(p, m, axis=0), tau, rng)
        t2 = time.perf_counter()
        cost["amplitudes"] += t1 - t0
        cost["acceptance"] += t2 - t1
        parts["tau"].append(tau.reshape(n, m))
        parts["acc"].append(accepted.reshape(n, m))
        parts["a"].append(a)
        parts["abar"].append(abar)
        parts["w"].append(w)
        done += n
    joined = {key: np.concatenate(value) for key, value in parts.items()}
    budget = variance_budget(
        joined["tau"],
        joined["a"],
        joined["abar"],
        joined["w"],
        joined["acc"],
        config.parameters,
    )
    per_amplitude = cost["amplitudes"] / args.points
    per_candidate = cost["acceptance"] / (args.points * m)
    print(f"study {config.name}: {args.points} points x {m} replicas")
    print(
        f"efficiency {budget.efficiency:.4f}; points with an accepted replica"
        f" {budget.points_accepted:.3f}"
    )
    print(
        f"cost: {1e3 * per_amplitude:.4f} ms per amplitude evaluation,"
        f" {1e3 * per_candidate:.4f} ms per acceptance candidate"
    )
    print(f"{'v':32s}" + "".join(f"{name:>11s}" for name in PARAMETERS))
    for key, value in budget.v.items():
        print(f"{key:32s}" + "".join(f"{x:11.3f}" for x in value))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "config": args.config,
                    "points": args.points,
                    "replicas": m,
                    "efficiency": budget.efficiency,
                    "points_accepted": budget.points_accepted,
                    "ms_per_amplitude": 1e3 * per_amplitude,
                    "ms_per_candidate": 1e3 * per_candidate,
                    "v": {
                        k: dict(zip(PARAMETERS, map(float, v), strict=True))
                        for k, v in budget.v.items()
                    },
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
