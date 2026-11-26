"""Phase 7: scenarios and the study driver."""

import functools
import json
import operator
from pathlib import Path

import numpy as np
import pytest

from conftest import SUBSET
from d02hhhh_toys.acceptance import LHCbAcceptance
from d02hhhh_toys.moments import decay_time_moments
from d02hhhh_toys.normalisation import normalisation_moments, normalisation_sample
from d02hhhh_toys.study import (
    CLOSURE,
    Scenario,
    StudyConfig,
    base_chunks,
    base_moments,
    load_moments,
    norm_chunks,
    norm_offset,
    read_records,
    run_study,
    save_moments,
    scenario_normalisation,
    summarise_records,
)

CONFIGS = Path(__file__).parents[1] / "configs"
TRUTH = {"x_cp_mix": 0.03, "y_cp_mix": 0.025, "dx_mix": 0.005, "dy_mix": -0.003}


def make_config(**changes):
    values = {
        "name": "test",
        "truth": TRUTH,
        "n_per_tag": 500,
        "n_norm": 6000,
        "n_toys": 2,
        "seed": 3,
        "acceptance": str(CONFIGS / "acceptance" / "base.yaml"),
        "chains": SUBSET,
        "scenarios": [
            {"name": CLOSURE, "mode": "full"},
            {"name": "factorised", "mode": "factorised"},
            {"name": "ignore_time", "mode": "ignore_time"},
            {"name": "ignore_ps", "mode": "ignore_ps"},
            {"name": "none", "mode": "none"},
            {
                "name": "ip+5%",
                "mode": "full",
                "overrides": {"ip_resolution.scale": 1.05},
            },
        ],
        "norm_check_parts": 2,
        "norm_chunk": 2000,
    }
    values.update(changes)
    return StudyConfig(**values)


def test_config_validation():
    with pytest.raises(ValueError, match="closure"):
        make_config(scenarios=[{"name": "full", "mode": "full"}])
    with pytest.raises(ValueError, match="closure"):
        make_config(
            scenarios=[
                {
                    "name": CLOSURE,
                    "mode": "full",
                    "overrides": {"ip_resolution.scale": 2},
                }
            ]
        )
    with pytest.raises(ValueError, match="mode"):
        Scenario("x", "partial")
    with pytest.raises(ValueError, match="Duplicate"):
        make_config(scenarios=[{"name": CLOSURE}, {"name": CLOSURE}])
    with pytest.raises(KeyError):
        make_config(truth={"x": 0.01})


def test_shipped_study_configs():
    for name in ("default", "inflated", "smoke"):
        config = StudyConfig.from_yaml(CONFIGS / "study" / f"{name}.yaml")
        truth = config.true_acceptance()
        for scenario in config.scenarios:
            truth.with_overrides(scenario.overrides)  # every override applies


@pytest.fixture(scope="module")
def norms(subset_model):
    config = make_config()
    return {
        s.name: scenario_normalisation(subset_model, config, s)
        for s in config.scenarios
    }


def test_norm_chunks():
    config = make_config(n_norm=6001, norm_chunk=2000, norm_check_parts=2)
    sizes = norm_chunks(config)
    assert len(sizes) == 4  # a multiple of the parts
    assert sum(sizes) == 6001
    assert max(sizes) - min(sizes) <= 1
    assert len(norm_chunks(make_config(norm_check_parts=0))) == 3


def test_scenario_normalisation_modes(subset_model, norms):
    config = make_config()
    full, none = norms[CLOSURE], norms["none"]
    acceptance = LHCbAcceptance(config.true_acceptance())
    direct = functools.reduce(
        operator.add,
        (
            normalisation_moments(
                subset_model, size, [config.seed, 1, i], acceptance=acceptance
            )
            for i, size in enumerate(norm_chunks(config))
        ),
    )
    assert full.count == config.n_norm
    np.testing.assert_allclose(full.aa, direct.aa, rtol=1e-12)
    assert full.mean_tau_powers[1] > 1.5  # the acceptance removes short decay times
    # factorised: the accepted points, with the accepted decay times made independent
    f = norms["factorised"]
    np.testing.assert_array_equal(f.aa[0], full.aa[0])
    np.testing.assert_allclose(f.aa[3], full.aa[0] * full.mean_tau_powers[3])
    # ignore_time: the accepted points with decay times from e^-tau
    t = norms["ignore_time"]
    np.testing.assert_allclose(
        t.ab[2], full.ab[0] * decay_time_moments(config.tau_max, 2)[2]
    )
    # ignore_ps: the points without acceptance, with the accepted decay times
    p = norms["ignore_ps"]
    np.testing.assert_array_equal(p.bb[0], none.bb[0])
    np.testing.assert_allclose(p.mean_tau_powers, full.mean_tau_powers)
    assert abs(none.mean_tau_powers[1] - 1) < 4 / np.sqrt(none.count)


def test_common_random_numbers(subset_model):
    """An override changes few of the accepted events: most decay times are shared."""
    config = make_config()
    true = config.true_acceptance()
    taus = [
        normalisation_sample(
            subset_model,
            3000,
            [config.seed, 1, 0],
            acceptance=LHCbAcceptance(true.with_overrides(overrides)),
        ).tau
        for overrides in ({}, {"ip_resolution.scale": 1.05})
    ]
    assert np.isin(taus[1], taus[0]).mean() > 0.8


def test_parts_and_chunks(subset_model):
    config = make_config(n_norm=4000, norm_chunk=1000, norm_check_parts=2)
    chunks = {
        i: base_chunks(subset_model, config, ["{}"], i)["{}"]
        for i in range(len(norm_chunks(config)))
    }
    total, parts = base_moments(chunks, 2)
    assert [p.count for p in parts] == [2000, 2000]
    np.testing.assert_allclose((parts[0] + parts[1]).aa, total.aa)
    np.testing.assert_array_equal(parts[1].aa, (chunks[1] + chunks[3]).aa)


def test_shared_base_samples_equal_separate_ones(subset_model):
    """Sharing candidates and amplitudes between acceptance models changes no moment."""
    config = make_config(n_norm=3000, norm_chunk=3000, norm_check_parts=0)
    keys = ['{"ip_resolution.scale": 1.05}', "{}", "none"]
    shared = base_chunks(subset_model, config, keys, 0)
    for key in keys:
        alone = base_chunks(subset_model, config, [key], 0)[key]
        assert shared[key].count == 3000
        for name in ("aa", "bb", "ab", "tau_powers"):
            np.testing.assert_allclose(
                getattr(shared[key], name), getattr(alone, name), rtol=1e-12
            )


def test_moments_round_trip(norms, tmp_path):
    save_moments(norms[CLOSURE], tmp_path / "x.npz")
    back = load_moments(tmp_path / "x.npz")
    for name in ("aa", "bb", "ab", "tau_powers", "weight_sum", "count", "tau_max"):
        np.testing.assert_array_equal(
            getattr(back, name), getattr(norms[CLOSURE], name)
        )
    np.savez(tmp_path / "old.npz", tau=np.ones(3))
    with pytest.raises(ValueError, match="earlier version"):
        load_moments(tmp_path / "old.npz")


def _record(toy, closure, other, parts):
    def fit_result(values):
        return {
            "values": values,
            "errors": dict.fromkeys(values, 2.0),
            "valid": True,
            "fval": 0,
            "nfcn": 1,
        }

    return {
        "toy": toy,
        "results": {CLOSURE: fit_result(closure), "other": fit_result(other)},
        "closure_parts": [fit_result(p) for p in parts],
    }


def test_summary_statistics():
    truth = dict.fromkeys(TRUTH, 0.0)
    names = list(TRUTH)
    records = []
    for toy, c in enumerate((-1.0, 1.0, 3.0)):
        closure = dict.fromkeys(names, c)
        other = dict.fromkeys(names, c + 1.0)  # shift of 1e-3 = 0.5 sigma
        parts = [dict.fromkeys(names, c + 0.2), dict.fromkeys(names, c - 0.2)]
        records.append(_record(toy, closure, other, parts))
    summary = summarise_records(records, truth)
    pulls = summary["closure_pulls"]["x_cp_mix"]
    assert pulls["mean"] == pytest.approx(0.5)  # mean of (-1, 1, 3)/2
    assert pulls["std"] == pytest.approx(1.0)
    shift = summary["shifts"]["other"]["y_cp_mix"]
    assert shift["mean"] == pytest.approx(0.5)
    assert shift["mean_1e3"] == pytest.approx(1.0)
    assert shift["error"] == pytest.approx(0.0)
    # parts shifted by +-0.1 sigma: sd of the two part means 0.141, / sqrt(2) = 0.1
    offset = norm_offset(records)
    assert offset["parts"] == 2
    assert offset["dx_mix"] == pytest.approx(0.1)


@pytest.mark.slow
def test_run_study_and_resume(tmp_path):
    config = make_config()
    summary = run_study(config, tmp_path, workers=2, log=lambda m: None)
    for name in (
        "toys.jsonl",
        "summary.json",
        "config.json",
        "shifts.png",
        "closure_pulls.png",
    ):
        assert (tmp_path / name).exists()
    assert sorted(p.stem for p in (tmp_path / "norm").glob("*.npz")) == sorted(
        [s.name for s in config.scenarios] + [f"{CLOSURE}.part0", f"{CLOSURE}.part1"]
    )
    records = read_records(tmp_path)
    assert [r["toy"] for r in records] == [0, 1]
    assert set(records[0]["results"]) == {s.name for s in config.scenarios}
    assert len(records[0]["closure_parts"]) == 2
    assert summary["n_toys"] == 2
    assert set(summary["shifts"]) == {s.name for s in config.scenarios} - {CLOSURE}
    # resume: more toys, the first two are not redone
    config.n_toys = 3
    lines = []
    run_study(config, tmp_path, workers=1, log=lines.append)
    assert "1 toys to run, 2 already done" in lines
    assert [r["toy"] for r in read_records(tmp_path)] == [0, 1, 2]
    assert json.loads((tmp_path / "summary.json").read_text())["n_toys"] == 3
