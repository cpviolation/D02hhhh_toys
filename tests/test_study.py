"""Phase 7: scenarios and the study driver."""

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import SUBSET
from d02hhhh_toys.acceptance import LHCbAcceptance
from d02hhhh_toys.normalisation import normalisation_sample
from d02hhhh_toys.study import (
    CLOSURE,
    Scenario,
    StudyConfig,
    load_rate_sample,
    norm_offset,
    read_records,
    run_study,
    save_rate_sample,
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


def test_scenario_normalisation_modes(subset_model, norms):
    config = make_config()
    full, none = norms[CLOSURE], norms["none"]
    direct = normalisation_sample(
        subset_model,
        config.n_norm,
        np.random.SeedSequence([config.seed, 1]),
        acceptance=LHCbAcceptance(config.true_acceptance()),
    )
    np.testing.assert_array_equal(full.tau, direct.tau)
    # factorised: the same decay times, permuted; the phase-space points are kept
    f = norms["factorised"]
    np.testing.assert_array_equal(np.sort(f.tau), np.sort(full.tau))
    assert not np.array_equal(f.tau, full.tau)
    np.testing.assert_array_equal(f.amplitude, full.amplitude)
    # ignore_time: the accepted phase-space points with new decay times from e^-tau
    t = norms["ignore_time"]
    np.testing.assert_array_equal(t.amplitude, full.amplitude)
    assert abs(t.tau.mean() - 1) < 4 / np.sqrt(len(t))
    assert full.tau.mean() > 1.5  # the acceptance removes short decay times
    # ignore_ps: the accepted decay times with points drawn without acceptance
    p = norms["ignore_ps"]
    np.testing.assert_array_equal(p.amplitude, none.amplitude)
    np.testing.assert_array_equal(np.sort(p.tau), np.sort(full.tau))
    assert abs(none.tau.mean() - 1) < 4 / np.sqrt(len(none))


def test_common_random_numbers(norms):
    """An override changes few of the accepted events: most decay times are shared."""
    shared = np.isin(norms["ip+5%"].tau, norms[CLOSURE].tau).mean()
    assert shared > 0.8


def test_rate_sample_round_trip(norms, tmp_path):
    save_rate_sample(norms[CLOSURE], tmp_path / "x.npz")
    back = load_rate_sample(tmp_path / "x.npz")
    for name in ("tau", "amplitude", "amplitude_bar", "weight"):
        np.testing.assert_array_equal(
            getattr(back, name), getattr(norms[CLOSURE], name)
        )


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
        s.name for s in config.scenarios
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
