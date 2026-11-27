"""Phase 5: the LHCb-like acceptance emulator."""

from pathlib import Path

import numpy as np
import pytest
from aman_examples import d02kkpipi
from tensorwaves.data.rng import NumpyUniformRNG

from d02hhhh_toys.acceptance import (
    HLT1,
    AcceptanceConfig,
    BoxLine,
    ChargeAsymmetry,
    LHCbAcceptance,
    OneTrackLine,
    TwoTrackLine,
    boost,
    make_line,
    random_rotations,
    sample_pt,
)
from d02hhhh_toys.model import cp_conjugate, from_sample

CONFIGS = Path(__file__).parents[1] / "configs" / "acceptance"
LINE_A = BoxLine("box_one_track", 1, 1.0, 7.4)
LINE_B = BoxLine("box_two_track", 2, 0.5, 4.0)


@pytest.fixture(scope="module")
def events():
    model = d02kkpipi.formulate_model()
    sample = d02kkpipi.phase_space_generator(model).generate(
        20_000, NumpyUniformRNG(seed=1)
    )
    return from_sample(sample)


def mass2(p):
    return p[..., 3] ** 2 - np.sum(p[..., :3] ** 2, axis=-1)


def test_rotations_are_orthogonal():
    r = random_rotations(1000, np.random.default_rng(1))
    np.testing.assert_allclose(
        r @ np.swapaxes(r, 1, 2), np.broadcast_to(np.eye(3), r.shape), atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.det(r), 1)
    # uniform: the rotated z axis has mean 0 and <cos^2> = 1/3
    z = r[:, :, 2]
    assert np.all(np.abs(z.mean(axis=0)) < 0.1)
    assert abs(np.mean(z[:, 2] ** 2) - 1 / 3) < 0.03


def test_boost_preserves_masses_and_adds_up(events):
    rng = np.random.default_rng(2)
    beta = rng.uniform(-0.5, 0.5, size=(len(events), 3))
    lab = boost(events, beta)
    np.testing.assert_allclose(mass2(lab), mass2(events), atol=1e-10)
    total = lab.sum(axis=1)
    gamma = 1 / np.sqrt(1 - np.sum(beta**2, axis=1))
    np.testing.assert_allclose(
        total[:, :3], (gamma * d02kkpipi_mass())[:, None] * beta, atol=1e-10
    )


def d02kkpipi_mass():
    return 1.86484


def test_pt_spectrum():
    production = AcceptanceConfig().production
    pt = sample_pt(production, 400_000, np.random.default_rng(3))
    assert pt.min() >= 0
    assert pt.max() <= production.pt_max
    grid = np.linspace(0, production.pt_max, 200_001)
    density = grid / (1 + (grid / production.pt_scale) ** 2) ** production.pt_power
    mean = np.trapezoid(grid * density, grid) / np.trapezoid(density, grid)
    assert abs(pt.mean() - mean) < 4 * pt.std() / np.sqrt(len(pt))


def test_flight_and_impact_parameter(events):
    acc = LHCbAcceptance()
    tau = np.full(len(events), 1.5)
    lab = acc.lab_event(events, tau, np.random.default_rng(4))
    # L = (p/m) c tau_D0 tau along the D0 momentum
    p_d = lab.d_momentum[:, :3]
    np.testing.assert_allclose(lab.flight, p_d / 1.86484 * 122.9 * 1.5)
    # IP of each daughter: distance of its line from the primary vertex (origin)
    u = lab.momenta[..., :3] / lab.p[..., None]
    along = np.sum(lab.flight[:, None, :] * u, axis=-1)
    closest = lab.flight[:, None, :] - along[..., None] * u
    np.testing.assert_allclose(
        lab.ip_true, np.linalg.norm(closest, axis=-1), rtol=1e-9, atol=1e-9
    )
    # daughters add up to the D0
    np.testing.assert_allclose(lab.momenta.sum(axis=1), lab.d_momentum, atol=1e-9)
    # no flight, no impact parameter
    lab0 = acc.lab_event(events, np.zeros(len(events)), np.random.default_rng(4))
    np.testing.assert_array_equal(lab0.ip_true, 0)


def test_ip_resolution_and_chi2(events):
    acc = LHCbAcceptance()
    lab = acc.lab_event(events, np.zeros(len(events)), np.random.default_rng(5))
    np.testing.assert_allclose(lab.ip_sigma, 12 + 24 / lab.pt)
    # with IP = 0, IPchi2 is chi2 with one degree of freedom
    assert abs(lab.ipchi2.mean() - 1) < 0.02
    assert abs(np.mean(lab.ipchi2 > 4) - 0.0455) < 0.004


def test_tracking_efficiency(events):
    config = AcceptanceConfig().with_overrides(
        {
            "tracks.p_min": 0.0,
            "tracks.pt_min": 0.0,
            "tracks.eta_min": -99.0,
            "tracks.eta_max": 99.0,
        }
    )
    lab = LHCbAcceptance(config).lab_event(
        events, np.ones(len(events)), np.random.default_rng(6)
    )
    p, found = lab.p.ravel(), lab.reconstructed.ravel()
    high = p > 20
    assert (
        abs(found[high].mean() - 0.97 * (1 - np.exp(-np.mean(p[high] - 1.5) / 1.5)))
        < 0.01
    )
    assert not found[p < 1.5].any()


def test_hlt1_line_counting():
    lab = LHCbAcceptance().lab_event(
        np.zeros((0, 4, 4)), np.zeros(0), np.random.default_rng(0)
    )
    # synthetic daughters: (pt, ipchi2) per event
    lab.pt = np.array(
        [
            [1.5, 0.3, 0.3, 0.3],
            [0.6, 0.7, 0.3, 0.3],
            [0.6, 0.3, 0.3, 0.3],
            [1.5, 0.3, 0.3, 0.3],
        ]
    )
    lab.ipchi2 = np.array(
        [[9.0, 0, 0, 0], [5.0, 5.0, 0, 0], [9.0, 0, 0, 0], [5.0, 0, 0, 0]]
    )
    lab.reconstructed = np.ones((4, 4), dtype=bool)
    acc = LHCbAcceptance(AcceptanceConfig(hlt1=HLT1([LINE_A, LINE_B])))
    np.testing.assert_array_equal(
        acc.hlt1_lines(lab, np.random.default_rng(0)),
        [[True, False], [False, True], [False, False], [False, False]],
    )
    np.testing.assert_array_equal(
        acc.hlt1(lab, np.random.default_rng(0)), [True, True, False, False]
    )
    lab.reconstructed[0, 0] = False  # an unreconstructed track does not trigger
    assert not acc.hlt1_lines(lab, np.random.default_rng(0))[0, 0]


def test_hlt1_modes(events):
    tau = np.random.default_rng(7).exponential(size=len(events))

    def lines_and_decision(hlt1, seed=8):
        acc = LHCbAcceptance(AcceptanceConfig(hlt1=hlt1))
        rng = np.random.default_rng(seed)
        lab = acc.lab_event(events, tau, rng)
        return acc.hlt1_lines(lab, np.random.default_rng(seed + 1)), acc.hlt1(
            lab, np.random.default_rng(seed + 1)
        )

    lines, either = lines_and_decision(HLT1([LINE_A, LINE_B], "any"))
    np.testing.assert_array_equal(either, lines[:, 0] | lines[:, 1])
    _, only_a = lines_and_decision(HLT1([LINE_A]))
    np.testing.assert_array_equal(only_a, lines[:, 0])  # same random numbers
    lines, mixed = lines_and_decision(HLT1([LINE_A, LINE_B], "mixture", [0.3, 0.7]))
    # a mixture passes only events that fire one line, at the rate of the mixture
    assert np.all(mixed <= (lines[:, 0] | lines[:, 1]))
    expected = 0.3 * lines[:, 0].mean() + 0.7 * lines[:, 1].mean()
    assert abs(mixed.mean() - expected) < 4 * np.sqrt(expected / len(events))


def test_hlt1_validation():
    with pytest.raises(ValueError, match="fraction"):
        HLT1([LINE_A, LINE_B], "mixture")
    with pytest.raises(ValueError, match="sum to 1"):
        HLT1([LINE_A, LINE_B], "mixture", [0.5, 0.6])
    with pytest.raises(ValueError, match="mode"):
        HLT1([LINE_A], "all")


def test_configs():
    base = AcceptanceConfig.from_yaml(CONFIGS / "base.yaml")
    assert base == AcceptanceConfig()
    assert [line.kind for line in base.hlt1.lines] == ["two_track"]
    assert base.decay_vertex.rho_max_um == 3500.0
    either = AcceptanceConfig.from_yaml(CONFIGS / "hlt1_one_or_two_track.yaml")
    assert either.hlt1.lines == [OneTrackLine(), TwoTrackLine()]
    one = AcceptanceConfig.from_yaml(CONFIGS / "hlt1_one_track.yaml")
    assert one.hlt1.lines == [OneTrackLine()]
    two = AcceptanceConfig.from_yaml(CONFIGS / "hlt1_two_track.yaml")
    assert two.hlt1.lines == [TwoTrackLine()]
    assert two.ip_resolution == base.ip_resolution
    mixture = AcceptanceConfig.from_yaml(CONFIGS / "hlt1_mixture.yaml")
    assert mixture.hlt1.mode == "mixture"
    assert mixture.hlt1.lines == either.hlt1.lines
    assert AcceptanceConfig.from_yaml(CONFIGS / "hlt1_box_a.yaml").hlt1.lines == [
        LINE_A
    ]
    assert AcceptanceConfig.from_yaml(CONFIGS / "hlt1_box_b.yaml").hlt1.lines == [
        LINE_B
    ]
    shifted = base.with_overrides(
        {"ip_resolution.scale": 1.05, "tracking": {"width": 3.0}}
    )
    assert shifted.ip_resolution.scale == 1.05
    assert shifted.tracking.width == 3.0
    assert shifted.tracking.plateau == base.tracking.plateau
    assert shifted.hlt1 == base.hlt1  # lines survive the dict round trip
    with pytest.raises(KeyError):
        base.with_overrides({"ip_resolution.sigma": 1})
    with pytest.raises(KeyError):
        AcceptanceConfig.from_dict({"tracks": {"p_max": 3}})
    with pytest.raises(ValueError, match="kind"):
        make_line({"kind": "topo", "name": "x"})


def test_one_track_boundary():
    """The IPχ² threshold diverges at the wall, falls with pT and tends to the floor."""
    line = OneTrackLine(pt_wall=1.5, k=2.0, ipchi2_floor=8.0)
    pt = np.array([1.0, 1.5, 1.6, 2.0, 3.0, 5.0, 10.0, 1000.0])
    threshold = line.ipchi2_threshold(pt)
    assert np.all(np.isinf(threshold[:2]))
    assert np.all(np.diff(threshold[2:]) < 0)
    np.testing.assert_allclose(threshold[3], 8 * np.exp(2.0 / 0.5))
    assert threshold[-1] == pytest.approx(8.0, rel=3e-3)


def test_two_track_pair_quantities(events):
    acc = LHCbAcceptance()
    rng = np.random.default_rng(15)
    lab = acc.lab_event(events, np.full(len(events), 1.0), rng)
    q = TwoTrackLine().pair_quantities(lab, acc.config, rng)
    assert q["mcorr"].shape == (len(events), 6)
    # the corrected mass is at least the pair mass; the vertex chi2 has 1 dof
    pair = lab.momenta[:, 0] + lab.momenta[:, 1]
    m01 = np.sqrt(pair[:, 3] ** 2 - np.sum(pair[:, :3] ** 2, axis=1))
    assert np.all(q["mcorr"][:, 0] >= m01 - 1e-9)
    assert abs(q["vertex_chi2"].mean() - 1) < 0.02
    np.testing.assert_allclose(q["sum_pt"][:, 0], lab.pt[:, 0] + lab.pt[:, 1])
    # the flight direction is the D0 direction, in 2 < eta < 5
    assert np.all((q["flight_eta"] > 2 - 1e-9) & (q["flight_eta"] < 5 + 1e-9))
    # without flight, FDchi2 has 1 dof and the score cut removes almost everything
    lab0 = acc.lab_event(events, np.zeros(len(events)), rng)
    assert TwoTrackLine().decide(lab0, acc.config, rng).mean() < 0.005


def test_two_track_line_is_most_efficient_for_soft_charm(events):
    """At large tau the two-track line keeps most events, the one-track line few."""
    tau = np.full(len(events), 4.0)
    rates = {}
    for line in (OneTrackLine(), TwoTrackLine()):
        acc = LHCbAcceptance(AcceptanceConfig(hlt1=HLT1([line])))
        rng = np.random.default_rng(16)
        lab = acc.lab_event(events, tau, rng)
        rates[line.kind] = acc.hlt1(lab, rng)[acc.offline(lab)].mean()
    assert rates["two_track"] > 0.8
    assert rates["one_track"] < 0.3


def test_acceptance_turns_on_with_decay_time(events):
    acc = LHCbAcceptance()
    rng = np.random.default_rng(9)
    short = acc.accept(events, np.full(len(events), 0.05), rng).mean()
    long = acc.accept(events, np.full(len(events), 2.0), rng).mean()
    assert short < 0.5 * long
    assert 0.01 < long < 0.9  # about 3.5% with the proposed toy values


def test_reproducible(events):
    tau = np.ones(len(events))
    a = LHCbAcceptance().accept(events, tau, np.random.default_rng(10))
    b = LHCbAcceptance().accept(events, tau, np.random.default_rng(10))
    np.testing.assert_array_equal(a, b)


def test_cp_symmetric(events):
    """The acceptance of the CP-conjugate of each event is the same, on average."""
    n_copies = 400
    picked = events[:50]
    acc = LHCbAcceptance()
    tau = np.full(len(picked) * n_copies, 1.0)
    e = np.repeat(picked, n_copies, axis=0)
    eff = (
        acc.accept(e, tau, np.random.default_rng(11)).reshape(-1, n_copies).mean(axis=1)
    )
    eff_bar = (
        acc.accept(cp_conjugate(e), tau, np.random.default_rng(12))
        .reshape(-1, n_copies)
        .mean(axis=1)
    )
    error = np.sqrt((eff * (1 - eff) + eff_bar * (1 - eff_bar)) / n_copies) + 1e-3
    pulls = (eff - eff_bar) / error
    assert abs(pulls.mean()) < 4 / np.sqrt(len(pulls))
    assert pulls.std() < 1.5


def test_offline_ipchi2_track_count(events):
    """Offline: at least n_tracks daughters above the IPchi2 cut; 0 is no cut, 4 is all."""
    tau = np.random.default_rng(13).exponential(size=len(events))
    base = AcceptanceConfig()
    assert base.offline.n_tracks == 0
    lab = LHCbAcceptance(base).lab_event(events, tau, np.random.default_rng(14))
    above = lab.reconstructed & (lab.ipchi2 > 4)
    all_reco = lab.reconstructed.all(axis=1)
    for n in range(5):
        acc = LHCbAcceptance(base.with_overrides({"offline.n_tracks": n}))
        np.testing.assert_array_equal(
            acc.offline(lab), all_reco & (above.sum(axis=1) >= n)
        )
    np.testing.assert_array_equal(LHCbAcceptance(base).offline(lab), all_reco)
    four = LHCbAcceptance(base.with_overrides({"offline.n_tracks": 4})).offline(lab)
    np.testing.assert_array_equal(four, above.all(axis=1))
    with pytest.raises(ValueError, match="n_tracks"):
        base.with_overrides({"offline.n_tracks": 5})


def test_decay_vertex_inside_rf_foil(events):
    """The decay vertex must be within 3.5 mm of the beam line: rho = (pT/m) c tau_D0 tau."""
    acc = LHCbAcceptance()
    tau = np.random.default_rng(17).exponential(size=len(events)) * 5
    lab = acc.lab_event(events, tau, np.random.default_rng(18))
    rho = np.hypot(lab.flight[:, 0], lab.flight[:, 1])
    pt_d = np.hypot(lab.d_momentum[:, 0], lab.d_momentum[:, 1])
    np.testing.assert_allclose(rho, pt_d / 1.86484 * 122.9 * tau)
    np.testing.assert_array_equal(acc.inside_foil(lab), rho < 3500)
    assert 0 < np.mean(~acc.inside_foil(lab)) < 0.5
    no_cut = LHCbAcceptance(
        acc.config.with_overrides({"decay_vertex.rho_max_um": None})
    )
    assert no_cut.inside_foil(lab).all()
    # accept() applies it
    far = np.full(len(events), 200.0)  # rho > 3.5 mm unless pT(D0) < 0.27 GeV
    lab_far = acc.lab_event(events, far, np.random.default_rng(19))
    expected = acc.inside_foil(lab_far).mean()
    assert acc.accept(events, far, np.random.default_rng(19)).mean() <= expected


def test_charge_asymmetry_factors():
    asym = ChargeAsymmetry(kaon_a=0.005, kaon_b_gev=0.05, pion_b_gev=0.005)
    p = np.array([[5.0, 5.0, 10.0, 10.0]])
    np.testing.assert_allclose(
        asym.factors(p), [[1.015, 0.985, 1.0005, 0.9995]], rtol=0, atol=1e-15
    )
    # one track of each charge per species: a constant A gives the same factor for
    # an event and its CP conjugate (K+ <-> K-, pi+ <-> pi- momenta); b does not
    momenta = np.array([[4.0, 12.0, 3.0, 30.0]])
    conjugate = momenta[:, [1, 0, 3, 2]]
    constant = ChargeAsymmetry(kaon_a=0.02, pion_a=0.01)
    assert np.prod(constant.factors(momenta)) == pytest.approx(
        np.prod(constant.factors(conjugate)), rel=1e-15
    )
    assert np.prod(asym.factors(momenta)) != pytest.approx(
        np.prod(asym.factors(conjugate)), rel=1e-6
    )


def test_kaon_detection_asymmetry(events):
    """K+ is found more often than K-, by A_K(p) = a + b/p."""
    config = AcceptanceConfig().with_overrides(
        {"charge_asymmetry": {"kaon_a": 0.02, "kaon_b_gev": 0.2}}
    )
    acc = LHCbAcceptance(config)
    e = np.repeat(events, 5, axis=0)
    lab = acc.lab_event(e, np.ones(len(e)), np.random.default_rng(13))
    plateau = lab.p > 10
    k_plus = lab.reconstructed[:, 0][plateau[:, 0]].mean()
    k_minus = lab.reconstructed[:, 1][plateau[:, 1]].mean()
    expected = 0.02 + 0.2 * np.mean(1 / lab.p[:, :2][plateau[:, :2]])
    measured = (k_plus - k_minus) / (k_plus + k_minus)
    assert measured == pytest.approx(expected, abs=0.006)
    pions = lab.reconstructed[:, 2][plateau[:, 2]].mean() - (
        lab.reconstructed[:, 3][plateau[:, 3]].mean()
    )
    assert abs(pions) < 0.006
