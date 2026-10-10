"""Small analytical checks for the v7 bounded-state and finite-key paths."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools.channel_v7.model import RateModelV7, bounded_collection, match_hv, state_gains
from tools.dataset_v2.finite_key import (
    FiniteKeyParams, QKDBlockAccumulator, asymptotic_key_from_stats, key_length_gains,
)


def test_bounded_collection_beta_matches_target_moments():
    mean = np.array([0.05, 0.5, 0.95, 1.0])
    si = np.array([0.2, 0.2, 0.0046, 0.0])
    nodes, weights, tail, actual_mean, actual_second = bounded_collection(mean, si, order=256)
    assert nodes.shape == weights.shape == (256, 4)
    assert np.all((nodes >= 0.0) & (nodes <= 1.0))
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(tail, 0.0)
    np.testing.assert_allclose(actual_mean, mean)
    np.testing.assert_allclose(actual_second, mean**2 * (1.0 + si))
    np.testing.assert_allclose(np.sum(weights * nodes, axis=0), mean, rtol=3e-5, atol=2e-7)
    np.testing.assert_allclose(
        np.sum(weights * nodes**2, axis=0), mean**2 * (1.0 + si),
        rtol=5e-5, atol=2e-7,
    )


def test_bounded_collection_physical_boundary_is_endpoint_bernoulli():
    mean = np.array([0.4])
    si = np.array([(1.0 - mean[0]) / mean[0]])
    nodes, weights, tail, actual_mean, actual_second = bounded_collection(mean, si, order=8)
    assert set(nodes[:, 0]) <= {0.0, 1.0}
    assert weights[0, 0] == pytest.approx(0.6)
    assert weights[1, 0] == pytest.approx(0.4)
    assert weights[2:, 0].sum() == pytest.approx(0.0)
    assert actual_mean[0] == pytest.approx(0.4)
    assert actual_second[0] == pytest.approx(0.4)


def test_bounded_collection_rejects_unphysical_inputs():
    with pytest.raises(ValueError, match="mean"):
        bounded_collection(1.01, 0.1)
    with pytest.raises(ValueError, match="scintillation"):
        bounded_collection(0.5, -0.1)
    with pytest.raises(ValueError, match="bounded-moment"):
        bounded_collection(0.95, 0.1)
    with pytest.raises(ValueError, match="quadrature order"):
        bounded_collection(0.5, 0.1, order=1)


def test_state_gains_are_probability_weighted_over_optical_states():
    fk = FiniteKeyParams()
    eta = np.array([[0.1], [0.6]])
    y0 = np.array([[1e-7], [2e-7]])
    weights = np.array([0.25, 0.75])
    gains, errors = state_gains(eta, y0, weights, fk)
    assert gains.shape == (1, 3)
    assert errors.shape == gains.shape
    assert np.all((errors >= 0.0) & (errors <= gains))
    expected = 0.25 * (y0[0, 0] + (1 - y0[0, 0]) * -np.expm1(-0.1 * np.asarray(fk.mu)))
    expected += 0.75 * (y0[1, 0] + (1 - y0[1, 0]) * -np.expm1(-0.6 * np.asarray(fk.mu)))
    np.testing.assert_allclose(gains[0], expected)


def test_activation_rate_recomputes_finite_key_length_at_scaled_pulse_count():
    fk = FiniteKeyParams()
    eta = np.array([0.1])
    y0 = np.array([1e-7])
    gains, errors = state_gains(eta, y0, np.ones(1), fk)
    n = fk.n_pulses
    full = float(key_length_gains(gains, errors, n, fk))
    first = float(key_length_gains(gains, errors, 0.5 * n, fk))
    assert full > 0.0
    assert 0.0 <= first < full
    assert first != pytest.approx(0.5 * full, rel=1e-4)


def test_most_cn2_is_matched_to_hv_and_reports_background_floor():
    cn_ref = np.array([1e-16, 1e-14])
    A, clipped = match_hv(cn_ref, z=10.0, wind=21.0, return_clipped=True)
    assert clipped.tolist() == [True, False]
    assert A[0] == 0.0
    background = (0.00594 * (21.0 / 27.0) ** 2 * (1e-5 * 10.0) ** 10
                  * np.exp(-10.0 / 1000.0) + 2.7e-16 * np.exp(-10.0 / 1500.0))
    assert A[1] == pytest.approx((cn_ref[1] - background) * np.exp(0.1))


def test_finite_key_eps_allocation_matches_spec_s5_2():
    """Spec §5.2: eps_{sec,b} = eps_sec_tot / N_max, eps_{cor,b} = eps_cor_tot / N_max.

    The fixed per-block pair (both totals None) must stay bit-identical, and the
    allocated per-block epsilons must be strictly tighter, lowering ell.
    """
    fixed = FiniteKeyParams()
    assert fixed.per_block_eps() == (fixed.eps_sec, fixed.eps_cor)
    assert fixed.effective() is fixed                      # no allocation -> same object, no reallocation

    n_max = 1000
    allocated = FiniteKeyParams(eps_sec_total=1e-9, eps_cor_total=1e-15, n_max_blocks=n_max)
    es, ec = allocated.per_block_eps()
    assert es == pytest.approx(1e-9 / n_max)
    assert ec == pytest.approx(1e-15 / n_max)
    # effective() bakes the allocation into the per-block fields the bounds read.
    eff = allocated.effective()
    assert eff.eps_sec == pytest.approx(es) and eff.eps_cor == pytest.approx(ec)
    # A stricter security budget can only reduce the key length.
    eta, y0 = 10 ** -3.0, 5e-7
    gains, errors = state_gains(np.array([eta]), np.array([y0]), np.ones(1), fixed)
    ell_fixed = float(key_length_gains(gains, errors, fixed.n_pulses, fixed))
    ell_alloc = float(key_length_gains(gains, errors, allocated.n_pulses, allocated))
    assert 0.0 <= ell_alloc <= ell_fixed


def test_qkd_block_accumulator_releases_once_on_close():
    fk = FiniteKeyParams()
    gains = np.array([2e-3, 1e-3, 5e-7])
    errors = np.array([3e-5, 2e-5, 2.5e-7])
    acc = QKDBlockAccumulator(fk)
    acc.add_slot(gains, errors, fk.n_pulses)
    acc.add_slot(gains * 0.8, errors * 0.9, 0.5 * fk.n_pulses)
    g, e, n = acc.aggregate()
    expected = float(key_length_gains(g, e, n, fk))
    assert acc.is_open
    released = acc.close_finite_key()
    assert released == pytest.approx(expected)
    assert released >= 0.0
    assert not acc.is_open
    assert acc.close_finite_key() == 0.0


def test_qkd_block_accumulator_asymptotic_release_uses_block_aggregate():
    fk = FiniteKeyParams()
    gains1 = np.array([2.0e-3, 9.0e-4, 5.0e-7])
    errors1 = np.array([3.0e-5, 1.5e-5, 2.5e-7])
    gains2 = np.array([1.2e-3, 5.5e-4, 5.0e-7])
    errors2 = np.array([2.0e-5, 1.0e-5, 2.5e-7])
    n1, n2 = fk.n_pulses, 0.5 * fk.n_pulses
    y11, y12 = 2.2e-3, 1.3e-3
    acc = QKDBlockAccumulator(fk)
    acc.add_slot(gains1, errors1, n1, y1=y11)
    acc.add_slot(gains2, errors2, n2, y1=y12)
    g, e, n = acc.aggregate()
    y1 = (n1 * y11 + n2 * y12) / n
    expected = asymptotic_key_from_stats(g, e, y1, n, fk)
    released = acc.close_asymptotic()
    assert released == pytest.approx(expected)
    assert released > 0.0
    assert not acc.is_open


def _station_cloud_state(cloud_on, *, capture_time_s=5.0, n_sub=4):
    """Minimal one-slot HAP-GS state for the v7.5 station-cloud path."""
    G = 1
    return dict(
        cfg=SimpleNamespace(
            hap_mobility="static", domain_policy="mark", protocol="finite_key",
            cloud_model="station_markov", capture_time_s=capture_time_s,
            intra_slot_samples=n_sub, alpha_min_deg=0.0,
            tau_cirrus=0.0, hap_alt_km=20.0,
        ),
        fk=FiniteKeyParams(),
        model=RateModelV7(),
        G=G,
        sat_cfgs=[],
        link_u=np.array([0]),
        link_v=np.array([G]),
        link_type=np.array(["HAP-GS"]),
        gs_lat=np.array([0.0]), gs_lon=np.array([0.0]),
        cloud_state=np.array([[int(bool(cloud_on))]], dtype=np.uint8),
        c_hi=np.zeros((G, 8760)),
        a_cn2=np.full((G, 8760), 1e-14),
        aod=np.full((G, 8760), 0.1),
    )


def test_station_cloud_state_gates_whole_slot():
    from tools.channel_v7.dataset import compute_block

    _, rates_blocked, first_blocked, geometry_blocked, _, _, blocked, *_ = compute_block(
        _station_cloud_state(False), 0, 1, _flat_hap_gs_geometry, _flat_sat_positions)
    _, rates_clear, first_clear, geometry_clear, _, _, _, *_ = compute_block(
        _station_cloud_state(True), 0, 1, _flat_hap_gs_geometry, _flat_sat_positions)

    # Dataset rates are conditional on low/mid cloud being clear; the realized
    # C_g(t) is deliberately hidden until the environment executes the action.
    np.testing.assert_allclose(rates_blocked, rates_clear)
    np.testing.assert_allclose(first_blocked, first_clear)
    np.testing.assert_array_equal(geometry_blocked, geometry_clear)
    assert geometry_clear[0, 0] == 1
    assert rates_clear[0, 0] > 0.0
    assert 0.0 < first_clear[0, 0] <= rates_clear[0, 0]
    assert blocked[0] == 1


def test_station_cloud_draw_is_independent_of_geometry_substeps():
    from tools.channel_v7.dataset import generate_station_cloud_state

    c_lm = np.full((2, 24), 0.35)
    c1, rho1 = generate_station_cloud_state(c_lm, 120, seed=7, tau_cld_s=1800.0)
    c2, rho2 = generate_station_cloud_state(c_lm, 120, seed=7, tau_cld_s=1800.0)
    np.testing.assert_array_equal(c1, c2)
    assert rho1 == pytest.approx(rho2)


def test_capture_time_is_bounded_and_changes_first_window():
    from tools.channel_v7.dataset import compute_block

    full = _station_cloud_state(True, capture_time_s=0.0)
    _, rates0, first0, *_ = compute_block(
        full, 0, 1, _flat_hap_gs_geometry, _flat_sat_positions)
    delayed = _station_cloud_state(True, capture_time_s=5.0)
    _, rates5, first5, *_ = compute_block(
        delayed, 0, 1, _flat_hap_gs_geometry, _flat_sat_positions)
    np.testing.assert_allclose(rates0, rates5)
    np.testing.assert_allclose(first0, rates0)
    assert np.all(first5 <= rates5 + 1e-9)

    bad = _station_cloud_state(True, capture_time_s=61.0)
    with pytest.raises(ValueError, match="capture_time_s"):
        compute_block(bad, 0, 1, _flat_hap_gs_geometry, _flat_sat_positions)


def _flat_hap_gs_geometry(c1, c2):
    """One HAP-GS pair at zenith, always visible (dist km, elev deg, los, graze)."""
    la1, _lo1, _al1 = c1
    la2, _lo2, _al2 = c2
    shape = np.broadcast(np.asarray(la1), np.asarray(la2)).shape
    return (np.full(shape, 20.0), np.full(shape, 90.0),
            np.ones(shape, dtype=bool), np.full(shape, 20.0))


def _flat_sat_positions(cfgs, t):
    t = np.asarray(t)
    return np.zeros((0, t.size)), np.zeros((0, t.size)), np.zeros((0, t.size))
