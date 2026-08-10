# -*- coding: utf-8 -*-
"""Correctness tests for the vectorized geometry/orbit helpers in build_global_tensor.py."""
from __future__ import annotations

import numpy as np
import pytest

import build_global_tensor as bgt
from helper import GeoMath, KeplerianPropagator


def _random_coords(n: int = 2000, seed: int = 0) -> tuple:
    rng = np.random.RandomState(seed)
    lat1 = rng.uniform(-80, 80, n)
    lon1 = rng.uniform(-180, 180, n)
    alt1 = rng.choice([0.0, 15.0, 20.0], n)
    lat2 = rng.uniform(-80, 80, n)
    lon2 = rng.uniform(-180, 180, n)
    alt2 = rng.uniform(200.0, 36000.0, n)
    return lat1, lon1, alt1, lat2, lon2, alt2


def test_vectorized_los_matches_scalar():
    lat1, lon1, alt1, lat2, lon2, alt2 = _random_coords()
    vec = bgt._vectorized_los(lat1, lon1, alt1, lat2, lon2, alt2)
    scal = np.array(
        [
            GeoMath.check_line_of_sight(lat1[i], lon1[i], alt1[i], lat2[i], lon2[i], alt2[i])
            for i in range(len(lat1))
        ]
    )
    np.testing.assert_array_equal(vec, scal)


def test_vectorized_distance_matches_scalar():
    lat1, lon1, alt1, lat2, lon2, alt2 = _random_coords()
    vec = bgt._vectorized_distance(lat1, lon1, alt1, lat2, lon2, alt2)
    scal = np.array(
        [
            GeoMath.calculate_3d_distance(lat1[i], lon1[i], alt1[i], lat2[i], lon2[i], alt2[i])
            for i in range(len(lat1))
        ]
    )
    np.testing.assert_allclose(vec, scal, rtol=1e-12, atol=1e-9)


def test_vectorized_elevation_matches_scalar():
    rng = np.random.RandomState(1)
    n = 2000
    obs_lat = rng.uniform(-70, 70, n)
    obs_lon = rng.uniform(-180, 180, n)
    obs_alt = rng.choice([0.0, 20.0], n)
    tgt_lat = rng.uniform(-70, 70, n)
    tgt_lon = rng.uniform(-180, 180, n)
    tgt_alt = rng.uniform(0.0, 36000.0, n)
    vec = bgt._vectorized_elevation_deg(obs_lat, obs_lon, obs_alt, tgt_lat, tgt_lon, tgt_alt)
    scal = np.array(
        [
            GeoMath.elevation_angle_deg(obs_lat[i], obs_lon[i], obs_alt[i], tgt_lat[i], tgt_lon[i], tgt_alt[i])
            for i in range(n)
        ]
    )
    np.testing.assert_allclose(vec, scal, rtol=1e-12, atol=1e-9)


def test_vectorized_sat_positions_match_propagator():
    cfg = {"alt_km": 1200.0, "inclination": 60.0, "init_lon": 30.0, "init_lat": 25.0}
    a = 6371.0 + cfg["alt_km"]
    inc = cfg["inclination"]
    val = max(-1.0, min(1.0, np.sin(np.radians(cfg["init_lat"])) / np.sin(np.radians(inc))))
    prop = KeplerianPropagator(a, 0.0, inc, cfg["init_lon"], 0.0, float(np.degrees(np.arcsin(val))))

    t = np.array([0.0, 61.0, 3600.0, 86399.0, 525600.0 * 60.0])
    lat, lon, alt = bgt._vectorized_sat_positions(cfg, t)
    for i, tt in enumerate(t):
        s_lat, s_lon, s_alt = prop.get_position_at_time(tt)
        assert abs(lat[i] - s_lat) < 1e-6
        assert abs(lon[i] - s_lon) < 1e-6
        assert abs(alt[i] - s_alt) < 1e-6


def test_compute_link_timeline_return_shape(tmp_path, monkeypatch):
    """compute_link_timeline returns (link_id, kmax) and skips impossible links."""
    ctx = {
        "T_link": 1440,
        "N_gs": 2,
        "N_hap": 1,
        "gs_coords": np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0]], dtype=np.float32),
        "hap_pos": np.zeros((1440, 1, 3), dtype=np.float32),
        "sat_coords_full": np.zeros((1440, 1, 3), dtype=np.float32),
        "T_HAP": 1440,
        "min_elev_deg": 5.0,
    }
    monkeypatch.setattr(bgt, "_WORKER_CONTEXT", ctx)
    link_id, kmax = bgt.compute_link_timeline((0, 0, 1, "HAP-GS", None, None, False))
    assert link_id == 0
    assert kmax.shape == (1440,)
    # no weather (None) -> kmax stays zero (no valid visible steps with coords at origin)
    assert kmax.dtype == np.float32
