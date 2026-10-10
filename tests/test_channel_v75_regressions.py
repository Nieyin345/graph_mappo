"""Regression coverage for the six channel/spec review findings."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.test_channel_v7 import (
    _flat_hap_gs_geometry, _flat_sat_positions, _station_cloud_state,
)
from tests.test_h5_rate_provider import _edges, _write_dataset
from qkd_rl.core.config import ConfigValidator, deep_merge, load_yaml
from qkd_rl.env.factory import load_default_config
from qkd_rl.env.request import RequestGenerator
from qkd_rl.link.rate_provider import build_rate_provider
from tools.channel_v7.dataset import compute_block
from tools.dataset_v2.build_dataset_v2 import read_stations

ROOT = Path(__file__).resolve().parents[1]


def _main_config():
    cfg = load_default_config(ROOT)
    for name in ("env_full.yaml", "env_full_demand_v2.yaml", "dataset_v7.yaml", "scenario_v7.yaml"):
        cfg = deep_merge(cfg, load_yaml(ROOT / "configs" / name))
    return cfg


def test_invisible_batch_has_zero_rates_and_complete_statistics():
    def invisible(a, b):
        distance, elevation, los, graze = _flat_hap_gs_geometry(a, b)
        return distance, elevation, np.zeros_like(los), graze

    result = compute_block(_station_cloud_state(True), 0, 1, invisible, _flat_sat_positions)
    for array in result[1:7]:
        assert np.all(np.asarray(array) == 0)
    for statistics in result[-2:]:
        for array in statistics.values():
            assert np.all(np.isfinite(array))
            assert np.all(array == 0)


def test_full_slot_acquisition_has_zero_first_statistics():
    result = compute_block(
        _station_cloud_state(True, capture_time_s=60), 0, 1,
        _flat_hap_gs_geometry, _flat_sat_positions,
    )
    assert result[1][0, 0] > 0
    assert result[2][0, 0] == 0
    for array in result[-1].values():
        assert np.all(array == 0)


@pytest.mark.parametrize("policy", ["mark", "extrapolate"])
def test_mixed_domain_reports_fraction_and_marks_both_windows(policy):
    state = _station_cloud_state(True, n_sub=4)
    state["cfg"].domain_policy = policy
    original = state["model"].optical_states

    def mixed(*args, **kwargs):
        result = original(*args, **kwargs)
        result["domain_valid"] = np.array([True, False, False, False])
        return result

    state["model"].optical_states = mixed
    result = compute_block(state, 0, 1, _flat_hap_gs_geometry, _flat_sat_positions)
    assert result[-2]["model_domain"][0, 0] == pytest.approx(0.25)
    assert result[-2]["domain_visible_substeps"][0, 0] == 4
    assert result[-2]["domain_valid_substeps"][0, 0] == 1
    if policy == "mark":
        assert np.isnan(result[1][0, 0])
        assert np.isnan(result[2][0, 0])
    else:
        assert np.isfinite(result[1][0, 0])
        assert np.isfinite(result[2][0, 0])


def test_ground_elevation_reaches_geometry_and_optics(tmp_path):
    path = tmp_path / "stations.csv"
    path.write_text("name,lat,lon,base,continent,alt_km,h_ERA5\nPlateau,30,90,1,Asia,3,2500\n")
    station = read_stations(path)[0]
    assert station["alt_km"] == 3
    assert station["h_GS_m"] == 3000
    assert station["h_ERA5_m"] == 2500
    state = _station_cloud_state(True)
    state["gs_alt_km"] = np.array([station["alt_km"]])
    original = state["model"].optical_states
    seen = []

    def optical(*args, **kwargs):
        seen.append(kwargs["h_gs_km"])
        return original(*args, **kwargs)

    def geometry(a, b):
        assert np.all(a[2] == 3)
        return _flat_hap_gs_geometry(a, b)

    state["model"].optical_states = optical
    compute_block(state, 0, 1, geometry, _flat_sat_positions)
    assert seen == [3.0, 3.0]
    assert state["model"]._ground_hv("HAP", 3).h_top == 17000
    model = state["model"]
    assert model.eta_mol(90, 20, 3) > model.eta_mol(90, 20, 0)


def test_main_causal_window_is_invariant_to_future_weather(tmp_path):
    _write_dataset(tmp_path)
    cfg = _main_config()
    ConfigValidator().validate(cfg)
    cfg["rate_provider"]["h5"]["dataset_dir"] = str(tmp_path)
    cfg["rate_provider"]["normalization"] = {"enabled": False}
    with h5py.File(tmp_path / "link_data.h5", "r+") as f:
        f.attrs["capture_time_s"] = 5.0
    provider = build_rate_provider(cfg, _edges(), seed=0)
    before = provider.get_window_blocks(7)[0].copy()
    provider.close()
    with h5py.File(tmp_path / "link_data.h5", "r+") as f:
        f["k_proxy"][8:] *= 10
    provider = build_rate_provider(cfg, _edges(), seed=0)
    try:
        after = provider.get_window_blocks(7)[0]
        assert before.shape[0] == 1
        np.testing.assert_array_equal(after, before)
    finally:
        provider.close()


def test_era5_pressure_temperature_are_corrected_before_most(tmp_path, monkeypatch):
    import tools.dataset_v2.build_dataset_v2 as builder

    station_file = tmp_path / "station.csv"
    station_file.write_text("name,lat,lon,base,continent,h_GS,h_ERA5\nPlateau,30,90,1,Asia,3000,2500\n")
    stations = read_stations(station_file)
    (tmp_path / "Plateau.csv").touch()
    n = builder.YEAR_HOURS
    weather = {"visibility": np.full(n, 23000.0)}
    era5 = {"H_up_Wm2": np.full(n, 50.), "ustar": np.full(n, 0.3),
            "t2m_K": np.full(n, 288.), "sp_Pa": np.full(n, 75000.)}
    monkeypatch.setattr(builder, "_read_csv_columns", lambda *a, **kw: weather)
    monkeypatch.setattr(builder, "_read_era5_station", lambda *a: (era5, None))
    observed = []
    original = builder.cn2_monin_obukhov

    def most(h, u, temperature, pressure, **kw):
        observed.append((temperature.copy(), pressure.copy()))
        return original(h, u, temperature, pressure, **kw)

    monkeypatch.setattr(builder, "cn2_monin_obukhov", most)
    builder.load_weather(stations, tmp_path, tmp_path, builder.ParamsV2(), False,
                         strict=True, apply_height_correction=True)
    temperature, pressure = observed[0]
    expected_temperature = 288. - 0.0065 * 500.
    expected_pressure = 75000. * np.exp(-9.81 * 500. / (287.05 * (288. + expected_temperature) / 2))
    np.testing.assert_allclose(temperature, expected_temperature)
    np.testing.assert_allclose(pressure, expected_pressure)


@pytest.mark.parametrize("flag", [
    "include_rate_future_window", "include_available_future_window",
    "include_rate_delta", "include_rate_mean", "include_rate_max",
])
def test_causal_config_rejects_future_derived_features(flag):
    cfg = _main_config()
    cfg["features"]["edge"][flag] = True
    with pytest.raises(ValueError, match="causal"):
        ConfigValidator().validate(cfg)


def test_zipf_v7_normalizes_each_hour_and_generates_integer_uniform_sizes(tmp_path):
    pairs = tmp_path / "pairs.csv"
    pairs.write_text("src,dst,rank\nA,B,1\nB,C,2\n")
    nodes = tmp_path / "nodes.csv"
    nodes.write_text("name,type,lon\nA,GS,0\nB,GS,120\nC,GS,-90\n")
    cfg = dict(_main_config()["requests"], fixed_pairs_csv=str(pairs),
               fixed_pair_count=2, node_registry_path=str(nodes),
               arrival_rate_per_second=0.1)
    gen = RequestGenerator(["A", "B", "C"], cfg, seed=7)
    rates = [gen._zipf_v7_pair_rates(t) for t in range(0, 1440, 60)]
    np.testing.assert_allclose(np.sum(rates, axis=1), 6.0)
    assert not np.allclose(rates[0], rates[12])
    requests = [r for t in range(500) for r in gen.generate(t)]
    amounts = np.array([r.amount for r in requests])
    assert len(requests) > 2500
    assert all((r.src_gs, r.dst_gs) in {("A", "B"), ("B", "C")} for r in requests)
    assert np.all((amounts >= 50000) & (amounts <= 200000))
    assert np.all(amounts == np.floor(amounts))
    assert abs(amounts.mean() - 125000) < 4000
