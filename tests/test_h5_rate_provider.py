from __future__ import annotations

import csv
import json
import math
import sys

import h5py
import numpy as np
import pytest

from qkd_rl.core.config import ConfigValidator
from qkd_rl.core.types import Edge, LinkType
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.link.rate_provider import H5RateProvider, RateNormalizer, build_rate_provider

NODES = [
    (0, "GS", "Beijing", 39.9, 116.4, 0.0),
    (1, "GS", "Shanghai", 31.2, 121.5, 0.0),
    (2, "HAP", "HAP_Test", 39.0, 116.0, 20.0),
    (3, "SAT", "Sat_Test", 0.0, 0.0, 600.0),
]
LINKS = [
    (0, 0, 2, "HAP-GS"),   # Beijing -- HAP_Test
    (1, 1, 2, "HAP-GS"),   # Shanghai -- HAP_Test
    (2, 0, 3, "SAT-GS"),   # Beijing -- Sat_Test
    (3, 2, 3, "SAT-HAP"),  # HAP_Test -- Sat_Test
]


def _edges() -> list[Edge]:
    return [
        Edge("E_Beijing__HAP_Test", "Beijing", "HAP_Test", LinkType.GS_HAP),
        Edge("E_Shanghai__HAP_Test", "Shanghai", "HAP_Test", LinkType.GS_HAP),
        Edge("E_Beijing__Sat_Test", "Beijing", "Sat_Test", LinkType.GS_SAT),
        Edge("E_HAP_Test__Sat_Test", "HAP_Test", "Sat_Test", LinkType.HAP_SAT),
    ]


def _write_dataset(tmp_path, t_steps: int = 500, write_link_registry_csv: bool = True) -> dict:
    with (tmp_path / "node_registry.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "type", "name", "lat", "lon", "alt_km"])
        writer.writerows(NODES)
    if write_link_registry_csv:
        with (tmp_path / "link_registry.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["link_id", "node_u", "node_v", "link_type"])
            writer.writerows(LINKS)
    L = len(LINKS)
    rng = np.random.RandomState(0)
    kmax = rng.uniform(0.5, 5.0, size=(t_steps, L)).astype(np.float32)
    kmax[100:150, 2] = 0.0   # Beijing-Sat outage window
    kmax[300, 0] = np.nan    # NaN sample on Beijing-HAP
    los = np.ones((t_steps, L), dtype=np.int8)
    los[100:150, 2] = 0
    distance = rng.uniform(100.0, 2000.0, size=(t_steps, L)).astype(np.float32)
    zenith = np.full((t_steps, L), np.nan, dtype=np.float32)
    with h5py.File(tmp_path / "link_data.h5", "w") as f:
        dt = np.dtype([("link_id", "i4"), ("node_u", "i4"), ("node_v", "i4"), ("link_type", "S10")])
        reg = np.array([(link_id, u, v, link_type.encode()) for link_id, u, v, link_type in LINKS], dtype=dt)
        f.create_dataset("link_registry", data=reg)
        f.create_dataset("k_max", data=kmax)
        kproxy = np.nan_to_num(kmax, nan=0.0) * 2.0 + 0.25
        kproxy_first = 0.9 * kproxy
        f.create_dataset("k_proxy", data=kproxy.astype(np.float32))
        f.create_dataset("k_proxy_first", data=kproxy_first.astype(np.float32))
        f.create_dataset("los", data=los)
        f.create_dataset("distance", data=distance)
        f.create_dataset("zenith", data=zenith)
        # Minimal v7.5 sufficient statistics for logical QKD-block tests.
        q = np.zeros((t_steps, L, 3), dtype=np.float32)
        q[..., 0] = 2.0e-3
        q[..., 1] = 9.0e-4
        q[..., 2] = 5.0e-7
        y1 = np.full((t_steps, L), 2.2e-3, dtype=np.float32)
        frac = np.ones((t_steps, L), dtype=np.float32)
        f.create_dataset("qkd_gain", data=q)
        f.create_dataset("qkd_y1", data=y1)
        f.create_dataset("qkd_usable_fraction", data=frac)
        f.create_dataset("qkd_gain_first", data=q)
        f.create_dataset("qkd_y1_first", data=y1)
        f.create_dataset("qkd_usable_fraction_first", data=0.9 * frac)
        from tools.dataset_v2.finite_key import FiniteKeyParams
        f.attrs["finite_key_params_json"] = json.dumps(FiniteKeyParams().as_dict())
        f.attrs["capture_time_s"] = 6.0
        f.attrs["theta_sec"] = 60
    return {"kmax": kmax, "kproxy": kproxy, "kproxy_first": kproxy_first, "los": los, "T": t_steps, "L": L}


def _provider(tmp_path, horizon: int = 6, min_link_rate: float = 0.001, **overrides) -> H5RateProvider:
    cfg = {"h5": {"dataset_dir": str(tmp_path)}, "rate": {"min_link_rate": min_link_rate, "unavailable_if_nan": True}}
    for key, value in overrides.items():
        cfg["h5"][key] = value
    provider = H5RateProvider(cfg, seed=0)
    provider.setup(_edges(), horizon=horizon, min_link_rate=min_link_rate)
    return provider


def test_h5_provider_maps_edges_and_returns_rates(tmp_path):
    data = _write_dataset(tmp_path)
    provider = _provider(tmp_path)
    assert provider.get_rate("E_Beijing__HAP_Test", 10) == pytest.approx(float(data["kmax"][10, 0]))
    assert provider.get_rate("E_HAP_Test__Sat_Test", 7) == pytest.approx(float(data["kmax"][7, 3]))
    assert provider.time_bounds == (0, data["T"])
    assert provider.slot_seconds == 60
    provider.close()


def test_h5_provider_can_select_v75_proxy_rate_surface(tmp_path):
    data = _write_dataset(tmp_path)
    with h5py.File(tmp_path / "link_data.h5", "r+") as f:
        f.attrs["capture_time_s"] = 5.0
    provider = _provider(
        tmp_path,
        rate_dataset_key="k_proxy",
        activation_rate_dataset_key="k_proxy_first",
    )
    edge = "E_Beijing__HAP_Test"
    assert provider.get_generation_rate(edge, 7) == pytest.approx(float(data["kproxy"][7, 0]))
    assert provider.get_rate(edge, 7) == pytest.approx(float(data["kproxy"][7, 0]))
    assert provider.get_activation_rate(edge, 7) == pytest.approx(float(data["kproxy_first"][7, 0]))
    provider.close()


def test_h5_provider_rejects_rate_stats_for_different_rate_surface(tmp_path):
    _write_dataset(tmp_path)
    stats_path = tmp_path / "rate_stats.json"
    stats_path.write_text(json.dumps({
        "rate_key": "k_max",
        "global": {"p99": 10.0},
        "per_link_type": {},
    }), encoding="utf-8")
    normalizer = RateNormalizer({"mode": "log_p99", "stats_path": str(stats_path)})
    cfg = {
        "h5": {"dataset_dir": str(tmp_path), "rate_dataset_key": "k_proxy"},
        "rate": {"min_link_rate": 0.001, "unavailable_if_nan": True},
    }
    provider = H5RateProvider(cfg, seed=0)
    with pytest.raises(ValueError, match="calibrated from 'k_max'.*reading 'k_proxy'"):
        provider.setup(_edges(), horizon=2, min_link_rate=0.001, normalizer=normalizer)
    provider.close()


def test_v75_cloud_probability_is_visible_but_realized_gate_is_hidden(tmp_path):
    data = _write_dataset(tmp_path, t_steps=400)
    with h5py.File(tmp_path / "link_data.h5", "r+") as f:
        state = np.ones((400, 2), dtype=np.uint8)
        state[7, 0] = 0
        prob = np.ones((400, 2), dtype=np.float32)
        prob[:, 0] = 0.25
        f.create_dataset("station_cloud_state", data=state)
        f.create_dataset("station_cloud_clear_probability", data=prob)
        f.create_dataset("link_ground_station", data=np.array([0, 1, 0, -1], dtype=np.int32))
    provider = _provider(tmp_path, horizon=2)
    edge = "E_Beijing__HAP_Test"
    raw = float(data["kmax"][7, 0])
    assert provider.get_generation_rate(edge, 7) == pytest.approx(raw)
    assert provider.get_rate(edge, 7) == pytest.approx(0.25 * raw)
    assert provider.get_cloud_clear_probability(edge, 7) == pytest.approx(0.25)
    assert provider.get_cloud_gate(edge, 7) == 0.0
    window = provider.get_edge_window(edge, 7)
    assert window.rates[0] == pytest.approx(0.25 * raw)
    provider.close()


def test_h5_provider_reads_v75_qkd_sufficient_stats(tmp_path):
    _write_dataset(tmp_path)
    provider = _provider(tmp_path)
    edge = "E_Beijing__HAP_Test"
    full = provider.get_qkd_slot_stats(edge, 7, activation=False)
    first = provider.get_qkd_slot_stats(edge, 7, activation=True)
    np.testing.assert_allclose(full["gains"], [2.0e-3, 9.0e-4, 5.0e-7], rtol=1e-6)
    assert full["y1"] == pytest.approx(2.2e-3, rel=1e-6)
    assert full["n_pulses"] == pytest.approx(6.0e9)
    assert first["n_pulses"] == pytest.approx(0.9 * 6.0e9)
    y0 = full["gains"][-1]
    np.testing.assert_allclose(full["errors"], 0.01 * full["gains"] + 0.49 * y0)
    provider.close()


def test_h5_provider_availability_follows_los_and_rate(tmp_path):
    _write_dataset(tmp_path)
    provider = _provider(tmp_path)
    assert provider.is_available("E_Beijing__HAP_Test", 10)
    # outage window: los=0 and rate=0
    assert not provider.is_available("E_Beijing__Sat_Test", 120)
    # NaN rate with unavailable_if_nan -> cleaned to 0 -> unavailable
    assert not provider.is_available("E_Beijing__HAP_Test", 300)
    provider.close()


def test_h5_provider_window_length_and_year_end_padding(tmp_path):
    data = _write_dataset(tmp_path, t_steps=500)
    provider = _provider(tmp_path, horizon=6)
    window = provider.get_edge_window("E_Beijing__HAP_Test", data["T"] - 2)
    assert len(window.rates) == 7
    # last two slots are beyond the dataset -> zero padded & unavailable
    assert window.rates[-1] == 0.0
    assert window.rates[-2] == 0.0
    assert window.available[-1] is False
    assert window.available[-2] is False
    provider.close()

    pad_provider = _provider(tmp_path, horizon=6, out_of_range_policy="pad_last")
    pad_window = pad_provider.get_edge_window("E_Beijing__HAP_Test", data["T"] - 2)
    assert pad_window.rates[-1] == pytest.approx(float(data["kmax"][data["T"] - 1, 0]))
    pad_provider.close()


def test_h5_window_blocks_honor_out_of_range_policy(tmp_path):
    data = _write_dataset(tmp_path, t_steps=500)

    zero_provider = _provider(tmp_path, horizon=2, out_of_range_policy="zero")
    zero_blocks = zero_provider.get_window_blocks(data["T"] - 2)
    assert zero_blocks[0][-1, 0] == 0.0
    zero_provider.close()

    pad_provider = _provider(tmp_path, horizon=2, out_of_range_policy="pad_last")
    pad_blocks = pad_provider.get_window_blocks(data["T"] - 2)
    assert pad_blocks[0][-1, 0] == pytest.approx(float(data["kmax"][data["T"] - 1, 0]))
    pad_provider.close()

    raise_provider = _provider(tmp_path, horizon=2, out_of_range_policy="raise")
    with pytest.raises(IndexError):
        raise_provider.get_window_blocks(data["T"] - 2)
    raise_provider.close()


def test_h5_provider_missing_edge_raises(tmp_path):
    _write_dataset(tmp_path)
    cfg = {"h5": {"dataset_dir": str(tmp_path)}, "rate": {"min_link_rate": 0.001}}
    provider = H5RateProvider(cfg, seed=0)
    extra_edges = _edges() + [Edge("E_Beijing__Urumqi", "Beijing", "Urumqi", LinkType.GS_HAP)]
    with pytest.raises(ValueError, match="No H5 link"):
        provider.setup(extra_edges, horizon=6, min_link_rate=0.001)
    provider.close()


def test_h5_provider_lean_file_without_los(tmp_path):
    """A compacted H5 (k_max only) must derive availability from rate>=min."""
    import h5py

    data = _write_dataset(tmp_path)
    # Rewrite as lean: keep only k_max + link_registry (drop los/distance/zenith).
    with h5py.File(tmp_path / "link_data.h5", "r") as src, h5py.File(tmp_path / "link_data_lean.h5", "w") as dst:
        dst.create_dataset("k_max", data=src["k_max"][:])
        dst.create_dataset("link_registry", data=src["link_registry"][:])
        dst.attrs["theta_sec"] = 60
    (tmp_path / "link_data.h5").unlink()
    (tmp_path / "link_data_lean.h5").replace(tmp_path / "link_data.h5")

    provider = _provider(tmp_path, min_link_rate=0.001)
    # rate=0 during outage window -> unavailable even without los
    assert not provider.is_available("E_Beijing__Sat_Test", 120)
    # nonzero rate -> available
    assert provider.is_available("E_Beijing__HAP_Test", 10)
    window = provider.get_all_edge_windows(0)
    assert window["E_Beijing__HAP_Test"].available[0] is True
    provider.close()


def test_h5_provider_link_registry_fallback_from_h5_dataset(tmp_path):
    _write_dataset(tmp_path, write_link_registry_csv=False)
    provider = _provider(tmp_path)
    assert provider.get_rate("E_Beijing__HAP_Test", 0) >= 0.0
    provider.close()


def test_full_scenario_and_env_integration_with_h5(tmp_path):
    data = _write_dataset(tmp_path, t_steps=500)
    config = load_default_config(".")
    config["scenario"]["mode"] = "full"
    config["rate_provider"]["provider"] = "h5"
    config["rate_provider"]["h5"]["dataset_dir"] = str(tmp_path)
    config["env"]["episode_steps"] = 30
    ConfigValidator().validate(config)

    env = build_env_from_config(config)
    obs = env.reset()
    assert obs.node_ids == ["Beijing", "Shanghai", "HAP_Test", "Sat_Test"]
    assert len(obs.physical_edge_ids) == data["L"]
    assert len(obs.node_features[0]) == config["features"]["dims"]["node_dim_resolved"]

    from qkd_rl.baselines.random_policy import RandomPolicy

    policy = RandomPolicy(seed=0)
    total_reward = 0.0
    for _ in range(20):
        actions, scores = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(actions, scores)
        assert not math.isnan(reward)
        total_reward += reward
    assert total_reward == total_reward  # not NaN


def test_build_rate_provider_h5_from_config(tmp_path):
    _write_dataset(tmp_path)
    config = load_default_config(".")
    config["rate_provider"]["provider"] = "h5"
    config["rate_provider"]["h5"]["dataset_dir"] = str(tmp_path)
    config["scenario"]["mode"] = "full"
    ConfigValidator().validate(config)
    from qkd_rl.data.scenario_builder import ScenarioBuilder

    scenario = ScenarioBuilder(config).build_full()
    provider = build_rate_provider(config, scenario.edges, seed=0)
    windows = provider.get_all_edge_windows(0)
    assert set(windows) == set(scenario.edge_ids)
    provider.close()


def test_h5_provider_reads_precomputed_activation_rate(tmp_path):
    data = _write_dataset(tmp_path)
    first = np.full_like(data["kmax"], 1.25)
    with h5py.File(tmp_path / "link_data.h5", "r+") as f:
        f.create_dataset("k_max_first", data=first)
        f.attrs["capture_gamma"] = 0.5
        # Exercise legacy gamma-only metadata rather than the new fixture's
        # authoritative capture_time_s attribute.
        del f.attrs["capture_time_s"]
    provider = _provider(tmp_path)
    assert provider.get_activation_rate("E_Beijing__HAP_Test", 7) == pytest.approx(1.25)
    assert provider.capture_gamma == pytest.approx(0.5)
    provider.close()


def test_h5_provider_rejects_marked_domain_dataset_by_default(tmp_path):
    _write_dataset(tmp_path)
    with h5py.File(tmp_path / "link_data.h5", "r+") as f:
        f.attrs["domain_policy"] = "mark"
    with pytest.raises(ValueError, match="domain_policy=mark"):
        _provider(tmp_path)


def test_h5_provider_allows_marked_domain_only_when_explicit(tmp_path):
    data = _write_dataset(tmp_path)
    with h5py.File(tmp_path / "link_data.h5", "r+") as f:
        f.attrs["domain_policy"] = "mark"
    provider = _provider(tmp_path, allow_domain_invalid=True)
    assert provider.get_rate("E_Beijing__HAP_Test", 0) == pytest.approx(float(data["kmax"][0, 0]))
    provider.close()


def test_rate_stats_can_be_calibrated_from_v75_proxy_surface(tmp_path, monkeypatch):
    data = _write_dataset(tmp_path)
    from scripts import estimate_rate_stats as ers

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "estimate_rate_stats.py",
            "--dataset-dir", str(tmp_path),
            "--rate-key", "k_proxy",
            "--stride", "1",
        ],
    )
    ers.main()
    stats_path = tmp_path / "rate_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["rate_key"] == "k_proxy"
    assert stats["global"]["p99"] == pytest.approx(
        float(np.percentile(data["kproxy"][data["kproxy"] > 0.0], 99))
    )

    normalizer = RateNormalizer({"mode": "log_p99", "stats_path": str(stats_path)})
    cfg = {
        "h5": {
            "dataset_dir": str(tmp_path),
            "rate_dataset_key": "k_proxy",
        },
        "rate": {"min_link_rate": 0.001, "unavailable_if_nan": True},
    }
    provider = H5RateProvider(cfg, seed=0)
    provider.setup(_edges(), horizon=2, min_link_rate=0.001, normalizer=normalizer)
    assert provider.get_rate("E_Beijing__HAP_Test", 7) == pytest.approx(float(data["kproxy"][7, 0]))
    provider.close()


def test_activation_adjusted_blocks_use_first_surface_only_for_new_links(tmp_path):
    data = _write_dataset(tmp_path)
    with h5py.File(tmp_path / "link_data.h5", "r+") as f:
        f.attrs["capture_time_s"] = 5.0
    provider = _provider(
        tmp_path,
        horizon=2,
        rate_dataset_key="k_proxy",
        activation_rate_dataset_key="k_proxy_first",
    )
    edge = "E_Beijing__HAP_Test"
    link_id = int(provider.edge_link_ids([edge])[0])
    base = provider.get_window_blocks(7)

    new_link = provider.activation_adjusted_window_blocks(7, base, [])
    assert new_link[0][0, link_id] == pytest.approx(float(data["kproxy_first"][7, 0]))
    assert new_link[0][1, link_id] == pytest.approx(base[0][1, link_id])

    kept_link = provider.activation_adjusted_window_blocks(7, base, [edge])
    assert kept_link[0][0, link_id] == pytest.approx(float(data["kproxy"][7, 0]))
    np.testing.assert_allclose(kept_link[0][1:], base[0][1:])
    provider.close()
