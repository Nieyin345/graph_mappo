from __future__ import annotations

import csv
import math

import h5py
import numpy as np
import pytest

from qkd_rl.core.config import ConfigValidator
from qkd_rl.core.types import Edge, LinkType
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.link.rate_provider import H5RateProvider, build_rate_provider

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
        f.create_dataset("los", data=los)
        f.create_dataset("distance", data=distance)
        f.create_dataset("zenith", data=zenith)
        f.attrs["theta_sec"] = 60
    return {"kmax": kmax, "los": los, "T": t_steps, "L": L}


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


def test_h5_provider_missing_edge_raises(tmp_path):
    _write_dataset(tmp_path)
    cfg = {"h5": {"dataset_dir": str(tmp_path)}, "rate": {"min_link_rate": 0.001}}
    provider = H5RateProvider(cfg, seed=0)
    extra_edges = _edges() + [Edge("E_Beijing__Urumqi", "Beijing", "Urumqi", LinkType.GS_HAP)]
    with pytest.raises(ValueError, match="No H5 link"):
        provider.setup(extra_edges, horizon=6, min_link_rate=0.001)
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
