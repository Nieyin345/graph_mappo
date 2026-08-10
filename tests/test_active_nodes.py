"""Tests for the global ``scenario.active_nodes`` subset knob.

The active-node subset is applied at scenario build time, so every consumer
(RL agent, all baselines, rate provider, request generator, masks) sees the
same shrunk graph without any per-algorithm logic.
"""
from __future__ import annotations

import csv

from qkd_rl.data.scenario_builder import ScenarioBuilder
from qkd_rl.env.factory import build_env_from_config, load_default_config
from tests.helpers import build_test_env, point_config_to_h5


def _small_config(gs_count: int = 0, hap_count: int = 0, sat_count: int = 0) -> dict:
    config = load_default_config(".")
    config["scenario"]["active_nodes"] = {
        "gs_count": gs_count,
        "hap_count": hap_count,
        "sat_count": sat_count,
    }
    return config


def test_default_keeps_all_nodes() -> None:
    config = load_default_config(".")
    config["scenario"].pop("active_nodes", None)
    scenario = ScenarioBuilder(config).build_small()
    assert scenario.node_ids == ["GS_001", "GS_002", "GS_003", "GS_004", "HAP_001", "SAT_001"]


def test_small_active_nodes_subset_keeps_first_n() -> None:
    scenario = ScenarioBuilder(_small_config(gs_count=2)).build_small()
    assert scenario.node_ids == ["GS_001", "GS_002", "HAP_001", "SAT_001"]
    kept = set(scenario.node_ids)
    for edge in scenario.edges:
        assert edge.src in kept and edge.dst in kept
    assert len(scenario.edges) == 5  # 2 GS x (HAP + SAT) + HAP-SAT


def test_small_active_nodes_zero_means_keep_all() -> None:
    # gs_count=0 (keep all) with relay limits active: only relays shrink.
    scenario = ScenarioBuilder(_small_config(gs_count=0, hap_count=0, sat_count=0)).build_small()
    assert len(scenario.nodes) == 6


def test_full_active_nodes_subset(tmp_path) -> None:
    node_csv = tmp_path / "node_registry.csv"
    with node_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "type", "name", "lat", "lon", "alt_km"])
        writer.writeheader()
        rows = [
            (0, "GS", "Beijing"), (1, "GS", "Shanghai"), (2, "GS", "Changsha"), (3, "GS", "Guangzhou"),
            (4, "HAP", "HAP_01"), (5, "HAP", "HAP_02"), (6, "SAT", "SAT_01"), (7, "SAT", "SAT_02"),
        ]
        for node_id, typ, name in rows:
            writer.writerow({"node_id": node_id, "type": typ, "name": name, "lat": 0.0, "lon": 0.0, "alt_km": 0.0})

    link_csv = tmp_path / "link_registry.csv"
    with link_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["link_id", "node_u", "node_v", "link_type"])
        writer.writeheader()
        links = [
            (0, 0, 4, "HAP-GS"), (1, 1, 4, "HAP-GS"), (2, 2, 5, "HAP-GS"), (3, 3, 5, "HAP-GS"),
            (4, 4, 6, "SAT-HAP"), (5, 5, 6, "SAT-HAP"), (6, 0, 6, "SAT-GS"), (7, 6, 7, "SAT-SAT"),
        ]
        for link_id, u, v, typ in links:
            writer.writerow({"link_id": link_id, "node_u": u, "node_v": v, "link_type": typ})

    config = load_default_config(".")
    config["scenario"]["mode"] = "full"
    config["scenario"]["active_nodes"] = {"gs_count": 2, "hap_count": 1, "sat_count": 0}
    config["rate_provider"]["h5"]["node_registry_path"] = str(node_csv)
    config["rate_provider"]["h5"]["link_registry_path"] = str(link_csv)

    scenario = ScenarioBuilder(config).build_full()
    assert scenario.node_ids == ["Beijing", "Shanghai", "HAP_01", "SAT_01", "SAT_02"]
    kept = set(scenario.node_ids)
    for edge in scenario.edges:
        assert edge.src in kept and edge.dst in kept
    edge_ids = set(scenario.edge_ids)
    assert "E_Beijing__HAP_01" in edge_ids
    assert "E_Changsha__HAP_02" not in edge_ids
    assert "E_HAP_02__SAT_01" not in edge_ids
    assert len(scenario.edges) == 5


def test_active_nodes_env_end_to_end() -> None:
    config = point_config_to_h5(load_default_config("."))
    config["scenario"]["active_nodes"] = {"gs_count": 2, "hap_count": 0, "sat_count": 0}
    env = build_env_from_config(config)
    assert env.scenario.node_ids == ["GS_001", "GS_002", "HAP_001", "SAT_001"]

    obs = env.reset(seed=7)
    idle = {node: "idle" for node in env.scenario.node_ids}
    for _ in range(3):
        obs, _reward, _term, _trunc, _info = env.step(idle)
    assert len(obs.node_ids) == 4
def test_smart_mode_picks_coherent_full_subset(tmp_path) -> None:
    node_csv = tmp_path / "node_registry.csv"
    with node_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "type", "name", "lat", "lon", "alt_km"])
        writer.writeheader()
        rows = [
            (0, "GS", "Beijing", 39.9, 116.4, 0.0),
            (1, "GS", "Shanghai", 31.2, 121.5, 0.0),
            (2, "GS", "Urumqi", 43.8, 87.6, 0.0),
            (3, "GS", "Lhasa", 29.7, 91.1, 0.0),
            (4, "HAP", "HAP_Beijing_Jingjinji", 39.5, 116.0, 20.0),
            (5, "HAP", "HAP_Shanghai_Yangtze", 31.0, 121.0, 20.0),
            (6, "HAP", "HAP_West_Remote", 42.0, 90.0, 20.0),
            (7, "SAT", "Sat_LEO_Eq_1", 0.0, 0.0, 500.0),
            (8, "SAT", "Sat_LEO_Pol_1", 90.0, 0.0, 600.0),
            (9, "SAT", "Sat_LEO_Mid_1", 53.0, 30.0, 550.0),
            (10, "SAT", "Sat_MEO_1", 45.0, 45.0, 10000.0),
        ]
        for node_id, typ, name, lat, lon, alt in rows:
            writer.writerow({"node_id": node_id, "type": typ, "name": name, "lat": lat, "lon": lon, "alt_km": alt})
    link_csv = tmp_path / "link_registry.csv"
    with link_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["link_id", "node_u", "node_v", "link_type"])
        writer.writeheader()
        for link_id, u, v in [(0, 0, 4), (1, 1, 5), (2, 2, 6)]:
            writer.writerow({"link_id": link_id, "node_u": u, "node_v": v, "link_type": "HAP-GS"})

    config = load_default_config(".")
    config["scenario"]["mode"] = "full"
    config["scenario"]["active_nodes"] = {
        "mode": "smart",
        "gs_count": 2,
        "hap_count": 2,
        "sat_count": 2,
        "gs_center": [35.0, 110.0],
    }
    config["rate_provider"]["h5"]["node_registry_path"] = str(node_csv)
    config["rate_provider"]["h5"]["link_registry_path"] = str(link_csv)

    scenario = ScenarioBuilder(config).build_full()
    # GS: nearest to the (35, 110) center -> Beijing + Shanghai, not the remote
    # Urumqi/Lhasa; HAP: the two nearest to the selected GS; SAT: inclined
    # orbits only (no Eq/Pol/GEO).
    assert sorted(scenario.node_ids) == sorted([
        "Beijing",
        "Shanghai",
        "HAP_Beijing_Jingjinji",
        "HAP_Shanghai_Yangtze",
        "Sat_LEO_Mid_1",
        "Sat_MEO_1",
    ])

def test_smart_mode_small_equals_first_n() -> None:
    # Small-scenario nodes all sit at (0, 0): smart ranking is stable and
    # degenerates to the same subset as first_n.
    config = _small_config(gs_count=2)
    config["scenario"]["active_nodes"]["mode"] = "smart"
    scenario = ScenarioBuilder(config).build_small()
    assert scenario.node_ids == ["GS_001", "GS_002", "HAP_001", "SAT_001"]

def test_smart_mode_does_not_pad_hap_count_with_far_platforms(tmp_path) -> None:
    node_csv = tmp_path / "node_registry.csv"
    with node_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "type", "name", "lat", "lon", "alt_km"])
        writer.writeheader()
        rows = [
            (0, "GS", "Beijing", 39.9, 116.4, 0.0),
            (1, "GS", "Shanghai", 31.2, 121.5, 0.0),
            (2, "HAP", "HAP_Beijing_Jingjinji", 39.5, 116.0, 20.0),
            (3, "HAP", "HAP_Shanghai_Yangtze", 31.0, 121.0, 20.0),
            (4, "HAP", "HAP_West_Remote", 42.0, 90.0, 20.0),
            (5, "SAT", "Sat_LEO_Mid_1", 53.0, 30.0, 550.0),
        ]
        for node_id, typ, name, lat, lon, alt in rows:
            writer.writerow({"node_id": node_id, "type": typ, "name": name, "lat": lat, "lon": lon, "alt_km": alt})
    link_csv = tmp_path / "link_registry.csv"
    with link_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["link_id", "node_u", "node_v", "link_type"])
        writer.writeheader()
        for link_id, u, v in [(0, 0, 2), (1, 1, 3)]:
            writer.writerow({"link_id": link_id, "node_u": u, "node_v": v, "link_type": "HAP-GS"})

    config = load_default_config(".")
    config["scenario"]["mode"] = "full"
    config["scenario"]["active_nodes"] = {
        "mode": "smart",
        "gs_count": 2,
        "hap_count": 3,
        "sat_count": 0,
        "hap_gs_max_km": 700.0,
    }
    config["rate_provider"]["h5"]["node_registry_path"] = str(node_csv)
    config["rate_provider"]["h5"]["link_registry_path"] = str(link_csv)

    scenario = ScenarioBuilder(config).build_full()
    # Only the two platforms within 700 km of a selected GS are kept; the far
    # HAP_West_Remote (~2200 km) is dropped even though hap_count requests 3.
    assert sorted(scenario.node_ids) == sorted([
        "Beijing",
        "Shanghai",
        "HAP_Beijing_Jingjinji",
        "HAP_Shanghai_Yangtze",
        "Sat_LEO_Mid_1",
    ])

def test_smart_mode_hap_gs_max_km_is_configurable(tmp_path) -> None:
    node_csv = tmp_path / "node_registry.csv"
    with node_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "type", "name", "lat", "lon", "alt_km"])
        writer.writeheader()
        rows = [
            (0, "GS", "Beijing", 39.9, 116.4, 0.0),
            (1, "GS", "Shanghai", 31.2, 121.5, 0.0),
            (2, "HAP", "HAP_Beijing_Jingjinji", 39.5, 116.0, 20.0),
            (3, "HAP", "HAP_West_Remote", 42.0, 90.0, 20.0),
            (4, "SAT", "Sat_LEO_Mid_1", 53.0, 30.0, 550.0),
        ]
        for node_id, typ, name, lat, lon, alt in rows:
            writer.writerow({"node_id": node_id, "type": typ, "name": name, "lat": lat, "lon": lon, "alt_km": alt})
    link_csv = tmp_path / "link_registry.csv"
    with link_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["link_id", "node_u", "node_v", "link_type"])
        writer.writeheader()
        for link_id, u, v in [(0, 0, 2)]:
            writer.writerow({"link_id": link_id, "node_u": u, "node_v": v, "link_type": "HAP-GS"})

    def _build(hap_gs_max_km: float) -> list[str]:
        config = load_default_config(".")
        config["scenario"]["mode"] = "full"
        config["scenario"]["active_nodes"] = {
            "mode": "smart",
            "gs_count": 2,
            "hap_count": 2,
            "sat_count": 0,
            "hap_gs_max_km": hap_gs_max_km,
        }
        config["rate_provider"]["h5"]["node_registry_path"] = str(node_csv)
        config["rate_provider"]["h5"]["link_registry_path"] = str(link_csv)
        return ScenarioBuilder(config).build_full().node_ids

    # 700 km: the remote HAP (~2200 km from the nearest selected GS) is dropped.
    assert "HAP_West_Remote" not in _build(700.0)
    # 3000 km: the threshold allows the remote HAP back in (still nearest first).
    assert "HAP_West_Remote" in _build(3000.0)

def test_unknown_active_nodes_mode_raises() -> None:
    config = _small_config(gs_count=2)
    config["scenario"]["active_nodes"]["mode"] = "bogus"
    try:
        ScenarioBuilder(config).build_small()
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown active_nodes mode")