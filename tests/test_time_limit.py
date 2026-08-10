"""Tests for the global ``scenario.time_limit`` window cap.

The time limit is applied at scenario build time, so every consumer (RL
agent, all baselines, rate provider, request generator) sees the same
shortened horizon without any per-algorithm logic.
"""
from __future__ import annotations

import csv

from qkd_rl.data.scenario_builder import ScenarioBuilder
from qkd_rl.env.factory import build_env_from_config, load_default_config
from tests.helpers import point_config_to_h5


def _small_config(days: int, end_index: int = 525600, day_steps: int = 1440) -> dict:
    config = load_default_config(".")
    config["scenario"]["time_limit"] = {"days": days}
    config["rate_provider"]["time"]["end_index"] = end_index
    config["env"]["day_steps"] = day_steps
    return config


def test_small_time_limit_zero_keeps_full_window() -> None:
    scenario = ScenarioBuilder(_small_config(days=0)).build_small()
    assert scenario.start_t == 0
    assert scenario.end_t == 525600


def test_small_time_limit_days_truncates_end_t() -> None:
    scenario = ScenarioBuilder(_small_config(days=2)).build_small()
    assert scenario.start_t == 0
    assert scenario.end_t == 2 * 1440


def test_small_time_limit_uses_env_day_steps() -> None:
    scenario = ScenarioBuilder(_small_config(days=2, day_steps=10)).build_small()
    assert scenario.end_t == 20


def test_time_limit_never_extends_configured_window() -> None:
    # 365 days exceeds the configured 5000-slot window; the cap must not grow it.
    scenario = ScenarioBuilder(_small_config(days=365, end_index=5000)).build_small()
    assert scenario.end_t == 5000


def _full_scenario(tmp_path, time_limit_days: int, end_index: int) -> ScenarioBuilder:
    node_csv = tmp_path / "node_registry.csv"
    with node_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "type", "name", "lat", "lon", "alt_km"])
        writer.writeheader()
        rows = [(0, "GS", "Beijing", 39.9, 116.4, 0.0), (1, "GS", "Shanghai", 31.2, 121.5, 0.0),
                (2, "SAT", "Sat_LEO_Mid_1", 53.0, 30.0, 550.0)]
        for node_id, typ, name, lat, lon, alt in rows:
            writer.writerow({"node_id": node_id, "type": typ, "name": name, "lat": lat, "lon": lon, "alt_km": alt})
    link_csv = tmp_path / "link_registry.csv"
    with link_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["link_id", "node_u", "node_v", "link_type"])
        writer.writeheader()
        writer.writerow({"link_id": 0, "node_u": 0, "node_v": 2, "link_type": "SAT-GS"})
        writer.writerow({"link_id": 1, "node_u": 1, "node_v": 2, "link_type": "SAT-GS"})

    config = load_default_config(".")
    config["scenario"]["mode"] = "full"
    config["scenario"]["time_limit"] = {"days": time_limit_days}
    config["rate_provider"]["time"]["start_index"] = 0
    config["rate_provider"]["time"]["end_index"] = end_index
    config["rate_provider"]["h5"]["node_registry_path"] = str(node_csv)
    config["rate_provider"]["h5"]["link_registry_path"] = str(link_csv)
    return ScenarioBuilder(config)


def test_full_time_limit_days_truncates_end_t(tmp_path) -> None:
    scenario = _full_scenario(tmp_path, time_limit_days=7, end_index=525600).build_full()
    assert scenario.start_t == 0
    assert scenario.end_t == 7 * 1440


def test_full_time_limit_zero_keeps_window(tmp_path) -> None:
    scenario = _full_scenario(tmp_path, time_limit_days=0, end_index=525600).build_full()
    assert scenario.end_t == 525600


def test_time_limit_env_end_to_end_terminates_early() -> None:
    config = point_config_to_h5(load_default_config("."))
    config["scenario"]["time_limit"] = {"days": 1}
    config["rate_provider"]["time"]["start_index"] = 0
    config["rate_provider"]["time"]["end_index"] = 50
    config["env"]["episode_steps"] = 10000
    env = build_env_from_config(config)
    assert env.scenario.end_t == 50

    env.reset(seed=7)
    idle = {node: "idle" for node in env.scenario.node_ids}
    first_truncated_step = None
    for step in range(60):
        _obs, _reward, terminated, truncated, _info = env.step(idle)
        assert not terminated, "episode_steps must not terminate the run"
        if truncated:
            first_truncated_step = step + 1
            break
    assert first_truncated_step == 50, f"expected truncation at step 50, got {first_truncated_step}"