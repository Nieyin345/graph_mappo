from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(paths: list[str | Path]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for path in paths:
        config = deep_merge(config, load_yaml(path))
    return config


class ConfigValidator:
    def resolve_feature_dims(self, config: dict[str, Any]) -> dict[str, int]:
        features = config["features"]
        node_cfg = features["node"]
        edge_cfg = features["edge"]
        demand_edge_cfg = features.get("demand_edge", {})

        node_dim = 0
        if node_cfg.get("include_node_type_one_hot", True):
            node_dim += int(node_cfg.get("node_type_dim", 3))
        for flag in (
            "include_qkp_level",
            "include_qkp_capacity_left",
            "include_qkp_utilization",
            "include_demand_in",
            "include_demand_out",
            "include_queue_pressure",
            "include_is_available",
        ):
            node_dim += int(bool(node_cfg.get(flag, False)))
        if node_cfg.get("include_recent_demand", False):
            node_dim += len(node_cfg.get("recent_demand_windows", []))
        if node_cfg.get("include_time_features", False):
            time_cfg = node_cfg.get("time_features", {})
            node_dim += 2 * int(bool(time_cfg.get("minute_of_day_sin_cos", False)))
            node_dim += 2 * int(bool(time_cfg.get("day_of_year_sin_cos", False)))
        if node_cfg.get("include_position", False):
            node_dim += int(node_cfg.get("position_dim", 3))

        edge_dim = 0
        if edge_cfg.get("include_link_type_one_hot", True):
            edge_dim += int(edge_cfg.get("link_type_dim", 4))
        edge_dim += int(bool(edge_cfg.get("include_available_now", True)))
        edge_dim += int(bool(edge_cfg.get("include_rate_now", True)))
        horizon = int(edge_cfg.get("prediction_horizon", 0))
        if edge_cfg.get("include_rate_future_window", True):
            edge_dim += horizon
        if edge_cfg.get("include_available_future_window", True):
            edge_dim += horizon
        for flag in ("include_rate_delta", "include_rate_mean", "include_rate_max", "include_last_activated", "include_switch_cost"):
            edge_dim += int(bool(edge_cfg.get(flag, False)))

        demand_edge_dim = 0
        if demand_edge_cfg.get("enabled", False):
            for flag in (
                "include_pending_amount",
                "include_pending_count",
                "include_min_deadline_left",
                "include_mean_deadline_left",
                "include_priority_sum",
            ):
                demand_edge_dim += int(bool(demand_edge_cfg.get(flag, False)))
            history_windows = demand_edge_cfg.get("history_windows", [])
            if demand_edge_cfg.get("include_arrival_history", False):
                demand_edge_dim += len(history_windows)
            if demand_edge_cfg.get("include_served_history", False):
                demand_edge_dim += len(history_windows)
            if demand_edge_cfg.get("include_failed_history", False):
                demand_edge_dim += len(history_windows)
        edge_dim += demand_edge_dim

        features.setdefault("dims", {})
        features["dims"]["node_dim_resolved"] = node_dim
        features["dims"]["edge_dim_resolved"] = edge_dim
        features["dims"]["physical_edge_dim_resolved"] = edge_dim - demand_edge_dim
        features["dims"]["demand_edge_dim_resolved"] = demand_edge_dim
        return {"node_dim": node_dim, "edge_dim": edge_dim}

    def validate(self, config: dict[str, Any]) -> None:
        dims = self.resolve_feature_dims(config)
        if dims["node_dim"] <= 0 or dims["edge_dim"] <= 0:
            raise ValueError(f"Invalid feature dims: {dims}")
        if config["features"]["edge"]["prediction_horizon"] < 0:
            raise ValueError("prediction_horizon must be non-negative.")
        if config["qkp"]["type"] != "link":
            raise ValueError("Current implementation expects qkp.type = link.")
