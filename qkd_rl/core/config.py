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
        for flag in (
            "include_rate_delta",
            "include_rate_mean",
            "include_rate_max",
            "include_last_activated",
            "include_relay_importance",
            "include_qkp_capacity_left",
        ):
            edge_dim += int(bool(edge_cfg.get(flag, False)))

        demand_edge_dim = 0
        if demand_edge_cfg.get("enabled", False):
            for flag in (
                "include_pending_amount",
                "include_pending_count",
                "include_min_deadline_left",
                "include_mean_deadline_left",
                "include_mean_wait_time",
                "include_priority_sum",
            ):
                demand_edge_dim += int(bool(demand_edge_cfg.get(flag, False)))
            if demand_edge_cfg.get("include_wait_bucket_amounts", False):
                demand_edge_dim += int(demand_edge_cfg.get("wait_bucket_count", 10))
        edge_dim += demand_edge_dim

        history_cfg = features.get("history_encoder", {})
        history_enabled = bool(history_cfg.get("enabled", False))
        history_hidden = int(history_cfg.get("hidden_dim", 128))
        node_history_dim = 0
        physical_history_dim = 0
        demand_history_dim = 0
        if history_enabled:
            node_cfg = history_cfg.get("node", {})
            if any(
                bool(node_cfg.get(key, default))
                for key, default in (
                    ("include_arrived", True),
                    ("include_served", True),
                    ("include_failed", True),
                    ("include_qkp_total", True),
                )
            ):
                node_history_dim = history_hidden
            phys_cfg = history_cfg.get("physical_edge", {})
            if any(
                bool(phys_cfg.get(key, default))
                for key, default in (
                    ("include_qkp_level", True),
                    ("include_available", True),
                    ("include_activated", False),
                )
            ):
                physical_history_dim = history_hidden
            demand_cfg = history_cfg.get("demand_edge", {})
            if demand_cfg.get("include_pending_wait_buckets", False):
                demand_history_dim = history_hidden
        history_dim = max(physical_history_dim, demand_history_dim)

        features.setdefault("dims", {})
        features["dims"]["node_dim_resolved"] = node_dim
        features["dims"]["edge_dim_resolved"] = edge_dim
        features["dims"]["physical_edge_dim_resolved"] = edge_dim - demand_edge_dim
        features["dims"]["demand_edge_dim_resolved"] = demand_edge_dim
        features["dims"]["history_dim_resolved"] = history_dim
        features["dims"]["node_history_dim_resolved"] = node_history_dim
        features["dims"]["physical_edge_history_dim_resolved"] = physical_history_dim
        features["dims"]["demand_edge_history_dim_resolved"] = demand_history_dim
        return {"node_dim": node_dim, "edge_dim": edge_dim}

    def validate(self, config: dict[str, Any]) -> None:
        dims = self.resolve_feature_dims(config)
        if dims["node_dim"] <= 0 or dims["edge_dim"] <= 0:
            raise ValueError(f"Invalid feature dims: {dims}")
        if config["features"]["edge"]["prediction_horizon"] < 0:
            raise ValueError("prediction_horizon must be non-negative.")
        if config["qkp"]["type"] != "link":
            raise ValueError("Current implementation expects qkp.type = link.")
        history_cfg = config["features"].get("history_encoder", {})
        if history_cfg.get("enabled", False):
            if int(history_cfg.get("seq_len", 0)) <= 0:
                raise ValueError("history_encoder.seq_len must be positive when enabled.")
            if int(history_cfg.get("hidden_dim", 0)) <= 0:
                raise ValueError("history_encoder.hidden_dim must be positive when enabled.")
            if history_cfg.get("type", "lstm") != "lstm":
                raise ValueError("history_encoder.type: only 'lstm' is implemented.")
        self._validate_options(config)

    @staticmethod
    def _validate_options(config: dict[str, Any]) -> None:
        """Reject option values that exist in YAML but are not implemented, so
        a silently ignored setting can never look like it is in control."""
        model_cfg = config["model"]
        if model_cfg.get("name", "graph_mappo") != "graph_mappo":
            raise ValueError("model.name: only 'graph_mappo' is implemented.")
        if model_cfg.get("mode", "mixed") not in ("mixed", "demand_edge"):
            raise ValueError("model.mode: supported values are 'mixed' and 'demand_edge'.")
        if not config["project"].get("name"):
            raise ValueError("project.name must be set.")
        enc = config["model"]["encoder"]
        if enc.get("gnn_type", "graphsage") != "graphsage":
            raise ValueError("model.encoder.gnn_type: only 'graphsage' is implemented.")
        if not bool(enc.get("share_actor_critic_encoder", True)):
            raise ValueError("model.encoder.share_actor_critic_encoder: False (separate actor/critic encoders) is not implemented.")
        actor_cfg = config["model"]["actor"]
        if not bool(actor_cfg.get("share_actor_across_node_types", True)):
            raise ValueError("model.actor.share_actor_across_node_types: False (per-node-type actors) is not implemented.")
        if float(actor_cfg.get("temperature", 1.0)) <= 0:
            raise ValueError("model.actor.temperature must be positive.")
        qkp_cfg = config["qkp"]
        if qkp_cfg.get("overflow_policy", "discard") != "discard":
            raise ValueError("qkp.overflow_policy: only 'discard' is implemented.")
        routing_cfg = config["routing"]
        if routing_cfg.get("mode", "link_path") != "link_path":
            raise ValueError("routing.mode: only 'link_path' is implemented.")
        if routing_cfg.get("path_selection", "shortest_available_path") != "shortest_available_path":
            raise ValueError("routing.path_selection: only 'shortest_available_path' is implemented.")
        if routing_cfg.get("serve_order", "earliest_deadline_first") != "earliest_deadline_first":
            raise ValueError("routing.serve_order: only 'earliest_deadline_first' is implemented.")
        rate_cfg = config["rate_provider"]["rate"]
        if rate_cfg.get("negative_rate_policy", "clip_to_zero") not in ("clip_to_zero", "raise"):
            raise ValueError("rate_provider.rate.negative_rate_policy: supported values are 'clip_to_zero' and 'raise'.")
        if config["runtime"].get("dtype", "float32") != "float32":
            raise ValueError("runtime.dtype: only 'float32' is implemented.")

