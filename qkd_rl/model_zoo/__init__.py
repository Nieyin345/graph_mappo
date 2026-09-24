"""Model plugins used by the common experiment runner."""

from __future__ import annotations

from importlib import import_module


def model_name(config: dict) -> str:
    name = str(config.get("experiment", {}).get("model", "v2"))
    if not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"Invalid experiment.model: {name!r}")
    return name


def build_model(action_space, config: dict):
    if "model" not in config.get("experiment", {}):
        from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
        return GraphMAPPOActorCritic(action_space, config)
    module = import_module(f"qkd_rl.model_zoo.{model_name(config)}.model")
    return module.build_model(action_space, config)


def graph_builder_class(config: dict):
    if "model" not in config.get("experiment", {}):
        from qkd_rl.env.graph_builder import GraphBuilder
        return GraphBuilder
    module = import_module(f"qkd_rl.model_zoo.{model_name(config)}.model")
    return module.graph_builder_class()
