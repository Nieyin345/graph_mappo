from __future__ import annotations

from pathlib import Path

from qkd_rl.core.config import ConfigValidator, load_config
from qkd_rl.data.scenario_builder import ScenarioBuilder
from qkd_rl.env.action_resolver import ActionResolver
from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.env import QKDEnv
from qkd_rl.env.graph_builder import GraphBuilder
from qkd_rl.env.history_buffer import HistoryBuffer
from qkd_rl.env.masks import ActionMaskBuilder
from qkd_rl.env.metrics import MetricsTracker
from qkd_rl.env.qkp import LinkQKPPool
from qkd_rl.env.request import RequestGenerator
from qkd_rl.env.reward import RewardFunction
from qkd_rl.env.routing import RoutingPolicy
from qkd_rl.link.rate_provider import RateNormalizer, build_rate_provider


DEFAULT_CONFIG_FILES = [
    "default.yaml",
    "rate_provider.yaml",
    "features.yaml",
    "env_small.yaml",
    "graph_mappo.yaml",
    "train_mappo.yaml",
]


def default_config_paths(project_root: str | Path) -> list[Path]:
    config_dir = Path(project_root) / "configs"
    return [config_dir / name for name in DEFAULT_CONFIG_FILES]


def load_default_config(project_root: str | Path) -> dict:
    config = load_config(default_config_paths(project_root))
    ConfigValidator().validate(config)
    return config


def build_env_from_config(config: dict) -> QKDEnv:
    scenario_builder = ScenarioBuilder(config)
    scenario_mode = config["scenario"].get("mode", "small")
    if scenario_mode == "small":
        scenario = scenario_builder.build_small()
    elif scenario_mode == "full":
        scenario = scenario_builder.build_full()
    else:
        raise ValueError(f"Unknown scenario mode: {scenario_mode!r}")
    normalizer = RateNormalizer(config["rate_provider"]["normalization"])
    rate_provider = build_rate_provider(
        config, scenario.edges, seed=int(config["seed"]["global_seed"]), normalizer=normalizer
    )
    qkp = LinkQKPPool(scenario.edges, config["qkp"])
    action_space = NodeActionSpace(
        scenario.node_ids,
        scenario.edges,
        include_idle=bool(config["features"]["action"].get("include_idle_action", True)),
    )
    mask_builder = ActionMaskBuilder(
        action_space,
        config["features"]["action"],
        min_link_rate=float(config["rate_provider"]["rate"]["min_link_rate"]),
        allowed_link_types=config["scenario"].get("allowed_link_types", []),
    )
    action_resolver = ActionResolver(action_space, config["action_resolver"])
    routing = RoutingPolicy(scenario.edges, config["routing"])
    history_cfg = config["features"].get("history_encoder", {})
    history_buffer = (
        HistoryBuffer(scenario.nodes, scenario.edges, config) if history_cfg.get("enabled", False) else None
    )
    graph_builder = GraphBuilder(
        scenario.nodes,
        scenario.edges,
        action_space,
        qkp,
        normalizer,
        config,
        history_buffer=history_buffer,
        routing=routing,
        rate_provider=rate_provider,
    )
    request_generator = RequestGenerator(
        [node.node_id for node in scenario.nodes if node.node_type.value == "gs"],
        config["requests"],
        seed=int(config["seed"]["env_seed"]),
    )
    return QKDEnv(
        scenario=scenario,
        rate_provider=rate_provider,
        request_generator=request_generator,
        qkp=qkp,
        routing=routing,
        reward_fn=RewardFunction(config["reward"]),
        graph_builder=graph_builder,
        mask_builder=mask_builder,
        action_resolver=action_resolver,
        metrics=MetricsTracker(),
        config=config,
        history_buffer=history_buffer,
    )


def build_default_env(project_root: str | Path) -> QKDEnv:
    return build_env_from_config(load_default_config(project_root))


