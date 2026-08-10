"""Tests that feature config switches stay consistent with resolved dimensions."""

from __future__ import annotations

from qkd_rl.core.config import ConfigValidator, deep_merge
from qkd_rl.env.factory import build_env_from_config, load_default_config
from tests.helpers import point_config_to_h5
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


def _build_with_switches(switches: dict):
    config = point_config_to_h5(load_default_config("."))
    config = deep_merge(config, switches)
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    obs = env.reset()
    dims = config["features"]["dims"]
    return config, env, obs, dims


def _assert_dims_and_forward(obs, dims, config, env):
    assert len(obs.node_features[0]) == dims["node_dim_resolved"]
    assert len(obs.edge_features[0]) == dims["edge_dim_resolved"]
    for row in obs.edge_features:
        assert len(row) == dims["edge_dim_resolved"]
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    output = model(obs)
    assert set(output.logits) == set(obs.node_ids)


def test_default_feature_dims():
    config, env, obs, dims = _build_with_switches({})
    assert dims["node_dim_resolved"] == 17
    assert dims["edge_dim_resolved"] == 40
    _assert_dims_and_forward(obs, dims, config, env)


def test_switches_off_keep_dims_consistent():
    switches = {
        "features": {
            "node": {
                "include_qkp_level": False,
                "include_qkp_utilization": False,
                "include_queue_pressure": False,
                "include_is_available": False,
                "include_recent_demand": False,
            },
            "edge": {
                "include_rate_delta": False,
                "include_rate_mean": False,
                "include_rate_max": False,
                "include_last_activated": False,
                "include_rate_future_window": False,
                "include_available_future_window": False,
            },
        }
    }
    config, env, obs, dims = _build_with_switches(switches)
    assert dims["node_dim_resolved"] == 17 - 1 - 1 - 1 - 1 - 3
    assert dims["edge_dim_resolved"] == 40 - 6 - 6 - 1 - 1 - 1 - 1
    _assert_dims_and_forward(obs, dims, config, env)


def test_shorter_prediction_horizon():
    switches = {"features": {"edge": {"prediction_horizon": 2}}}
    config, env, obs, dims = _build_with_switches(switches)
    assert dims["edge_dim_resolved"] == 40 - 4 - 4
    _assert_dims_and_forward(obs, dims, config, env)
