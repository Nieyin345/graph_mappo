"""Stored keys remain visible to the GNN without becoming generation actions."""
import numpy as np
import pytest
import torch

from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.rl.algos.policy import MAPPOPolicy
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
from tests.helpers import point_config_to_h5


@pytest.mark.parametrize("attention", [False, True])
def test_stocked_unavailable_edge_is_visible_but_not_actionable(attention):
    config = point_config_to_h5(load_default_config("."))
    config["features"]["edge"]["include_stocked_edges"] = True
    config["qkp"]["initial_level"] = 0.0
    config["model"]["actor"]["demand_residual"]["enabled"] = attention
    env = build_env_from_config(config)
    initial = env.reset(seed=7)
    edge = next(e for e in env.scenario.edges if e.edge_id not in initial.physical_edge_ids)
    env.qkp.add_keys(edge.edge_id, 100.0, env.t)
    obs = env._build_observation()
    assert edge.edge_id in obs.physical_edge_ids
    assert edge.dst not in obs.action_candidates[edge.src]
    assert edge.src not in obs.action_candidates[edge.dst]
    assert edge.edge_id not in env._active_edge_ids(obs.raw_action_masks)
    assert edge.edge_id not in obs.generation_edge_ids
    assert np.isfinite(obs.edge_features).all()
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    single = model(obs)
    batch = model.batched_forward([obs], "cpu")
    assert (edge.src, edge.dst) not in single.edge_scores
    torch.testing.assert_close(single.value, batch.values[0])
    for arc in single.edge_scores:
        torch.testing.assert_close(single.edge_scores[arc], batch.edge_score_maps[0][arc])
    policy = MAPPOPolicy(model)
    step = policy.act(obs, deterministic=True)
    env.step(step.actions, expected_matched_edges=step.matched_edges)


def test_legacy_observation_switch_keeps_stocked_edge_hidden():
    config = point_config_to_h5(load_default_config("."))
    config["features"]["edge"]["include_stocked_edges"] = False
    config["qkp"]["initial_level"] = 0.0
    env = build_env_from_config(config)
    obs = env.reset(seed=7)
    edge = next(e for e in env.scenario.edges if e.edge_id not in obs.physical_edge_ids)
    env.qkp.add_keys(edge.edge_id, 100.0, env.t)
    assert edge.edge_id not in env._build_observation().physical_edge_ids


def test_stocked_graph_does_not_change_expert_generation_actions():
    from qkd_rl.baselines.path_greedy import PathScoreGreedy

    config = point_config_to_h5(load_default_config("."))
    config["features"]["edge"]["include_stocked_edges"] = False
    env = build_env_from_config(config)
    obs = env.reset(seed=7)
    edge = next(e for e in env.scenario.edges if e.edge_id not in obs.physical_edge_ids)
    env.qkp.add_keys(edge.edge_id, 100.0, env.t)
    legacy = env._build_observation()
    env.config["features"]["edge"]["include_stocked_edges"] = True
    extended = env._build_observation()
    assert len(extended.physical_edge_ids) > len(legacy.physical_edge_ids)
    assert extended.generation_edge_ids == legacy.generation_edge_ids
    assert PathScoreGreedy(phased=True).act(legacy) == PathScoreGreedy(phased=True).act(extended)
