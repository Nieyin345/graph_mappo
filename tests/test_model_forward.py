from __future__ import annotations

import torch
import pytest

from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.core.config import ConfigValidator
from qkd_rl.env.factory import build_env_from_config, load_default_config
from tests.helpers import build_test_env, point_config_to_h5
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic
import numpy as np


def test_graph_mappo_actor_critic_forward_matches_action_candidates():
    config = load_default_config(".")
    env = build_test_env(".")
    obs = env.reset()
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)

    output = model(obs)

    assert set(output.logits) == set(obs.node_ids)
    assert output.value.shape == torch.Size([])
    for node_id, logits in output.logits.items():
        assert logits.shape[0] == len(obs.action_candidates[node_id])


def test_demand_edge_mode_uses_edge_only_actor():
    config = point_config_to_h5(load_default_config("."))
    config["model"]["mode"] = "demand_edge"
    config["features"]["edge"]["include_relay_importance"] = True
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    obs = env.reset()
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)

    output = model(obs)

    assert set(output.logits) == set(obs.node_ids)
    assert output.edge_scores
    assert not model.encoder.layers[0].fuse_physical_to_node
    assert model.actor.edge_scorer[0].in_features == int(
        config["model"]["encoder"]["hidden_dim"]
    )


def test_demand_edge_mode_batched_policy_roundtrip():
    config = point_config_to_h5(load_default_config("."))
    config["model"]["mode"] = "demand_edge"
    config["features"]["edge"]["include_relay_importance"] = True
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    obs = env.reset()
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model)

    step = policy.act_batched([obs], deterministic=True)[0]
    assert step.edge_scores
    log_probs, entropies, value = policy.evaluate_actions(
        obs, step.actions, step.matched_edges
    )
    assert value.shape == torch.Size([])
    assert log_probs
    assert entropies


def test_edge_projection_split_phys_vs_demand():
    """Physical and demand edge projections are separate modules: identical
    feature rows get different embeddings depending on which projection runs,
    so the two edge types are not forced to share a transform."""
    config = load_default_config(".")
    env = build_test_env(".")
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    enc = model.encoder
    assert enc.edge_proj_phys is not enc.edge_proj_demand
    feat = torch.randn(1, enc.edge_dim)
    emb = torch.cat([enc.edge_proj_phys(feat), enc.edge_proj_demand(feat)], dim=0)
    assert emb.shape == (2, int(config["model"]["encoder"]["hidden_dim"]))
    assert not torch.allclose(emb[0], emb[1])


def test_flat_action_masks_reuse_matches_dict_path():
    """The mask builder's flat legal array must be reusable by the graph
    builder and the actor plan without changing any output (implementation
    speedup only)."""
    config = load_default_config(".")
    env = build_test_env(".")
    obs = env.reset()
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)

    state = env._build_state()
    masks = env.mask_builder.build(state, env.qkp, env.requests)
    flat = env.mask_builder.last_flat_legal
    assert flat is not None
    assert flat.size == env.graph_builder._mask_total

    # Graph builder: flat-mask path must produce identical observations.
    obs_flat = env.graph_builder.build(
        state, env.requests, env.request_history, masks, flat_masks=flat
    )
    obs_dict = env.graph_builder.build(state, env.requests, env.request_history, masks)
    assert np.array_equal(obs_flat.node_features, obs_dict.node_features)
    assert np.array_equal(obs_flat.edge_index, obs_dict.edge_index)
    assert np.array_equal(obs_flat.edge_features, obs_dict.edge_features)
    assert obs_flat.action_candidates == obs_dict.action_candidates
    assert obs_flat.physical_edge_ids == obs_dict.physical_edge_ids
    assert obs_flat.flat_action_masks is flat

    # Actor plan: obs carrying the flat array must produce the identical
    # plan as the raw-mask path.
    plan_flat = model.actor._build_plan(obs_flat, env.action_resolver.action_space)
    plan_dict = model.actor._build_plan(obs_dict, env.action_resolver.action_space)
    for a, b in zip(plan_flat, plan_dict):
        assert isinstance(a, np.ndarray) == isinstance(b, np.ndarray)
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b)
        else:
            assert a == b


def test_mappo_policy_produces_env_compatible_actions():
    config = load_default_config(".")
    env = build_test_env(".")
    obs = env.reset()
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model)

    step = policy.act(obs, deterministic=True)
    next_obs, reward, terminated, truncated, info = env.step(
        step.actions,
        step.action_scores,
        edge_scores=step.edge_scores,
    )

    assert next_obs.node_ids == obs.node_ids
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "served_keys" in info


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU cache lifetime coverage")
def test_cuda_batched_rollout_does_not_retain_observation_tensors():
    """Rollout observations must not retain GPU tensors for the whole PPO buffer."""
    config = load_default_config(".")
    env = build_test_env(".")
    obs = env.reset()
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config).cuda()
    policy = MAPPOPolicy(model, "cuda")

    policy.act_batched([obs], deterministic=True)

    assert not hasattr(obs, "_tensors_cache")

