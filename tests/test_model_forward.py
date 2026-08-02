from __future__ import annotations

import torch

from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.env.factory import build_default_env, load_default_config
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


def test_graph_mappo_actor_critic_forward_matches_action_candidates():
    config = load_default_config(".")
    env = build_default_env(".")
    obs = env.reset()
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)

    output = model(obs)

    assert set(output.logits) == set(obs.node_ids)
    assert output.value.shape == torch.Size([])
    for node_id, logits in output.logits.items():
        assert logits.shape[0] == len(obs.action_candidates[node_id])


def test_mappo_policy_produces_env_compatible_actions():
    config = load_default_config(".")
    env = build_default_env(".")
    obs = env.reset()
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model)

    step = policy.act(obs, deterministic=True)
    next_obs, reward, terminated, truncated, info = env.step(step.actions, step.action_scores)

    assert next_obs.node_ids == obs.node_ids
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "served_keys" in info

