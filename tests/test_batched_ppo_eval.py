"""Batched PPO evaluation must match per-step evaluation exactly.

``MAPPOPolicy.evaluate_actions_batched`` merges several graphs into one
block-diagonal forward. This is an implementation-level speedup: log probs,
entropies, critic values and gradients must match the per-step path within
floating point tolerance.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


def _build(device: str = "cpu"):
    config = load_default_config(ROOT)
    for name in ["env_full.yaml", "graph_mappo.yaml"]:
        config = deep_merge(config, load_config([ROOT / "configs" / name]))
    config["runtime"]["device"] = device
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config).to(device)
    policy = MAPPOPolicy(model, device)
    return env, policy


def _collect(env, policy, n: int, seed: int):
    obs = env.reset(seed=seed)
    pairs = []
    for _ in range(n):
        step = policy.act(obs)
        pairs.append((obs, step.actions, step.matched_edges))
        obs, *_ = env.step(step.actions, step.action_scores, edge_scores=step.edge_scores)
    return pairs


def test_env_full_max_weight_matching_runs():
    env, policy = _build("cpu")
    obs = env.reset(seed=0)
    step = policy.act(obs, deterministic=True)
    env.step(
        step.actions,
        step.action_scores,
        edge_scores=step.edge_scores,
    )
    assert set(env.last_activated_edges) <= set(obs.physical_edge_ids)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_batched_matches_single(device):
    env, policy = _build(device)
    pairs = _collect(env, policy, 6, seed=0)
    policy.model.train()
    for _ in range(2):
        for obs, actions, matched_edges in pairs:
            lp, ent, val = policy.evaluate_actions(obs, actions, matched_edges=matched_edges)
            (val + sum(lp.values()).sum()).backward()
        policy.model.zero_grad()

    single = [
        policy.evaluate_actions(obs, actions, matched_edges=matched_edges)
        for obs, actions, matched_edges in pairs
    ]
    batched = policy.evaluate_actions_batched(
        [obs for obs, _actions, _matched in pairs],
        [actions for _obs, actions, _matched in pairs],
        [matched for _obs, _actions, matched in pairs],
    )
    assert len(single) == len(batched)
    for (obs, _actions, _matched), (lp_s, ent_s, val_s), (lp_b, ent_b, val_b) in zip(
        pairs, single, batched
    ):
        assert set(lp_s) == set(lp_b)
        assert set(ent_s) == set(ent_b)
        for node_id in obs.node_ids:
            assert torch.allclose(lp_s[node_id], lp_b[node_id], atol=1.0e-4), node_id
            assert torch.allclose(ent_s[node_id], ent_b[node_id], atol=1.0e-4), node_id
        assert torch.allclose(val_s, val_b, atol=1.0e-3), (val_s, val_b)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_batched_gradients_match_single(device):
    env, policy = _build(device)
    pairs = _collect(env, policy, 4, seed=1)
    policy.model.train()

    def total_loss(policy, pairs, batched: bool):
        policy.model.zero_grad()
        loss = torch.zeros((), dtype=torch.float32, device=policy.device)
        if batched:
            results = policy.evaluate_actions_batched(
                [obs for obs, _actions, _matched in pairs],
                [actions for _obs, actions, _matched in pairs],
                [matched for _obs, _actions, matched in pairs],
            )
            for (_obs, _actions, _matched), (lp, _ent, val) in zip(pairs, results):
                loss = loss + val + sum(lp.values()).sum()
        else:
            for obs, actions, matched_edges in pairs:
                lp, _ent, val = policy.evaluate_actions(obs, actions, matched_edges=matched_edges)
                loss = loss + val + sum(lp.values()).sum()
        loss.backward()
        return {name: param.grad.detach().cpu().clone() for name, param in policy.model.named_parameters() if param.grad is not None}

    grads_single = total_loss(policy, pairs, batched=False)
    grads_batched = total_loss(policy, pairs, batched=True)
    assert set(grads_single) == set(grads_batched)
    for name in grads_single:
        assert torch.allclose(grads_single[name], grads_batched[name], atol=1.0e-3), name
