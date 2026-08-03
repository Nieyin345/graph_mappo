"""Tests for the MAPPO trainer rollout and PPO update loop."""

from __future__ import annotations

import json

import torch

from qkd_rl.algos.mappo_trainer import MAPPOTrainer
from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.core.config import deep_merge
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


def _tiny_config(tmp_path):
    config = load_default_config(".")
    config = deep_merge(
        config,
        {
            "env": {"episode_steps": 40},
            "train": {
                "num_updates": 2,
                "rollout_steps": 40,
                "episodes_per_update": 1,
                "ppo": {"epochs": 2, "minibatch_size": 32, "clip_eps": 0.2},
                "logging": {"checkpoint_interval": 1, "log_interval": 0},
            },
            "project": {"output_dir": str(tmp_path), "run_name": "test"},
        },
    )
    return config


def test_evaluate_actions_matches_act_log_probs():
    config = load_default_config(".")
    env = build_env_from_config(config)
    obs = env.reset()
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model)

    step = policy.act(obs, deterministic=True)
    log_probs, entropies, value = policy.evaluate_actions(obs, step.actions)

    for node_id in obs.node_ids:
        torch.testing.assert_close(log_probs[node_id], step.log_probs[node_id])
        assert entropies[node_id].shape == torch.Size([])
    assert value.shape == torch.Size([])


def test_collect_rollout_and_update(tmp_path):
    config = _tiny_config(tmp_path)
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model)
    trainer = MAPPOTrainer(env, policy, config, tmp_path)

    buffer = trainer.collect_rollout()
    assert len(buffer) == 40

    params_before = [p.detach().clone() for p in model.parameters()]
    stats = trainer.update(buffer)
    assert stats.actor_loss > 0.0
    assert stats.critic_loss > 0.0
    assert stats.entropy > 0.0
    assert torch.isfinite(torch.tensor(stats.actor_loss))
    assert torch.isfinite(torch.tensor(stats.kl))

    params_after = [p.detach().clone() for p in model.parameters()]
    changed = any(not torch.equal(a, b) for a, b in zip(params_before, params_after))
    assert changed


def test_train_writes_log_and_checkpoints(tmp_path):
    config = _tiny_config(tmp_path)
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model)
    trainer = MAPPOTrainer(env, policy, config, tmp_path)

    trainer.train(num_updates=2)

    assert trainer.update_count == 2
    log_path = tmp_path / "metrics.jsonl"
    assert log_path.exists()
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 2
    assert (tmp_path / "checkpoint_update_000001.pt").exists()

    eval_stats = trainer.evaluate(num_episodes=1)
    assert set(eval_stats) == {"mean_reward", "mean_success_rate", "mean_served_keys"}

def test_checkpoint_roundtrip(tmp_path):
    config = _tiny_config(tmp_path)
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model)
    trainer = MAPPOTrainer(env, policy, config, tmp_path)
    trainer.update_count = 5

    ckpt_path = trainer.save_checkpoint(tmp_path / "ckpt.pt")
    assert ckpt_path.exists()

    model2 = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    trainer2 = MAPPOTrainer(env, MAPPOPolicy(model2), config, tmp_path)
    trainer2.load_checkpoint(ckpt_path)
    assert trainer2.update_count == 5
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        torch.testing.assert_close(p1.detach(), p2.detach())
