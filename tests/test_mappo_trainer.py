"""Tests for the MAPPO trainer rollout and PPO update loop."""

from __future__ import annotations

import json

import pytest
import torch

from qkd_rl.algos.mappo_trainer import MAPPOTrainer
from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.core.config import deep_merge
from qkd_rl.env.factory import build_env_from_config, load_default_config
from tests.helpers import point_config_to_h5
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


def _tiny_config(tmp_path):
    config = point_config_to_h5(load_default_config("."))
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
    config = point_config_to_h5(load_default_config("."))
    env = build_env_from_config(config)
    obs = env.reset()
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model)

    step = policy.act(obs, deterministic=True)
    log_probs, entropies, value = policy.evaluate_actions(
        obs,
        step.actions,
        matched_edges=step.matched_edges,
    )

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
    # PPO actor loss may be negative when advantages are positive; only the
    # critic MSE and the entropy bonus are guaranteed non-negative.
    assert stats.critic_loss > 0.0
    assert stats.entropy >= 0.0
    assert torch.isfinite(torch.tensor(stats.actor_loss))
    assert torch.isfinite(torch.tensor(stats.critic_loss))
    assert torch.isfinite(torch.tensor(stats.kl))

    params_after = [p.detach().clone() for p in model.parameters()]
    changed = any(not torch.equal(a, b) for a, b in zip(params_before, params_after))
    assert changed


def test_train_writes_rollout_debug(tmp_path):
    config = _tiny_config(tmp_path)
    config["train"]["num_updates"] = 1
    config["train"]["n_rollout_workers"] = 1
    config["train"]["rollout_batch"] = False
    config["train"]["logging"]["checkpoint_interval"] = 100000
    config["train"]["logging"]["eval_interval"] = 0
    env = build_env_from_config(config)
    trainer = MAPPOTrainer(env, MAPPOPolicy(GraphMAPPOActorCritic(env.action_resolver.action_space, config)), config, tmp_path)
    trainer.train()

    debug_path = tmp_path / "rollout_debug.jsonl"
    assert debug_path.exists()
    row = json.loads(debug_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["steps"] > 0
    assert row["mean_activated_edges"] >= 0
    assert row["mean_reward_served"] is not None
    assert row["actor_grad_norm"] >= 0


def test_collect_rollout_stores_sampled_matching_log_prob(tmp_path):
    """The rollout must keep the sampled matching order and its log-prob,
    not re-derive them from the resolver's output order."""
    config = _tiny_config(tmp_path)
    config["action_resolver"]["mode"] = "mutual_choice"
    config["train"]["n_rollout_workers"] = 1
    config["train"]["rollout_batch"] = False
    env_manual = build_env_from_config(config)
    env_trainer = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env_trainer.action_resolver.action_space, config)
    policy = MAPPOPolicy(model)
    trainer = MAPPOTrainer(env_trainer, policy, config, tmp_path)

    obs = env_manual.reset(seed=int(config["seed"]["env_seed"]))
    torch.manual_seed(123)
    step = policy.act(obs)
    env_manual.step(step.actions, step.action_scores, edge_scores=step.edge_scores)

    torch.manual_seed(123)
    buffer = trainer.collect_rollout()
    first = buffer.steps[0]
    assert first.matched_edges == list(step.matched_edges or [])
    torch.testing.assert_close(first.joint_log_prob, step.joint_log_prob)
    torch.testing.assert_close(first.joint_entropy, step.joint_entropy)


def test_trainer_rejects_unsupported_resolver_mode(tmp_path):
    config = _tiny_config(tmp_path)
    config["action_resolver"]["mode"] = "greedy_rate_matching"
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    with pytest.raises(ValueError, match="incompatible with the global matching policy"):
        MAPPOTrainer(env, MAPPOPolicy(model), config, tmp_path)


def test_actor_parameters_learn_from_ppo_update(tmp_path):
    """The joint matching log-prob must back-propagate into the actor.

    Regression: `_fill_node_tensors` used to detach the joint log prob during
    PPO evaluation, so the actor head received no gradient (KL stayed ~0 and
    the policy never left its random initialization).
    """
    config = _tiny_config(tmp_path)
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model)
    trainer = MAPPOTrainer(env, policy, config, tmp_path)

    actor_params_before = [p.detach().clone() for p in model.actor.parameters()]
    buffer = trainer.collect_rollout()
    trainer.update(buffer)
    actor_params_after = [p.detach().clone() for p in model.actor.parameters()]

    changed = any(not torch.equal(a, b) for a, b in zip(actor_params_before, actor_params_after))
    assert changed, "actor parameters did not change after a PPO update"


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
    assert lines[0]["rollout_s"] >= 0.0
    assert lines[0]["update_s"] >= 0.0
    assert lines[0]["elapsed_s"] >= lines[0]["rollout_s"] + lines[0]["update_s"]
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


def test_update_trains_critic_on_raw_returns(tmp_path):
    config = _tiny_config(tmp_path)
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model)
    trainer = MAPPOTrainer(env, policy, config, tmp_path)

    buffer = trainer.collect_rollout()
    stats = trainer.update(buffer)
    # The critic target is the raw return; Huber loss keeps it bounded.
    assert all(step.returns is not None for step in buffer.steps)
    assert stats.critic_loss < 50.0


def test_update_with_replay_steps(tmp_path):
    config = _tiny_config(tmp_path)
    config["train"]["replay_days"] = 1
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    trainer = MAPPOTrainer(env, MAPPOPolicy(model), config, tmp_path)

    buffer1 = trainer.collect_rollout()
    trainer.update(buffer1)
    trainer._remember_replay(buffer1)

    buffer2 = trainer.collect_rollout()
    stats = trainer.update(buffer2)
    assert stats.actor_loss is not None


def test_target_kl_stops_all_remaining_ppo_epochs(tmp_path):
    """A KL breach must stop the update, not merely one epoch's minibatches."""
    config = _tiny_config(tmp_path)
    config["train"]["ppo"]["epochs"] = 3
    config["train"]["ppo"]["target_kl"] = 0.01
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    trainer = MAPPOTrainer(env, MAPPOPolicy(model), config, tmp_path)

    class Buffer:
        steps = []

        def __init__(self):
            self.sample_calls = 0

        def sample(self, _minibatch_size, _rng):
            self.sample_calls += 1
            return [[object()], [object()]]

    buffer = Buffer()
    parameter = next(model.parameters())

    def high_kl_loss(*_args, **_kwargs):
        zero = parameter.sum() * 0.0
        return zero, zero, zero, torch.tensor(1.0), zero

    trainer._loss_for_batch = high_kl_loss
    trainer.update(buffer)

    assert buffer.sample_calls == 1


def test_curriculum_switches_rollout_length_and_episode_count(tmp_path):
    config = _tiny_config(tmp_path)
    config["train"]["curriculum"] = {
        "stages": [
            {"until_update": 10, "rollout_steps": 40, "episodes_per_update": 2},
            {"until_update": 20, "rollout_steps": 80, "episodes_per_update": 4},
            {"until_update": 30, "rollout_steps": 120, "episodes_per_update": 1},
        ]
    }
    env = build_env_from_config(config)
    trainer = MAPPOTrainer(env, MAPPOPolicy(GraphMAPPOActorCritic(env.action_resolver.action_space, config)), config, tmp_path)

    trainer._rollout_envs = ["stale"] * 2
    trainer.episodes_per_update = 2
    trainer.update_count = 0
    trainer._apply_curriculum()
    assert trainer.rollout_steps == 40
    assert trainer.episodes_per_update == 2
    assert trainer._rollout_envs is not None

    trainer._rollout_envs = ["stale"] * 4
    trainer.update_count = 10
    trainer._apply_curriculum()
    assert trainer.rollout_steps == 80
    assert trainer.episodes_per_update == 4
    assert trainer._rollout_envs is None  # episode count changed -> rebuild

    trainer.update_count = 20
    trainer._apply_curriculum()
    assert trainer.rollout_steps == 120
    assert trainer.episodes_per_update == 1

    trainer.update_count = 25
    trainer._apply_curriculum()
    assert trainer.rollout_steps == 120  # beyond last stage keeps final stage


def test_checkpoint_reward_change_resets_critic_head(tmp_path):
    config = _tiny_config(tmp_path)
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    trainer = MAPPOTrainer(env, MAPPOPolicy(model), config, tmp_path)
    ckpt = trainer.save_checkpoint(tmp_path / "ckpt.pt")
    head_before = [p.detach().clone() for p in model.critic.value_head.parameters()]

    # Different reward config -> the stale critic head must be re-initialized.
    # env_small.yaml already defaults to served_weight=5.0; use a value that
    # is actually different so the critic-head reset branch is exercised.
    config2 = deep_merge(config, {"reward": {"served_weight": 3.0}})
    model2 = GraphMAPPOActorCritic(env.action_resolver.action_space, config2)
    trainer2 = MAPPOTrainer(env, MAPPOPolicy(model2), config2, tmp_path)
    trainer2.load_checkpoint(ckpt)
    head_after = [p.detach().clone() for p in model2.critic.value_head.parameters()]
    assert not all(torch.equal(a, b) for a, b in zip(head_before, head_after))

    # Identical reward config -> the critic head is preserved.
    model3 = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    trainer3 = MAPPOTrainer(env, MAPPOPolicy(model3), config, tmp_path)
    trainer3.load_checkpoint(ckpt)
    head_same = [p.detach().clone() for p in model3.critic.value_head.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(head_before, head_same))
