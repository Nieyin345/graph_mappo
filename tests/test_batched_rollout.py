"""Tests for the lockstep batched rollout (throughput optimization).

The batched path must be mathematically identical to the serial path: same
sampling distribution, same buffer contents. For a single graph the batched
logits are bit-identical to the single forward (block-diagonal batching of one
graph is a no-op); extra graphs in the batch may differ by ~1e-7 from
floating-point ordering, so exact-equality assertions are limited to the
single-graph case and the deterministic path.
"""

from __future__ import annotations

import pytest

from qkd_rl.algos.mappo_trainer import MAPPOTrainer
from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.core.config import deep_merge
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic
from tests.helpers import ROOT, point_config_to_h5

torch = pytest.importorskip("torch")


def _config(tmp_path, episodes: int, steps: int):
    config = point_config_to_h5(load_default_config(ROOT))
    config = deep_merge(
        config,
        {
            "env": {"episode_steps": 40},
            "train": {
                "num_updates": 1,
                "rollout_steps": steps,
                "episodes_per_update": episodes,
                "n_rollout_workers": 1,
                "rollout_batch": True,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "value_target": "mc",
                "ppo": {"epochs": 1, "minibatch_size": 32, "clip_eps": 0.2},
                "logging": {"checkpoint_interval": 10, "log_interval": 0},
            },
            "project": {"output_dir": str(tmp_path), "run_name": "test"},
        },
    )
    return config


def _trainer(config, device="cpu"):
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config).to(device)
    policy = MAPPOPolicy(model, device)
    return MAPPOTrainer(env, policy, config, config["project"]["output_dir"], device=device)


def test_act_batched_deterministic_matches_act() -> None:
    config = point_config_to_h5(load_default_config(ROOT))
    env = build_env_from_config(config)
    obs = env.reset(seed=1)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model)

    step = policy.act(obs, deterministic=True)
    batched = policy.act_batched([obs, obs], deterministic=True)
    for out in batched:
        assert out.actions == step.actions
        for node_id in obs.node_ids:
            torch.testing.assert_close(out.log_probs[node_id], step.log_probs[node_id])
            torch.testing.assert_close(out.entropies[node_id], step.entropies[node_id])
        torch.testing.assert_close(out.value, step.value)


def test_act_batched_stochastic_single_graph_matches_act() -> None:
    """Same RNG state -> single-graph batched sampling equals ``act`` exactly."""
    config = point_config_to_h5(load_default_config(ROOT))
    env = build_env_from_config(config)
    obs = env.reset(seed=1)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model)

    torch.manual_seed(0)
    step = policy.act(obs)
    torch.manual_seed(0)
    batched = policy.act_batched([obs])

    assert step.actions == batched[0].actions
    for node_id in obs.node_ids:
        torch.testing.assert_close(step.log_probs[node_id], batched[0].log_probs[node_id])
        torch.testing.assert_close(step.entropies[node_id], batched[0].entropies[node_id])
    torch.testing.assert_close(step.value, batched[0].value)


def test_batched_single_episode_matches_serial_exactly(tmp_path) -> None:
    """With one episode the batched path must reproduce the serial buffer."""
    cfg_b = _config(tmp_path, episodes=1, steps=10)
    cfg_s = _config(tmp_path / "s", episodes=1, steps=10)
    # One shared model: the two paths must produce identical trajectories for
    # the same seeds, so the weights must not differ between the trainers.
    env_b = build_env_from_config(cfg_b)
    model = GraphMAPPOActorCritic(env_b.action_resolver.action_space, cfg_b)
    policy = MAPPOPolicy(model)
    batched = MAPPOTrainer(env_b, policy, cfg_b, cfg_b["project"]["output_dir"])
    env_s = build_env_from_config(cfg_s)
    serial = MAPPOTrainer(env_s, policy, cfg_s, cfg_s["project"]["output_dir"])

    torch.manual_seed(123)
    buf_b = batched.collect_rollout()
    torch.manual_seed(123)
    buf_s = serial.collect_rollout()

    assert len(buf_b.steps) == len(buf_s.steps) == 10
    for s_b, s_s in zip(buf_b.steps, buf_s.steps):
        assert s_b.actions == s_s.actions
        assert s_b.reward == pytest.approx(s_s.reward)
        for node_id in s_s.log_probs:
            torch.testing.assert_close(s_b.log_probs[node_id], s_s.log_probs[node_id])
            torch.testing.assert_close(s_b.entropies[node_id], s_s.entropies[node_id])
        torch.testing.assert_close(s_b.value, s_s.value)
        torch.testing.assert_close(s_b.returns, s_s.returns)
        torch.testing.assert_close(s_b.advantages, s_s.advantages)


def test_batched_multi_episode_collects_all_episodes(tmp_path) -> None:
    """Two lockstep episodes: both collected, per-episode returns computed."""
    cfg = _config(tmp_path, episodes=2, steps=10)
    trainer = _trainer(cfg)
    buf = trainer.collect_rollout()

    assert len(buf.steps) == 20
    assert len(trainer.last_episode_rewards) == 2
    assert len(trainer.last_episode_summaries) == 2
    for step in buf.steps:
        assert step.returns is not None and step.advantages is not None
        assert step.value.dim() == 0
