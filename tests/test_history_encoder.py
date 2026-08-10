"""HistoryEncoder integration: cold-start padding, model forward, short training."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from qkd_rl.algos.mappo_trainer import MAPPOTrainer
from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.core.config import ConfigValidator
from qkd_rl.core.types import KeyRequest
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic
from tests.helpers import ROOT, point_config_to_h5


def _enabled_config():
    config = load_default_config(ROOT)
    config["features"]["history_encoder"]["enabled"] = True
    config["features"]["history_encoder"]["hidden_dim"] = 32
    config["features"]["history_encoder"]["seq_len"] = 16
    config["features"]["demand_edge"]["wait_bucket_count"] = 4
    config["train"]["rollout_steps"] = 6
    config["train"]["episodes_per_update"] = 1
    config["train"]["ppo"]["epochs"] = 1
    ConfigValidator().validate(config)
    return point_config_to_h5(config)


def test_history_dim_zero_when_disabled():
    config = load_default_config(ROOT)
    assert config["features"]["dims"]["history_dim_resolved"] == 0


def test_history_encoder_cold_start_and_forward():
    config = _enabled_config()
    assert config["features"]["dims"]["history_dim_resolved"] == 32
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model)

    obs = env.reset()
    # Cold start: windows exist but nothing observed yet.
    assert obs.node_history == []
    assert obs.physical_edge_history == []
    assert obs.demand_edge_history == []
    output = model(obs)
    assert output.value.shape == torch.Size([])

    big = KeyRequest("REQ_LSTM", "GS_001", "GS_002", 1.0e9, env.t, env.t + 100)
    env.requests.add_arrivals([big])
    for step_idx in range(4):
        step = policy.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(
            step.actions,
            step.action_scores,
            edge_scores=step.edge_scores,
        )
        our_idx = obs.demand_edge_ids.index("D_GS_001__GS_002")
        assert len(obs.demand_edge_history) >= 1
        assert len(obs.demand_edge_history[our_idx]) == 16
        expected = min(step_idx + 1, 16)
        assert obs.history_valid[2][our_idx] == expected
        assert torch.isfinite(output.value)


def test_history_encoder_short_training():
    config = _enabled_config()
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = MAPPOTrainer(env, policy, config, Path(tmp) / "out", device="cpu")
        result = trainer.train(num_updates=1)
    stats = result["last_stats"]
    assert torch.isfinite(torch.tensor(stats["actor_loss"]))
    assert torch.isfinite(torch.tensor(stats["critic_loss"]))
    assert stats["critic_loss"] > 0.0
