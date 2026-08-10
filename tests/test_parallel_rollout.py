"""Tests for parallel rollout workers (spawn-safe on Windows)."""

from __future__ import annotations

from pathlib import Path

import pytest

from qkd_rl.algos.mappo_trainer import MAPPOTrainer
from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic
from tests.helpers import ROOT, build_test_env, point_config_to_h5

torch = pytest.importorskip("torch")


def _build_trainer(n_workers: int, device: str = "cpu", output_dir: Path | None = None):
    config = point_config_to_h5(__import__("qkd_rl.env.factory", fromlist=["load_default_config"]).load_default_config(ROOT))
    config["train"]["n_rollout_workers"] = n_workers
    config["train"]["rollout_steps"] = 10
    config["train"]["episodes_per_update"] = 2
    config["runtime"]["device"] = device
    env = build_test_env(ROOT)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config).to(device)
    policy = MAPPOPolicy(model, device)
    trainer = MAPPOTrainer(
        env,
        policy,
        config,
        output_dir or ROOT / "outputs" / "_test_parallel",
        device=device,
    )
    return trainer


def test_parallel_collect_matches_serial_step_count(tmp_path) -> None:
    serial = _build_trainer(n_workers=1, output_dir=tmp_path / "serial")
    parallel = _build_trainer(n_workers=2, output_dir=tmp_path / "parallel")

    buf_serial = serial.collect_rollout()
    buf_parallel = parallel.collect_rollout()

    assert len(buf_serial.steps) > 0
    assert len(buf_parallel.steps) == len(buf_serial.steps)
    for step in buf_parallel.steps:
        assert step.returns is not None and step.advantages is not None
        assert step.value.dim() == 0
    assert len(parallel.last_episode_rewards) == 2
    parallel._rollout_pool.shutdown() if parallel._rollout_pool else None
