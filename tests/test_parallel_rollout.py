"""Tests for parallel rollout workers (spawn-safe on Windows)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer
from qkd_rl.rl.algos.policy import MAPPOPolicy
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
from tests.helpers import ROOT, build_test_env, point_config_to_h5

torch = pytest.importorskip("torch")


def _build_trainer(n_workers: int, device: str = "cpu", output_dir: Path | None = None):
    config = point_config_to_h5(__import__("qkd_rl.env.factory", fromlist=["load_default_config"]).load_default_config(ROOT))
    config["train"]["n_rollout_workers"] = n_workers
    config["train"]["rollout_steps"] = 10
    config["train"]["episodes_per_update"] = 2
    # Keep train()-based tests hermetic and fast: no checkpoints, no periodic
    # validation rollouts (240-step episodes would dominate the runtime).
    config["train"].setdefault("logging", {})["checkpoint_interval"] = 100000
    config["train"]["logging"]["eval_interval"] = 0
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


def test_parallel_collect_fills_rollout_debug(tmp_path) -> None:
    """The worker path must accumulate the same per-component telemetry as the
    single-process loop (this is what keeps rollout_debug.jsonl available with
    n_workers > 1 -- the historical reason the pin to 1 existed)."""
    serial = _build_trainer(n_workers=1, output_dir=tmp_path / "serial")
    parallel = _build_trainer(n_workers=2, output_dir=tmp_path / "parallel")

    buf_serial = serial.collect_rollout()
    buf_parallel = parallel.collect_rollout()

    # One accumulation per env step on both paths.
    assert serial._rollout_debug["steps"] == float(len(buf_serial.steps))
    assert parallel._rollout_debug["steps"] == float(len(buf_parallel.steps))
    assert parallel._rollout_debug["steps"] > 0
    # Identical key set, so jsonl consumers cannot tell the paths apart.
    assert set(parallel._rollout_debug) == set(serial._rollout_debug)
    parallel._rollout_pool.shutdown() if parallel._rollout_pool else None


def test_parallel_train_writes_rollout_debug(tmp_path) -> None:
    """End to end: a 2-worker run must write rollout_debug.jsonl with the same
    record keys as the single-process path."""
    serial = _build_trainer(n_workers=1, output_dir=tmp_path / "serial")
    serial.train(num_updates=1)
    parallel = _build_trainer(n_workers=2, output_dir=tmp_path / "parallel")
    parallel.train(num_updates=1)
    parallel._rollout_pool.shutdown() if parallel._rollout_pool else None

    row_s = json.loads((tmp_path / "serial" / "rollout_debug.jsonl").read_text(encoding="utf-8").splitlines()[0])
    row_p = json.loads((tmp_path / "parallel" / "rollout_debug.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row_p["steps"] > 0
    assert set(row_p) == set(row_s)
