"""Measure Graph-MAPPO rollout and PPO update wall-clock time."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from qkd_rl.algos.mappo_trainer import MAPPOTrainer
from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    config = load_default_config(ROOT)
    for name in ("env_full.yaml", "graph_mappo.yaml", "train_mappo.yaml"):
        config = deep_merge(config, load_config([ROOT / "configs" / name]))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = deep_merge(
        config,
        {
            "runtime": {"device": device},
            "env": {"episode_steps": args.steps},
            "train": {
                "rollout_steps": args.steps,
                "episodes_per_update": args.episodes,
                "rollout_batch": True,
                "n_rollout_workers": 1,
                "ppo": {
                    "epochs": 1,
                    "minibatch_size": args.minibatch_size,
                    "batch_chunk": args.minibatch_size,
                },
                "logging": {"log_interval": 0, "checkpoint_interval": 1000000},
            },
            "project": {"output_dir": str(ROOT / "outputs" / "_diagnostic"), "run_name": "profile"},
        },
    )
    ConfigValidator().validate(config)

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config).to(device)
    trainer = MAPPOTrainer(env, MAPPOPolicy(model, device), config, config["project"]["output_dir"], device=device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    started = time.perf_counter()
    buffer = trainer.collect_rollout()
    synchronize(device)
    rollout_seconds = time.perf_counter() - started
    rollout_peak_mib = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0.0

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    stats = trainer.update(buffer)
    synchronize(device)
    update_seconds = time.perf_counter() - started

    memory = ""
    if device == "cuda":
        memory = (
            f" rollout_peak_cuda_mib={rollout_peak_mib:.1f}"
            f" update_peak_cuda_mib={torch.cuda.max_memory_allocated()/1024**2:.1f}"
        )
    print(f"device={device} steps={len(buffer)} rollout_s={rollout_seconds:.3f} update_s={update_seconds:.3f}{memory}")
    print(f"success_rate={stats.mean_success_rate:.4f} reward={stats.mean_reward:.3f}")


if __name__ == "__main__":
    main()
