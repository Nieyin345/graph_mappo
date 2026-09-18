"""Evaluate a checkpoint on the fixed smoke scenario (day 0, fixed seeds).

Re-usable trend probe for the fixed-scenario smoke run: builds the env from a
train config (smoke profile), runs N deterministic episodes of ``--steps``,
and prints per-seed served/arrived success rates plus the average. Same
settings as the smoke rollout (fixed_episode_seed, episode_start_day=0), so
the numbers are directly comparable with the per-update training
success_rate in metrics.jsonl.

Usage:
    python scripts/train/eval_fixed_scenario.py \
        --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
        --steps 720 --seeds 7-14 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qkd_rl.core.config import deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.rl.algos.checkpoint import load_checkpoint
from qkd_rl.rl.algos.policy import MAPPOPolicy
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic


def parse_range(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/train_mappo_smoke.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--seeds", type=str, default="7-14")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_range(args.seeds)
    base = load_default_config(ROOT)
    # Same config chain as train_graph_mappo.build_config so the env matches
    # the training rollout exactly: full H5 scenario + random_episode profile
    # + global window + the smoke override file.
    base = deep_merge(base, load_config([ROOT / "configs" / "env_full.yaml"]))
    profiles = load_config([ROOT / "configs" / "train_profiles.yaml"]).get("train_profiles", {})
    base = deep_merge(base, profiles.get("random_episode", {}))
    base = deep_merge(base, load_config([ROOT / "configs" / "global.yaml"]))
    cfg = deep_merge(base, load_config([ROOT / args.config]))
    cfg["train"]["rollout_steps"] = args.steps
    env = build_env_from_config(cfg)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
    data = load_checkpoint(args.checkpoint, device=args.device)
    model.load_state_dict(data.model_state)
    policy = MAPPOPolicy(model, args.device)
    model.eval()

    rates: list[float] = []
    for seed in seeds:
        obs = env.reset(seed=seed)
        served = 0.0
        steps = 0
        while steps < args.steps:
            with torch.no_grad():
                step = policy.act(obs)
            obs, _reward, terminated, truncated, info = env.step(
                step.actions,
                step.action_scores,
                edge_scores=step.edge_scores,
                expected_matched_edges=list(step.matched_edges or []),
            )
            served += float(info.get("served_keys", 0.0))
            steps += 1
            if terminated or truncated:
                break
        arrived = float(env.metrics.arrived_keys)
        sr = served / arrived if arrived > 0 else 0.0
        rates.append(sr)
        print(f"seed={seed:>3d}  success_rate={sr:.4f}  served={served:.0f}  arrived={arrived:.0f}", flush=True)
    if rates:
        print(f"AVG over seeds {seeds[0]}-{seeds[-1]}: {sum(rates) / len(rates):.4f}", flush=True)


if __name__ == "__main__":
    main()
