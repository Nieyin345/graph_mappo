"""Evaluate one or more RL checkpoints on a fixed first-day window.

Usage:
    python scripts/eval_checkpoints_quick.py outputs/a/checkpoint_final.pt \
        outputs/b/checkpoint_final.pt --seed 1234 --out outputs/eval/quick_cmp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.algos.checkpoint import load_checkpoint
from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.core.config import deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config = deep_merge(config, load_config([ROOT / "configs" / "baselines.yaml"]))
    config["rate_provider"]["provider"] = "h5"
    config["env"]["episode_start_mode"] = "fixed"
    config["env"]["episode_steps"] = int(10**9)
    config["scenario"]["time_limit"]["days"] = args.days

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = build_env_from_config(config)
    results = {}
    for ckpt_path in args.checkpoints:
        ckpt = Path(ckpt_path)
        model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
        policy = MAPPOPolicy(model, device)
        data = load_checkpoint(ckpt, device)
        model.load_state_dict(data.model_state)
        model.eval()
        obs = env.reset(seed=args.seed)
        done = False
        total_reward = 0.0
        while not done:
            step = policy.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(
                step.actions,
                step.action_scores,
                edge_scores=step.edge_scores,
                expected_matched_edges=list(step.matched_edges or []),
            )
            total_reward += reward
            done = terminated or truncated
        summary = env.metrics.episode_summary()
        result = {
            "checkpoint": str(ckpt),
            "update": data.update,
            "seed": args.seed,
            "total_reward": round(float(total_reward), 3),
            **{k: round(float(v), 6) if isinstance(v, float) else v for k, v in summary.items()},
        }
        results[str(ckpt)] = result
        print(json.dumps(result, ensure_ascii=False))
    if args.out:
        out_path = Path(ROOT) / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"written: {out_path}")


if __name__ == "__main__":
    main()
