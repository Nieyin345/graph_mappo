"""Frozen evaluation with expert/model actions interleaved at fixed intervals."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qkd_rl.core.config import load_config
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.model_zoo import build_model
from qkd_rl.rl.algos.checkpoint import load_checkpoint
from qkd_rl.rl.algos.policy import MAPPOPolicy
from scripts.diag.probe_v3_counterfactual import expert_for
from scripts.eval.frozen_eval import EPISODE_STEPS, WINDOWS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--window", choices=WINDOWS, default="A")
    ap.add_argument("--periods", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    config = load_config([Path(args.run) / "experiment_config.yaml"])
    spec = WINDOWS[args.window]
    config["env"].update({
        "episode_start_mode": "random_day", "episode_steps": EPISODE_STEPS,
        "continuous": False, "activation_window_start_day": spec["start_day"],
        "activation_window_end_day": spec["end_day"],
        "activation_window_days": spec["end_day"] - spec["start_day"],
    })
    config["scenario"]["time_limit"]["days"] = spec["end_day"] + max(
        1, math.ceil(EPISODE_STEPS / int(config["env"].get("day_steps", 1440)))
    )
    env = build_env_from_config(config)
    model = build_model(env.action_resolver.action_space, config)
    model.load_state_dict(load_checkpoint(args.checkpoint, "cpu").model_state)
    model.eval()
    policy = MAPPOPolicy(model, "cpu")
    results = {}
    for period in args.periods:
        if period < 1:
            raise ValueError("period must be positive")
        per_seed = []
        for seed in spec["request_seeds"]:
            expert = expert_for(env)
            obs = env.reset(seed=seed, start_seed=seed)
            for step in range(EPISODE_STEPS):
                if step % period == 0:
                    with torch.no_grad():
                        result = policy.act(obs, deterministic=True, build_scores=False)
                    obs, _, _, _, _ = env.step(
                        result.actions, result.action_scores,
                        expected_matched_edges=list(result.matched_edges or []),
                    )
                else:
                    actions, scores = expert.act(obs)
                    obs, _, _, _, _ = env.step(actions, scores)
            summary = env.metrics.episode_summary()
            per_seed.append({
                "seed": seed,
                "success_rate": float(summary["success_rate"]),
                "served_keys": float(summary["served_keys"]),
                "generated_keys": float(summary["generated_keys"]),
            })
        mean = sum(row["success_rate"] for row in per_seed) / len(per_seed)
        results[str(period)] = {"mean_success_rate": mean, "per_seed": per_seed}
        print(f"period={period} mean_success_rate={mean:.6f}", flush=True)
    Path(args.out).write_text(json.dumps({
        "window": args.window, "checkpoint": args.checkpoint, "results": results,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
