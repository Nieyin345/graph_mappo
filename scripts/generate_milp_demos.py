"""Generate MILP demonstration trajectories for behavior cloning / fine-tuning.

Each run adds `--episodes N` demos (default 5) with random seeds and random
start times. Demo files are stored under outputs/milp_demos/ as individual .pt
files so you can keep adding more runs:

    python scripts/generate_milp_demos.py                        # 5 demos
    python scripts/generate_milp_demos.py --episodes 3 --append  # +3 → 8 total

Each .pt file contains a full trajectory: per-step node/edge features, MILP
actions, masks, and final success rate for quality inspection.
"""
from __future__ import annotations
import os, sys, time, json, random
from pathlib import Path
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
from qkd_rl.core.config import (ConfigValidator, deep_merge, load_config)
from qkd_rl.env.factory import load_default_config, build_env_from_config
from qkd_rl.baselines.receding_horizon_milp import RecedingHorizonMILPPolicy


def _strip_obs(obs, actions):
    """Extract the minimal observable state needed for actor forward."""
    return {
        "node_features": obs.node_features,
        "edge_features": obs.edge_features,
        "edge_index": obs.edge_index,
        "edge_ids": obs.edge_ids,
        "node_ids": obs.node_ids,
        "physical_edge_ids": obs.physical_edge_ids,
        "action_candidates": obs.action_candidates,
        "action_masks": obs.action_masks,
        "raw_action_masks": obs.raw_action_masks,
        "milp_actions": actions,
        "t": int(obs.state.t),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="Number of demos to generate this run.")
    parser.add_argument("--append", action="store_true", help="Append to existing demo set instead of overwriting.")
    parser.add_argument("--steps", type=int, default=240, help="Steps per episode.")
    parser.add_argument("--window-steps", type=int, default=120, help="MILP window size.")
    parser.add_argument("--time-limit", type=float, default=15.0, help="MILP solve time limit per window.")
    parser.add_argument("--out", type=str, default="outputs/milp_demos")
    args = parser.parse_args()

    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config = deep_merge(config, load_config([ROOT / "configs" / "global.yaml"]))
    config["env"]["episode_steps"] = args.steps
    config["scenario"]["time_limit"]["days"] = 366
    ConfigValidator().validate(config)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.json"

    # Load existing metadata for append mode
    existing = 0
    if args.append and metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing = meta.get("total_episodes", 0)
        print(f"Appending to existing: {existing} episodes already saved.")

    pol = RecedingHorizonMILPPolicy(
        config, window_steps=args.window_steps, max_requests=512,
        max_paths_per_request=256, max_path_hops=10, time_limit_s=args.time_limit,
    )

    env = build_env_from_config(config)
    results = []
    for ep_idx in range(args.episodes):
        ep_num = existing + ep_idx
        seed = 7 + ep_num
        rng = random.Random(seed + 1000)
        start_seed = rng.randint(0, 100000)
        obs = env.reset(seed=seed, start_seed=start_seed)

        pol._ensure_init(obs)
        pol._gen_t = int(obs.state.t)

        trajectory = []
        total_served = 0.0
        total_arrived = 0.0
        t0 = time.perf_counter()
        for step in range(args.steps):
            actions, scores = pol.act(obs)
            trajectory.append(_strip_obs(obs, actions))
            obs, reward, term, trunc, info = env.step(actions, scores)
            total_served += info.get("served_keys", 0.0)
            total_arrived += info.get("arrived_keys", 0.0)
            if term or trunc:
                break
        elapsed = time.perf_counter() - t0
        summary = env.metrics.episode_summary()
        sr = summary["success_rate"]
        print(f"episode {ep_num}: seed={seed} start_seed={start_seed} "
              f"steps={len(trajectory)} SR={sr:.4f} served={summary['served_keys']:,.0f} "
              f"({elapsed:.0f}s)", flush=True)

        # Save individual episode
        demo = {
            "seed": seed,
            "start_seed": start_seed,
            "steps": len(trajectory),
            "success_rate": sr,
            "served_keys": summary["served_keys"],
            "arrived_keys": summary["arrived_keys"],
            "failed_keys": summary["failed_keys"],
            "trajectory": trajectory,
        }
        ep_path = out_dir / f"episode_{ep_num:04d}.pt"
        torch.save(demo, ep_path)
        results.append({
            "file": ep_path.name,
            "episode": ep_num,
            "seed": seed,
            "steps": len(trajectory),
            "success_rate": sr,
            "served_keys": summary["served_keys"],
            "arrived_keys": summary["arrived_keys"],
        })

    total = existing + args.episodes
    meta = {
        "total_episodes": total,
        "episodes_per_run": args.episodes,
        "steps_per_episode": args.steps,
        "window_steps": args.window_steps,
        "time_limit_s": args.time_limit,
        "all_episodes": results,
    }
    metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Total MILP demos: {total}")
    print(f"Location: {out_dir}/")
    print(f"Run again with --append to add more.")


if __name__ == "__main__":
    main()