"""Run the trained RL model over the full fixed-start window (same protocol
as scripts/compute_milp_reference.py) and write steps.csv + summary.json so
the RL agent can be compared directly with outputs/milp_reference/.

Usage:
    python scripts/eval_rl_reference.py --checkpoint outputs/milp_ref_baseline_cmp/checkpoint_final.pt
        [--out outputs/eval/rl_30day] [--seed 0]
"""
from __future__ import annotations

import os

# Conda/matplotlib on Windows loads libiomp5md.dll twice; this documented
# workaround prevents the OMP Error #15 crash at import/exit.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.algos.checkpoint import load_checkpoint
from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.core.config import deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic

STEP_FIELDS = [
    "step", "t", "reward", "served_keys", "generated_keys", "failed_keys",
    "waiting_keys", "pending_count", "n_activated",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained RL model over the full window.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=str, default="outputs/eval/rl_30day")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = full window")
    parser.add_argument("--progress", type=int, default=1000)
    args = parser.parse_args()

    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config = deep_merge(config, load_config([ROOT / "configs" / "baselines.yaml"]))
    config["rate_provider"]["provider"] = "h5"
    config["env"]["episode_start_mode"] = "fixed"
    config["env"]["episode_steps"] = int(10 ** 9)

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, device)
    data = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(data.model_state)
    model.eval()
    print(f"rl model: {args.checkpoint} (update={data.update}) device={device}")

    out_dir = Path(ROOT) / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    obs = env.reset(seed=args.seed)
    total_reward = 0.0
    arrived_keys = 0.0
    step_rows = []
    done = False
    step = 0
    while not done:
        act = policy.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(
            act.actions,
            act.action_scores,
            edge_scores=act.edge_scores,
            expected_matched_edges=list(act.matched_edges or []),
        )
        total_reward += reward
        step_rows.append({
            "step": step,
            "t": env.t - 1,
            "reward": round(float(reward), 6),
            "served_keys": round(float(info.get("served_keys", 0.0)), 3),
            "generated_keys": round(float(info.get("generated_keys", 0.0)), 3),
            "failed_keys": round(float(info.get("failed_keys", 0.0)), 3),
            "waiting_keys": round(float(info.get("waiting_keys", 0.0)), 3),
            "pending_count": int(info.get("pending_count", 0)),
            "n_activated": int(info.get("n_activated", 0)),
        })
        step += 1
        if args.max_steps and step >= args.max_steps:
            break
        done = terminated or truncated
        if args.progress and step % args.progress == 0:
            print(f"step={step} t={env.t} reward={reward:.1f}", flush=True)
    summary = env.metrics.episode_summary()
    with (out_dir / "steps.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STEP_FIELDS)
        writer.writeheader()
        writer.writerows(step_rows)
    result = {
        "policy": "rl_model",
        "checkpoint": args.checkpoint,
        "checkpoint_update": data.update,
        "seed": args.seed,
        "steps": summary["steps"],
        "total_reward": round(total_reward, 3),
        "arrived_keys": summary["arrived_keys"],
        "served_keys": summary["served_keys"],
        "failed_keys": summary["failed_keys"],
        "success_rate": summary["success_rate"],
        "arrived_requests": summary["arrived_requests"],
        "completed_requests": summary["completed_requests"],
        "request_completion_rate": summary["request_completion_rate"],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2))
    print(f"written: {out_dir / 'steps.csv'}, {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
