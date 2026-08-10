"""Compute the MILP-optimal reference trajectory on the unified scenario.

Builds the unified scenario (configs/env_full.yaml, simplified by
scenario.time_limit + scenario.active_nodes), then steps the environment
for the whole time window with the per-slot MILP-optimal policy and stores
the reference trajectory so the RL agent can be compared against the
optimal per-step decisions:

    outputs/milp_reference/
        metadata.json      scenario + MILP parameters snapshot
        steps.csv          per-step optimal actions and outcomes
        summary.json       aggregate totals for the window

Usage:
    python scripts/compute_milp_reference.py [--seed 0] [--out outputs/milp_reference] [--max-steps 0]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.baselines.ilp_optimal import ILPOptimalPolicy
from qkd_rl.core.config import deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config

STEP_FIELDS = [
    "step", "t", "reward", "served_keys", "generated_keys", "failed_keys",
    "waiting_keys", "pending_count", "milp_served", "milp_status",
    "solve_time_s", "n_activated", "activated_edges",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the MILP-optimal reference trajectory.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="outputs/milp_reference")
    parser.add_argument("--max-steps", type=int, default=0, help="0 = full window")
    parser.add_argument("--progress", type=int, default=1000)
    args = parser.parse_args()

    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config = deep_merge(config, load_config([ROOT / "configs" / "baselines.yaml"]))
    config["rate_provider"]["provider"] = "h5"
    # Deterministic full-window run: start at t=0 and rely on
    # terminate_on_year_end (truncated at scenario.end_t) to stop.
    config["env"]["episode_start_mode"] = "fixed"
    config["env"]["episode_steps"] = int(10 ** 9)

    env = build_env_from_config(config)
    ilp_cfg = config["baselines"]["ilp_optimal"]
    policy = ILPOptimalPolicy(
        slot_seconds=ilp_cfg.get("slot_seconds", 60.0),
        max_requests=ilp_cfg.get("max_requests", 64),
        max_paths_per_request=ilp_cfg.get("max_paths_per_request", 8),
        max_path_hops=ilp_cfg.get("max_path_hops", 6),
        time_limit_s=ilp_cfg.get("time_limit_s", 5.0),
        mip_rel_gap=ilp_cfg.get("mip_rel_gap", 0.0),
        switch_decay=ilp_cfg.get("switch_decay", 0.5),
    )
    obs = env.reset(seed=args.seed)

    out_dir = Path(ROOT) / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_path = out_dir / "steps.csv"

    max_steps = args.max_steps or (env.scenario.end_t - env.scenario.start_t)
    window_len = env.scenario.end_t - env.scenario.start_t
    print(
        "scenario: {} nodes, {} edges, window {}->{} ({} slots), max_steps={}".format(
            len(env.scenario.nodes), len(env.scenario.edges),
            env.scenario.start_t, env.scenario.end_t, window_len, max_steps,
        ),
        flush=True,
    )

    start_wall = time.time()
    n_steps = 0
    total = {"reward": 0.0, "served_keys": 0.0, "generated_keys": 0.0, "failed_keys": 0.0, "waiting_keys": 0.0}
    with steps_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STEP_FIELDS)
        writer.writeheader()
        done = False
        while not done:
            t_decision = env.t
            n_pending = len(obs.state.pending_requests)
            actions, scores = policy.act(obs)
            outcome = policy.last_outcome
            obs, reward, terminated, truncated, info = env.step(actions, scores)
            n_steps += 1
            total["reward"] += reward
            for key in ("served_keys", "generated_keys", "failed_keys", "waiting_keys"):
                total[key] += info.get(key, 0.0)
            writer.writerow({
                "step": n_steps - 1,
                "t": t_decision,
                "reward": round(float(reward), 6),
                "served_keys": round(float(info.get("served_keys", 0.0)), 3),
                "generated_keys": round(float(info.get("generated_keys", 0.0)), 3),
                "failed_keys": round(float(info.get("failed_keys", 0.0)), 3),
                "waiting_keys": round(float(info.get("waiting_keys", 0.0)), 3),
                "pending_count": n_pending,
                "milp_served": round(float(outcome.served_amount), 3),
                "milp_status": outcome.status,
                "solve_time_s": round(outcome.solve_time_s, 5),
                "n_activated": len(outcome.activated_edges),
                "activated_edges": ";".join(outcome.activated_edges),
            })
            if n_steps % args.progress == 0:
                elapsed = time.time() - start_wall
                rate = n_steps / elapsed if elapsed > 0 else 0.0
                eta_min = (max_steps - n_steps) / rate / 60.0 if rate > 0 else 0.0
                print(
                    "[{}/{}] t={} served={:.0f} rate={:.1f} steps/s eta={:.1f} min".format(
                        n_steps, max_steps, t_decision, total["served_keys"], rate, eta_min,
                    ),
                    flush=True,
                )
            done = (terminated or truncated) or n_steps >= max_steps
    wall = time.time() - start_wall

    summary = env.metrics.episode_summary()
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        payload = dict(summary)
        payload.update({k: round(v, 3) for k, v in total.items()})
        payload["steps"] = n_steps
        payload["wall_s"] = round(wall, 2)
        json.dump(payload, f, indent=2, ensure_ascii=False)

    metadata = {
        "seed": args.seed,
        "scenario": {
            "mode": config["scenario"]["mode"],
            "active_nodes": config["scenario"].get("active_nodes", {}),
            "time_limit": config["scenario"].get("time_limit", {}),
            "window": [env.scenario.start_t, env.scenario.end_t],
            "window_len": window_len,
            "slot_seconds": env.scenario.slot_seconds,
            "n_nodes": len(env.scenario.nodes),
            "n_edges": len(env.scenario.edges),
            "node_ids": env.scenario.node_ids,
        },
        "milp": {k: ilp_cfg.get(k) for k in ("slot_seconds", "max_requests", "max_paths_per_request", "max_path_hops", "time_limit_s", "mip_rel_gap", "switch_decay")},
        "steps_computed": n_steps,
        "wall_seconds": round(wall, 2),
        "steps_per_second": round(n_steps / wall, 3) if wall > 0 else 0.0,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("done: {} steps in {:.1f} s -> {}".format(n_steps, wall, out_dir), flush=True)


if __name__ == "__main__":
    main()
