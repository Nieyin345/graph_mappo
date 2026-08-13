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
from qkd_rl.evaluation.test_protocol import (
    build_validation_env_config,
    load_validation_profile,
    resolve_seeds,
)

STEP_FIELDS = [
    "step", "t", "reward", "served_keys", "generated_keys", "failed_keys",
    "waiting_keys", "pending_count", "milp_served", "milp_status",
    "solve_time_s", "n_activated", "activated_edges",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the MILP-optimal reference trajectory.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="outputs/milp_reference")
    parser.add_argument("--max-steps", type=int, default=0, help="0 = full window")
    parser.add_argument("--progress", type=int, default=1000)
    parser.add_argument("--start-mode", choices=("fixed", "random_day"), default=None)
    parser.add_argument("--window-start-day", type=int, default=None)
    parser.add_argument("--window-days", type=int, default=None)
    parser.add_argument("--episode-days", type=int, default=None, help="Episode length in days; 0 = full fixed window.")
    parser.add_argument("--wait-ratio", type=float, default=None)
    parser.add_argument("--service-slack", type=float, default=None)
    args = parser.parse_args()

    profile = load_validation_profile(ROOT / "configs" / "global.yaml")
    episode_days = args.episode_days
    if episode_days is None:
        episode_days = int(profile["episode_days"])
    start_mode = args.start_mode or profile["start_mode"]
    window_start_day = args.window_start_day
    if window_start_day is None:
        window_start_day = int(profile["window_start_day"])
    window_days = args.window_days
    if window_days is None:
        window_days = int(profile["window_end_day"]) - window_start_day
    seed = args.seed if args.seed is not None else resolve_seeds(profile)[0]

    if episode_days > 0:
        milp_profile = {
            "window_start_day": window_start_day,
            "window_end_day": window_start_day + window_days,
            "episode_days": episode_days,
            "episode_steps": 0,
            "start_mode": start_mode,
        }
        config = build_validation_env_config(
            milp_profile,
            include_baselines=True,
            episode_days=episode_days,
            start_mode=start_mode,
        )
    else:
        config = load_default_config(ROOT)
        config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
        config = deep_merge(config, load_config([ROOT / "configs" / "baselines.yaml"]))
        config["rate_provider"]["provider"] = "h5"
        config["env"]["episode_start_mode"] = "fixed"
        config["env"]["episode_steps"] = int(10 ** 9)
        config["scenario"]["time_limit"]["days"] = window_start_day + window_days

    env = build_env_from_config(config)
    ilp_cfg = config["baselines"]["ilp_optimal"]
    if args.wait_ratio is not None:
        ilp_cfg["wait_urgency_tau_ratio"] = args.wait_ratio
    if args.service_slack is not None:
        ilp_cfg["service_slack_ratio"] = args.service_slack
    policy = ILPOptimalPolicy(
        slot_seconds=ilp_cfg.get("slot_seconds", 60.0),
        max_requests=ilp_cfg.get("max_requests", 64),
        max_paths_per_request=ilp_cfg.get("max_paths_per_request", 8),
        max_path_hops=ilp_cfg.get("max_path_hops", 6),
        time_limit_s=ilp_cfg.get("time_limit_s", 5.0),
        mip_rel_gap=ilp_cfg.get("mip_rel_gap", 0.0),
        switch_decay=ilp_cfg.get("switch_decay", 0.5),
        importance_weight=ilp_cfg.get("importance_weight", 2.0),
        max_path_links=ilp_cfg.get("max_path_links", 3),
        hop_decay_factor=ilp_cfg.get("hop_decay_factor", 0.25),
        wait_urgency_tau_ratio=ilp_cfg.get("wait_urgency_tau_ratio", 0.8),
        service_slack_ratio=ilp_cfg.get("service_slack_ratio", 0.05),
        ignore_consumption=ilp_cfg.get("ignore_consumption", False),
    )
    obs = env.reset(
        seed=seed,
        start_seed=int(profile.get("start_seed", 0)) + seed,
    )

    out_dir = Path(ROOT) / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_path = out_dir / "steps.csv"

    max_steps = args.max_steps or (
        episode_days * 1440 if episode_days > 0 else env.scenario.end_t - env.scenario.start_t
    )
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
        "seed": seed,
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
        "milp": {k: ilp_cfg.get(k) for k in ("slot_seconds", "max_requests", "max_paths_per_request", "max_path_hops", "time_limit_s", "mip_rel_gap", "switch_decay", "importance_weight", "max_path_links", "hop_decay_factor", "wait_urgency_tau_ratio", "service_slack_ratio", "ignore_consumption")},
        "steps_computed": n_steps,
        "wall_seconds": round(wall, 2),
        "steps_per_second": round(n_steps / wall, 3) if wall > 0 else 0.0,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("done: {} steps in {:.1f} s -> {}".format(n_steps, wall, out_dir), flush=True)


if __name__ == "__main__":
    main()
