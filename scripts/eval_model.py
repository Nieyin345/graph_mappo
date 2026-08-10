"""Evaluate RL checkpoints on fixed and random-day windows.

Usage:
    python scripts/eval_model.py outputs/run/checkpoint_update_000100.pt
    python scripts/eval_model.py outputs/run/checkpoint_final.pt \
        --random-count 10 --out outputs/eval/model_report.json

The script loads the current demand_edge config, so it must be run against
checkpoints trained with that mode.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.algos.checkpoint import load_checkpoint
from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


def _config(days: int, start_mode: str, episode_steps: int):
    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config = deep_merge(config, load_config([ROOT / "configs" / "train_demand_edge.yaml"]))
    config["rate_provider"]["provider"] = "h5"
    config["env"]["episode_start_mode"] = start_mode
    config["env"]["episode_steps"] = episode_steps
    config["scenario"]["time_limit"]["days"] = days
    ConfigValidator().validate(config)
    return config


def _beam_step(env, model, obs, device, endpoints):
    idle = env.action_resolver.action_space.IDLE
    output = model(obs, device, build_logits_dict=False)
    scores = {
        eid: float(s.detach().cpu())
        for eid, s in (output.edge_scores or {}).items()
    }
    beams = [([], set(), 0.0)]
    while True:
        new_beams = []
        for matched, used, total in beams:
            avail = [
                (eid, s)
                for eid, s in scores.items()
                if endpoints[eid][0] not in used and endpoints[eid][1] not in used
            ]
            if not avail:
                new_beams.append((matched, used, total))
                continue
            for eid, s in avail:
                u, v = endpoints[eid]
                new_beams.append((matched + [eid], used | {u, v}, total + s))
        if not new_beams or len(new_beams) == len(beams):
            beams = new_beams or beams
            break
        beams = sorted(new_beams, key=lambda item: -item[2])[:4]
    best = max(beams, key=lambda item: item[2])
    actions = {n: idle for n in env.scenario.node_ids}
    action_scores = {n: {idle: 0.0} for n in env.scenario.node_ids}
    for eid in best[0]:
        u, v = endpoints[eid]
        actions[u] = v
        actions[v] = u
        action_scores[u] = {v: scores[eid]}
        action_scores[v] = {u: scores[eid]}
    return actions, action_scores, scores


def _run_episode(policy, model, env, seed: int, selection: str, endpoints) -> tuple[dict, float, int]:
    obs = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    while not done:
        if selection == "beam4":
            actions, action_scores, edge_scores = _beam_step(
                env, model, obs, policy.device, endpoints
            )
        else:
            step = policy.act(obs, deterministic=True)
            actions = step.actions
            action_scores = step.action_scores
            edge_scores = step.edge_scores
        obs, reward, terminated, truncated, _info = env.step(
            actions,
            action_scores,
            edge_scores=edge_scores,
            expected_matched_edges=(
                list(step.matched_edges or []) if selection != "beam4" else None
            ),
        )
        total_reward += reward
        done = terminated or truncated
    start_day = (env.t - env.scenario.start_t) // 1440 if env.scenario.start_t is not None else 0
    return env.metrics.episode_summary(), total_reward, start_day


def _evaluate_fixed(policy, model, device, seed: int, days: int, selection: str, endpoints) -> dict:
    env = build_env_from_config(_config(days, "fixed", int(10**9)))
    summary, reward, _ = _run_episode(policy, model, env, seed, selection, endpoints)
    return {
        "seed": seed,
        "days": days,
        "success_rate": round(float(summary["success_rate"]), 6),
        "request_completion_rate": round(float(summary["request_completion_rate"]), 6),
        "total_reward": round(float(reward), 3),
        "served_keys": round(float(summary["served_keys"]), 3),
        "failed_keys": round(float(summary["failed_keys"]), 3),
        "conflict_count": int(summary["conflict_count"]),
    }


def _evaluate_random_day(policy, model, device, seeds: list[int], selection: str, endpoints) -> dict:
    env = build_env_from_config(_config(365, "random_day", 1440))
    per_seed = []
    for seed in seeds:
        summary, reward, start_day = _run_episode(policy, model, env, seed, selection, endpoints)
        per_seed.append(
            {
                "seed": seed,
                "start_day": start_day,
                "success_rate": round(float(summary["success_rate"]), 6),
                "request_completion_rate": round(float(summary["request_completion_rate"]), 6),
                "total_reward": round(float(reward), 3),
            }
        )
    rates = [row["success_rate"] for row in per_seed]
    return {
        "n": len(rates),
        "unique_start_days": len({row["start_day"] for row in per_seed}),
        "mean_success_rate": round(statistics.mean(rates), 6) if rates else 0.0,
        "min_success_rate": round(min(rates), 6) if rates else 0.0,
        "max_success_rate": round(max(rates), 6) if rates else 0.0,
        "std_success_rate": round(statistics.pstdev(rates), 6) if rates else 0.0,
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="JSON output path under outputs/")
    parser.add_argument("--fixed-seed", type=int, default=1234)
    parser.add_argument("--fixed-days", type=int, default=1)
    parser.add_argument("--multi-day-seed", type=int, default=4321)
    parser.add_argument("--multi-days", type=int, default=3)
    parser.add_argument("--random-count", type=int, default=10)
    parser.add_argument("--random-start-seed", type=int, default=1001)
    parser.add_argument("--selection", default="beam4", choices=["greedy", "beam4"])
    args = parser.parse_args()

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    random_seeds = list(range(args.random_start_seed, args.random_start_seed + args.random_count))
    results = {}

    for checkpoint in args.checkpoints:
        ckpt_path = Path(checkpoint)
        env = build_env_from_config(_config(1, "fixed", int(10**9)))
        endpoints = {e.edge_id: (e.src, e.dst) for e in env.scenario.edges}
        model = GraphMAPPOActorCritic(env.action_resolver.action_space, _config(1, "fixed", int(10**9)))
        policy = MAPPOPolicy(model, device)
        data = load_checkpoint(ckpt_path, device)
        model.load_state_dict(data.model_state)
        model.eval()

        entry = {
            "checkpoint": str(ckpt_path),
            "update": data.update,
            "selection": args.selection,
            "fixed": _evaluate_fixed(policy, model, device, args.fixed_seed, args.fixed_days, args.selection, endpoints),
            "multi_day": _evaluate_fixed(policy, model, device, args.multi_day_seed, args.multi_days, args.selection, endpoints),
            "random_day": _evaluate_random_day(policy, model, device, random_seeds, args.selection, endpoints),
        }
        results[str(ckpt_path)] = entry
        print(json.dumps(entry, ensure_ascii=False, indent=2))

    if args.out:
        out_path = Path(ROOT) / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"written: {out_path}")


if __name__ == "__main__":
    main()
