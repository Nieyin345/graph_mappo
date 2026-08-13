"""Universal RL evaluation driven by a YAML test profile.

Default profile: ``configs/global.yaml``. CLI arguments override the YAML
values when provided, so the script can be used unchanged for routine tests
and still be parameterized for one-off experiments.

Usage:
    python scripts/eval_long_horizon.py
    python scripts/eval_long_horizon.py --config configs/global.yaml
    python scripts/eval_long_horizon.py outputs/ckpt.pt --episodes 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.algos.checkpoint import load_checkpoint
from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.evaluation.test_protocol import build_validation_env_config
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="*")
    parser.add_argument("--config", type=str, default="configs/global.yaml")
    parser.add_argument("--window-start-day", type=int, default=None)
    parser.add_argument("--window-end-day", type=int, default=None)
    parser.add_argument("--episode-days", type=int, default=None)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--start-mode", choices=("random_day", "fixed"), default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_settings(args: argparse.Namespace) -> dict:
    settings = {
        "checkpoints": [],
        "window_start_day": 0,
        "window_end_day": 30,
        "episode_days": 1,
        "episode_steps": 0,
        "episodes": 3,
        "seed_start": 7,
        "seeds": [],
        "start_seed": 0,
        "start_mode": "random_day",
        "out": "outputs/eval/test_default.json",
        "device": "cuda",
    }
    config_path = Path(ROOT) / args.config
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        ev = raw.get("global", {}).get("validation", {}) or {}
        window = ev.get("window", {}) or {}
        episode = ev.get("episode", {}) or {}
        settings.update(
            {
                "checkpoints": list(ev.get("checkpoints", []) or []),
                "window_start_day": int(window.get("start_day", settings["window_start_day"])),
                "window_end_day": int(window.get("end_day", settings["window_end_day"])),
                "episode_days": int(
                    episode.get("days", ev.get("episode_days", settings["episode_days"]))
                ),
                "episode_steps": int(
                    episode.get("steps", ev.get("episode_steps", settings["episode_steps"])) or 0
                ),
                "episodes": int(ev.get("episodes", settings["episodes"])),
                "seed_start": int(ev.get("seed_start", settings["seed_start"])),
                "seeds": [
                    int(s)
                    for s in (ev.get("seeds", []) or ev.get("request_seeds", []) or [])
                ],
                "start_seed": int(ev.get("start_seed", settings["start_seed"])),
                "start_mode": str(ev.get("start_mode", settings["start_mode"])),
                "out": str(ev.get("out", settings["out"])),
                "device": str(ev.get("device", settings["device"])),
            }
        )

    if args.checkpoints:
        settings["checkpoints"] = args.checkpoints
    if args.window_start_day is not None:
        settings["window_start_day"] = args.window_start_day
    if args.window_end_day is not None:
        settings["window_end_day"] = args.window_end_day
    if args.episode_days is not None:
        settings["episode_days"] = args.episode_days
    if args.episode_steps is not None:
        settings["episode_steps"] = args.episode_steps
    if args.episodes is not None:
        settings["episodes"] = args.episodes
    if args.seed_start is not None:
        settings["seed_start"] = args.seed_start
    if args.seeds:
        settings["seeds"] = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.start_mode is not None:
        settings["start_mode"] = args.start_mode
    if args.out is not None:
        settings["out"] = args.out
    if args.device is not None:
        settings["device"] = args.device
    return settings


def build_env_config(settings: dict) -> tuple[dict, list[int]]:
    profile = {
        "window_start_day": settings["window_start_day"],
        "window_end_day": settings["window_end_day"],
        "episode_days": settings["episode_days"],
        "episode_steps": settings["episode_steps"],
        "episodes": settings["episodes"],
        "seed_start": settings["seed_start"],
        "seeds": settings["seeds"],
        "start_seed": settings["start_seed"],
        "start_mode": settings["start_mode"],
    }
    if settings["window_end_day"] <= settings["window_start_day"]:
        raise ValueError(
            f"window end day must be > start day: "
            f"{settings['window_end_day']} <= {settings['window_start_day']}"
        )
    config = build_validation_env_config(profile)
    seeds = settings["seeds"] or list(
        range(int(settings["seed_start"]), int(settings["seed_start"]) + int(settings["episodes"]))
    )
    return config, seeds


def main() -> None:
    args = parse_args()
    settings = load_settings(args)
    if not settings["checkpoints"]:
        raise SystemExit("No checkpoints configured. Add them to configs/global.yaml or pass paths on CLI.")
    config, seeds = build_env_config(settings)
    device = settings["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    env = build_env_from_config(config)
    output: dict = {
        "start_mode": settings["start_mode"],
        "window_start_day": settings["window_start_day"],
        "window_end_day": settings["window_end_day"],
        "episode_steps": int(config["env"]["episode_steps"]),
        "seeds": seeds,
        "checkpoints": {},
    }

    incremental_path = None
    if settings["out"]:
        incremental_path = Path(ROOT) / (settings["out"] + ".jsonl")
        incremental_path.parent.mkdir(parents=True, exist_ok=True)
        incremental_path.write_text("", encoding="utf-8")

    for ckpt_path in settings["checkpoints"]:
        ckpt = Path(ckpt_path)
        model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
        policy = MAPPOPolicy(model, device)
        data = load_checkpoint(ckpt, device)
        model.load_state_dict(data.model_state)
        model.eval()

        rows = []
        for seed in seeds:
            obs = env.reset(
                seed=seed,
                start_seed=int(settings.get("start_seed", 0)) + seed,
            )
            start_t = int(env.t)
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
            rows.append(
                {
                    "seed": seed,
                    "start_t": start_t,
                    "end_t": int(env.t),
                    "steps": summary["steps"],
                    "arrived_keys": summary["arrived_keys"],
                    "served_keys": summary["served_keys"],
                    "failed_keys": summary["failed_keys"],
                    "success_rate": summary["success_rate"],
                    "request_completion_rate": summary["request_completion_rate"],
                    "arrived_requests": summary["arrived_requests"],
                    "completed_requests": summary["completed_requests"],
                    "total_reward": round(float(total_reward), 3),
                }
            )
            if incremental_path is not None:
                with incremental_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")

        result = {
            "checkpoint": str(ckpt),
            "update": data.update,
            "mean_success_rate": round(statistics.mean(r["success_rate"] for r in rows), 6),
            "mean_request_completion_rate": round(
                statistics.mean(r["request_completion_rate"] for r in rows), 6
            ),
            "mean_served_keys": round(statistics.mean(r["served_keys"] for r in rows), 3),
            "mean_arrived_keys": round(statistics.mean(r["arrived_keys"] for r in rows), 3),
            "min_success_rate": round(min(r["success_rate"] for r in rows), 6),
            "max_success_rate": round(max(r["success_rate"] for r in rows), 6),
            "episodes": rows,
        }
        output["checkpoints"][str(ckpt)] = result
        print(json.dumps(result, ensure_ascii=False))

    if settings["out"]:
        out_path = Path(ROOT) / settings["out"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"written: {out_path}")


if __name__ == "__main__":
    main()
