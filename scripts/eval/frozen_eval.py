"""Frozen evaluation protocol (2026-09-24, do not modify — new protocol = new file).

One entry point that evaluates every policy under the SAME held-out protocol
and writes one JSON per (policy, window). After this run, the numbers recorded
in docs are cited from ``outputs/frozen_eval_20260924/`` — no ad-hoc probes.

Protocol (frozen):
  window A   330-365 days, request seeds 100-114, 15 episodes x 240 steps
  window B   300-329 days, request seeds 200-214, 15 episodes x 240 steps
  start_mode random_day; deterministic policy rollouts
  RL token path identical to trainer.evaluate_validation (edge_scores +
  expected_matched_edges passed to env.step)

Policies:
  expert        PathScoreGreedy(weights=(1,10,1,0.5,0.2), phased=True)
  rl:<name>     frozen checkpoint, config read from its experiment_config.yaml

Usage (on the node):
  python scripts/eval/frozen_eval.py expert --window A --out outputs/frozen_eval
  python scripts/eval/frozen_eval.py rl:outputs/experiments/v3/frozen_s42 \
      --window A --out outputs/frozen_eval
  python scripts/eval/frozen_eval.py expert rl:outputs/experiments/v3/frozen_s42 ... \
      --window A B --out outputs/frozen_eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qkd_rl.core.config import load_config
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.evaluation.test_protocol import build_validation_env_config
from qkd_rl.rl.algos.checkpoint import load_checkpoint
from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer
from qkd_rl.rl.algos.policy import MAPPOPolicy

WINDOWS = {
    "A": {"start_day": 330, "end_day": 365, "request_seeds": list(range(100, 115))},
    "B": {"start_day": 300, "end_day": 329, "request_seeds": list(range(200, 215))},
}
EPISODE_STEPS = 240
START_MODE = "random_day"
EXPERT_WEIGHTS = (1.0, 10.0, 1.0, 0.5, 0.2)


def _window_cfg(window: str) -> dict:
    spec = WINDOWS[window]
    return {
        "window": {"start_day": spec["start_day"], "end_day": spec["end_day"]},
        "request_seeds": list(spec["request_seeds"]),
        "episodes": len(spec["request_seeds"]),
        "episode_steps": EPISODE_STEPS,
        "start_mode": START_MODE,
    }


def _run_name(spec: str) -> str:
    return spec.split(":", 1)[1] if spec.startswith("rl:") else spec


def eval_rl(spec: str, window: str, device: str, checkpoint_override: str | None = None) -> dict:
    run_dir = Path(_run_name(spec))
    config = load_config([run_dir / "experiment_config.yaml"])
    config["validation"] = _window_cfg(window)
    checkpoint = Path(checkpoint_override) if checkpoint_override else run_dir / "checkpoint_final.pt"
    device_obj = torch.device(device)
    torch.set_num_threads(int(config.get("experiment", {}).get("cpu_threads", 4)))
    env = build_env_from_config(config)
    from qkd_rl.model_zoo import build_model
    model = build_model(env.action_resolver.action_space, config)
    model.load_state_dict(load_checkpoint(str(checkpoint), device).model_state)
    model.to(device_obj)
    policy = MAPPOPolicy(model, device_obj)
    run_stub = run_dir / f"frozen_eval_{window}"
    run_stub.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, run_stub, device=device_obj)
    result = trainer.evaluate_validation(num_episodes=int(config["validation"]["episodes"]))
    result["window"] = window
    return result


def eval_expert(window: str, device: str) -> dict:
    from qkd_rl.baselines.path_greedy import PathScoreGreedy
    from qkd_rl.baselines.serve_probe import ServeProbe

    spec = WINDOWS[window]
    config = build_validation_env_config(
        {
            "window_start_day": spec["start_day"],
            "window_end_day": spec["end_day"],
            "episode_days": 1,
            "episode_steps": EPISODE_STEPS,
            "episodes": len(spec["request_seeds"]),
            "seed_start": 0,
            "seeds": list(spec["request_seeds"]),
            "start_seed": 0,
            "start_mode": START_MODE,
        },
        include_baselines=False,
        episode_steps=EPISODE_STEPS,
        start_mode=START_MODE,
    )
    per_seed = []
    key_ratio = []
    for seed in spec["request_seeds"]:
        env = build_env_from_config(config)
        obs = env.reset(seed=seed, start_seed=seed)
        expert = PathScoreGreedy(
            weights=EXPERT_WEIGHTS, phased=True, principles=False,
            router=ServeProbe(env),
        )
        served = generated = 0.0
        done = False
        steps = 0
        while not done:
            actions, scores = expert.act(obs)
            obs, _r, terminated, truncated, info = env.step(actions, scores)
            served += float(info.get("served_keys", 0.0))
            generated += float(info.get("generated_keys", 0.0))
            steps += 1
            done = terminated or truncated
        arrived = float(env.metrics.arrived_keys)
        sr = served / arrived if arrived else 0.0
        per_seed.append(sr)
        key_ratio.append(generated / served if served else 0.0)
        print(f"  [{window}] expert seed {seed}: sr={sr:.4f}", flush=True)
    return {
        "mean_success_rate": sum(per_seed) / len(per_seed),
        "per_seed_success": per_seed,
        "mean_key_efficiency": sum(key_ratio) / len(key_ratio),
        "window": window,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("policies", nargs="+", help="expert | rl:<run_dir>")
    ap.add_argument("--window", nargs="+", default=["A"], choices=sorted(WINDOWS))
    ap.add_argument("--out", default="outputs/frozen_eval")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint", default=None,
                    help="override the checkpoint (default <run>/checkpoint_final.pt); "
                         "e.g. the BC-only weights for a pre-PPO reading")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for window in args.window:
        for spec in args.policies:
            name = spec.replace(":", "_").replace("/", "_")
            out_path = out_dir / f"{name}_{window}.json"
            if out_path.exists() and not args.checkpoint:
                print(f"skip existing {out_path}", flush=True)
                continue
            t0 = time.perf_counter()
            if spec == "expert":
                result = eval_expert(window, args.device)
            elif spec.startswith("rl:"):
                result = eval_rl(spec, window, args.device, args.checkpoint)
            else:
                raise SystemExit(f"unknown policy spec: {spec} (use 'expert' or 'rl:<run_dir>')")
            result.update({
                "policy": spec,
                "protocol": {"window": window, **WINDOWS[window], "episode_steps": EPISODE_STEPS,
                             "start_mode": START_MODE},
                "evaluated_at": stamp,
            })
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            dt = time.perf_counter() - t0
            mean = result.get("mean_success_rate")
            print(f"[{window}] {spec}: sr={mean:.4f} ({dt:.0f}s) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
