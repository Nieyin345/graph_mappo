"""Diagnose multi-day key flow: utilization, saturation, bottleneck links.

Usage:
    python scripts/diagnose_eval_flow.py outputs/train_curriculum_v3/checkpoint_update_000125.pt
        --seed 4321 --days 3
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
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
    parser.add_argument("checkpoint")
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config["rate_provider"]["provider"] = "h5"
    config["env"]["episode_start_mode"] = "fixed"
    config["env"]["episode_steps"] = int(10**9)
    config["scenario"]["time_limit"]["days"] = args.days
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, device)
    data = load_checkpoint(Path(args.checkpoint), device)
    model.load_state_dict(data.model_state)
    model.eval()

    obs = env.reset(seed=args.seed)
    day_steps = int(config["env"].get("day_steps", 1440))
    day = 0
    served_day = generated_day = failed_day = 0.0
    act_counter: Counter = Counter()
    switches = 0
    prev_activated: set[str] = set()
    util_samples: list[float] = []
    step = 0
    done = False
    while not done:
        act = policy.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(
            act.actions,
            act.action_scores,
            edge_scores=act.edge_scores,
        )
        activated = set(env.last_activated_edges)
        switches += len(activated - prev_activated)
        prev_activated = activated
        act_counter.update(activated)
        served_day += info.get("served_keys", 0.0)
        generated_day += info.get("generated_keys", 0.0)
        failed_day += info.get("failed_keys", 0.0)
        total_cap = sum(env.qkp.capacities.values()) or 1.0
        total_level = sum(env.qkp.levels.values())
        util_samples.append(total_level / total_cap)
        step += 1
        done = terminated or truncated
        if step % day_steps == 0:
            summary = env.metrics.episode_summary()
            print(
                f"day {day}: served={served_day:.3g} generated={generated_day:.3g} "
                f"failed={failed_day:.3g} util_mean={sum(util_samples)/len(util_samples):.4f} "
                f"util_max={max(util_samples):.4f} steps={step}"
            )
            day += 1
            served_day = generated_day = failed_day = 0.0
            util_samples = []

    summary = env.metrics.episode_summary()
    print(f"\nTOTAL: served={summary['served_keys']:.3g} arrived={summary['arrived_keys']:.3g} "
          f"success={summary['success_rate']:.4f} failed={summary['failed_keys']:.3g}")
    print(f"switches_total={switches} steps={step} avg_activated={sum(act_counter.values())/max(1,step):.2f} "
          f"unique_activated={len(act_counter)}")
    print("\ntop activated edges:")
    for edge_id, count in act_counter.most_common(8):
        cap = env.qkp.capacities.get(edge_id, 0.0)
        level = env.qkp.levels.get(edge_id, 0.0)
        print(f"  {edge_id}: activated={count} final_level={level:.3g}/{cap:.3g} util={level/cap if cap else 0:.3f}")


if __name__ == "__main__":
    main()
