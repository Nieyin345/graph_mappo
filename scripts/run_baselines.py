"""Run heuristic baselines on the QKD env and export paper-ready figures.

Usage:
    conda run -n pytorch python scripts/run_baselines.py \
        --episodes 5 --seeds 1000,1001,1002 --out outputs/eval/small

Writes per-episode CSV, per-step CSV, aggregate JSON and SVG/PDF/PNG figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.baselines.greedy_rate import GreedyRatePolicy
from qkd_rl.baselines.random_policy import RandomPolicy
from qkd_rl.env.factory import build_default_env
from qkd_rl.evaluation import Evaluator, plot_episode_timeline, plot_policy_comparison
from qkd_rl.evaluation.evaluator import write_episodes_csv, write_steps_csv, write_summary_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run heuristic baselines and export figures.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="1000,1001,1002,1003,1004")
    parser.add_argument("--out", type=str, default="outputs/eval/small")
    parser.add_argument("--name", type=str, default="")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(ROOT) / args.out

    def env_builder(seed: int):
        env = build_default_env(ROOT)
        env.reset(seed=seed)
        return env

    evaluator = Evaluator(env_builder=env_builder)
    policies = {
        "random": RandomPolicy(seed=0),
        "greedy_rate": GreedyRatePolicy(),
    }
    episodes, step_rows = evaluator.compare(
        policies, num_episodes=args.episodes, seeds=seeds, collect_steps=True
    )

    write_episodes_csv(episodes, out_dir / "episodes.csv")
    write_summary_json(episodes, out_dir / "summary.json")
    for policy, rows in step_rows.items():
        write_steps_csv(rows, out_dir / f"steps_{policy}.csv")

    run_name = args.name or "baselines"
    comparison = plot_policy_comparison(
        episodes, out_dir / f"figures/{run_name}_policy_comparison", title=f"{run_name} (n={args.episodes})"
    )
    timeline = plot_episode_timeline(
        step_rows, out_dir / f"figures/{run_name}_episode_timeline", episode_index=0, title=run_name
    )

    print(f"episodes written : {out_dir / 'episodes.csv'}")
    print(f"summary written  : {out_dir / 'summary.json'}")
    for f in comparison + timeline:
        print(f"figure           : {f}")


if __name__ == "__main__":
    main()
