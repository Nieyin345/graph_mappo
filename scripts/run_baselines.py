"""Run heuristic baselines on the QKD env and export paper-ready figures.

Usage:
    conda run -n pytorch python scripts/run_baselines.py \
        --episodes 5 --seeds 1000,1001,1002 --out outputs/eval/small

Writes per-episode CSV, per-step CSV, aggregate JSON and SVG/PDF/PNG figures.
"""
from __future__ import annotations

import os

# Conda/matplotlib on Windows loads libiomp5md.dll twice; this documented
# workaround prevents the OMP Error #15 crash at import/exit.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.baselines.greedy_demand import GreedyDemandPolicy
from qkd_rl.baselines.greedy_matching import GreedyMatchingPolicy
from qkd_rl.baselines.greedy_qkp import GreedyQKPPolicy
from qkd_rl.baselines.greedy_rate import GreedyRatePolicy
from qkd_rl.baselines.greedy_relay import GreedyRelayPolicy
from qkd_rl.baselines.ilp_optimal import ILPOptimalPolicy
from qkd_rl.baselines.random_policy import RandomPolicy
from qkd_rl.algos.checkpoint import load_checkpoint
from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic
from qkd_rl.core.config import deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.evaluation import Evaluator, plot_episode_timeline, plot_policy_comparison
from qkd_rl.evaluation.evaluator import write_episodes_csv, write_steps_csv, write_summary_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run heuristic baselines and export figures.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="1000,1001,1002,1003,1004")
    parser.add_argument("--out", type=str, default="outputs/eval/small")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--rl-checkpoint", type=str, default=None, help="Optional trained RL checkpoint .pt to include in the comparison.")
    parser.add_argument("--rl-name", type=str, default="rl_model")
    parser.add_argument("--device", type=str, default=None, help="Device for the RL model (auto: cuda if available).")
    parser.add_argument("--episode-start-mode", type=str, default=None, help="Override env.episode_start_mode (e.g. fixed for a deterministic t=0 comparison).")
    parser.add_argument("--time-limit-days", type=int, default=None, help="Override scenario.time_limit.days (1 = first-day protocol).")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(ROOT) / args.out

    def base_config():
        config = load_default_config(ROOT)
        config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
        config = deep_merge(config, load_config([ROOT / "configs" / "baselines.yaml"]))
        config["rate_provider"]["provider"] = "h5"
        if args.episode_start_mode:
            config["env"]["episode_start_mode"] = args.episode_start_mode
        if args.time_limit_days is not None:
            config["scenario"]["time_limit"]["days"] = args.time_limit_days
        return config

    def env_builder(seed: int):
        env = build_env_from_config(base_config())
        env.reset(seed=seed)
        return env

    # Baseline configuration lives in configs/baselines.yaml; every policy is
    # independent of the training stack and each entry can be disabled.
    base_cfg = load_config([ROOT / "configs" / "baselines.yaml"]).get("baselines", {})
    policies = {}
    if base_cfg.get("random", {}).get("enabled", True):
        policies["random"] = RandomPolicy(seed=0)
    gr_cfg = base_cfg.get("greedy_rate", {})
    if gr_cfg.get("enabled", True):
        policies["greedy_rate"] = GreedyRatePolicy(use_future_mean_rate=gr_cfg.get("use_future_mean_rate", False))
    gq_cfg = base_cfg.get("greedy_qkp", {})
    if gq_cfg.get("enabled", True):
        policies["greedy_qkp"] = GreedyQKPPolicy(
            low_inventory_weight=gq_cfg.get("low_inventory_weight", 0.0),
            rate_weight=gq_cfg.get("rate_weight", 0.2),
        )
    gd_cfg = base_cfg.get("greedy_demand", {})
    if gd_cfg.get("enabled", True):
        policies["greedy_demand"] = GreedyDemandPolicy(rate_weight=gd_cfg.get("rate_weight", 0.2))
    relay_cfg = base_cfg.get("greedy_relay", {})
    if relay_cfg.get("enabled", True):
        policies["greedy_relay"] = GreedyRelayPolicy(
            rate_weight=relay_cfg.get("rate_weight", 1.0),
            demand_weight=relay_cfg.get("demand_weight", 2.0),
            completion_multiplier=relay_cfg.get("completion_multiplier", 3.0),
            keep_weight=relay_cfg.get("keep_weight", 0.5),
            deadline_window=relay_cfg.get("deadline_window", 60.0),
        )
    gm_cfg = base_cfg.get("greedy_matching", {})
    if gm_cfg.get("enabled", True):
        policies["greedy_matching"] = GreedyMatchingPolicy(
            rate_weight=gm_cfg.get("rate_weight", 1.0),
            level_weight=gm_cfg.get("level_weight", 0.5),
            demand_weight=gm_cfg.get("demand_weight", 1.0),
            keep_weight=gm_cfg.get("keep_weight", 0.5),
        )
    if args.rl_checkpoint:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        template_env = build_env_from_config(base_config())
        model = GraphMAPPOActorCritic(template_env.action_resolver.action_space, base_config())
        rl_policy = MAPPOPolicy(model, device)
        data = load_checkpoint(args.rl_checkpoint, device)
        model.load_state_dict(data.model_state)
        model.eval()

        class _RLPolicyAdapter:
            def __init__(self, policy):
                self.policy = policy

            def act(self, obs):
                step = self.policy.act(obs, deterministic=True)
                return step.actions, step.action_scores

        policies[args.rl_name] = _RLPolicyAdapter(rl_policy)
        print(f"rl policy: loaded {args.rl_checkpoint} (update={data.update}) on {device}")

    ilp_cfg = base_cfg.get("ilp_optimal", {})
    if ilp_cfg.get("enabled", True):
        policies["ilp_optimal"] = ILPOptimalPolicy(
            slot_seconds=ilp_cfg.get("slot_seconds", 60.0),
            max_requests=ilp_cfg.get("max_requests", 64),
            max_paths_per_request=ilp_cfg.get("max_paths_per_request", 8),
            max_path_hops=ilp_cfg.get("max_path_hops", 6),
            time_limit_s=ilp_cfg.get("time_limit_s", 5.0),
            mip_rel_gap=ilp_cfg.get("mip_rel_gap", 0.0),
            switch_decay=ilp_cfg.get("switch_decay", 0.5),
        )
    evaluator = Evaluator(env_builder)
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

