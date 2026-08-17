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
# Suppress harmless libpng iCCP warnings from matplotlib figure generation
import warnings
warnings.filterwarnings("ignore", message=".*iCCP.*")

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
from qkd_rl.baselines.greedy_relay_diffusion import GreedyRelayDiffusionPolicyV3
from qkd_rl.baselines.random_policy import RandomPolicy
from qkd_rl.algos.checkpoint import load_checkpoint
from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic
from qkd_rl.core.config import load_config
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.evaluation.test_protocol import (
    build_validation_env_config,
    load_validation_profile,
    resolve_seeds,
)
from qkd_rl.evaluation import Evaluator, plot_episode_timeline, plot_policy_comparison
from qkd_rl.evaluation.evaluator import merge_eval_summary, write_episodes_csv, write_steps_csv, write_summary_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run heuristic baselines and export figures.")
    parser.add_argument("--config", type=str, default="configs/global.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--out", type=str, default="outputs/eval/small")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--rl-checkpoint", type=str, default=None, help="Optional trained RL checkpoint .pt to include in the comparison.")
    parser.add_argument("--rl-name", type=str, default="rl_model")
    parser.add_argument("--device", type=str, default=None, help="Device for the RL model (auto: cuda if available).")
    parser.add_argument("--episode-start-mode", type=str, default=None, help="Override env.episode_start_mode (e.g. fixed for a deterministic t=0 comparison).")
    parser.add_argument("--time-limit-days", type=int, default=None, help="Override scenario.time_limit.days.")
    parser.add_argument("--policies", type=str, default=None, help="Comma-separated list of policies to run (overrides baselines.yaml enabled flag).")
    parser.add_argument("--skip-ilp", action="store_true", help="Skip the slow MILP baseline (run it separately with compute_milp_reference.py).")
    args = parser.parse_args()

    profile = load_validation_profile(ROOT / args.config)
    if args.episodes is None:
        args.episodes = profile["episodes"]
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = resolve_seeds(profile, args.episodes)
    if args.episode_start_mode is None:
        args.episode_start_mode = profile["start_mode"]
    episode_steps = profile["episode_steps"] or profile["episode_days"] * 1440
    out_dir = Path(ROOT) / args.out

    def base_config():
        config = build_validation_env_config(
            profile,
            include_baselines=True,
            episode_steps=episode_steps,
            start_mode=args.episode_start_mode,
        )
        if args.time_limit_days is not None:
            config["scenario"]["time_limit"]["days"] = args.time_limit_days
        return config

    def env_builder(seed: int):
        env = build_env_from_config(base_config())
        env.reset(
            seed=seed,
            start_seed=int(profile.get("start_seed", 0)) + seed,
        )
        return env

    # Baseline configuration lives in configs/baselines.yaml; every policy is
    # independent of the training stack and each entry can be disabled.
    base_cfg = load_config([ROOT / "configs" / "baselines.yaml"]).get("baselines", {})
    # Parse --policies filter: if provided, only run these policies
    policies_filter: set[str] | None = None
    if args.policies:
        policies_filter = {p.strip() for p in args.policies.split(",") if p.strip()}
    def _enabled(name: str) -> bool:
        if policies_filter is not None:
            return name in policies_filter
        return base_cfg.get(name, {}).get("enabled", True)
    policies = {}
    template_env = None
    if _enabled("random"):
        policies["random"] = RandomPolicy(seed=0)
    gr_cfg = base_cfg.get("greedy_rate", {})
    if _enabled("greedy_rate"):
        policies["greedy_rate"] = GreedyRatePolicy(use_future_mean_rate=gr_cfg.get("use_future_mean_rate", False))
    gq_cfg = base_cfg.get("greedy_qkp", {})
    if _enabled("greedy_qkp"):
        policies["greedy_qkp"] = GreedyQKPPolicy(
            low_inventory_weight=gq_cfg.get("low_inventory_weight", 0.0),
            rate_weight=gq_cfg.get("rate_weight", 0.2),
        )
    gd_cfg = base_cfg.get("greedy_demand", {})
    if _enabled("greedy_demand"):
        policies["greedy_demand"] = GreedyDemandPolicy(rate_weight=gd_cfg.get("rate_weight", 0.2))
    relay_cfg = base_cfg.get("greedy_relay", {})
    if _enabled("greedy_relay"):
        policies["greedy_relay"] = GreedyRelayPolicy(
            rate_weight=relay_cfg.get("rate_weight", 1.0),
            demand_weight=relay_cfg.get("demand_weight", 2.0),
            completion_multiplier=relay_cfg.get("completion_multiplier", 3.0),
            keep_weight=relay_cfg.get("keep_weight", 0.5),
            deadline_window=relay_cfg.get("deadline_window", 60.0),
        )
    grd_v3_cfg = base_cfg.get("greedy_relay_diffusion_v3", {})
    if _enabled("greedy_relay_diffusion_v3"):
        policies["greedy_relay_diffusion_v3"] = GreedyRelayDiffusionPolicyV3(
            rate_weight=grd_v3_cfg.get("rate_weight", 1.0),
            importance_weight=grd_v3_cfg.get("importance_weight", 10.0),
            completion_weight=grd_v3_cfg.get("completion_weight", 1.0),
            keep_weight=grd_v3_cfg.get("keep_weight", 0.5),
            switch_weight=grd_v3_cfg.get("switch_weight", 0.2),
            hop_decay_factor=grd_v3_cfg.get("hop_decay_factor", 0.25),
            max_path_links=grd_v3_cfg.get("max_path_links", 3),
            wait_urgency_tau_ratio=grd_v3_cfg.get("wait_urgency_tau_ratio", 0.8),
            ignore_consumption=grd_v3_cfg.get("ignore_consumption", False),
            include_stocked_unavailable=grd_v3_cfg.get("include_stocked_unavailable", True),
        )
    gm_cfg = base_cfg.get("greedy_matching", {})
    if _enabled("greedy_matching"):
        policies["greedy_matching"] = GreedyMatchingPolicy(
            rate_weight=gm_cfg.get("rate_weight", 1.0),
            level_weight=gm_cfg.get("level_weight", 0.5),
            demand_weight=gm_cfg.get("demand_weight", 1.0),
            keep_weight=gm_cfg.get("keep_weight", 0.5),
        )
    if args.rl_checkpoint:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if template_env is None:
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

    # NOTE: the env-replay MILP baseline was removed — it produced poor results
    # (SR ~0.19) and is not a valid upper bound (solver ideal flow vs env replay
    # diverge). The MILP offline ideal upper bound runs separately via
    # scripts/compute_milp_upper_bound.py (UI 'milp' checkbox triggers it).
    evaluator = Evaluator(env_builder)
    episodes, step_rows = evaluator.compare(
        policies,
        num_episodes=args.episodes,
        seeds=seeds,
        collect_steps=True,
        start_seed=int(profile.get("start_seed", 0)),
    )

    # Per-policy output: each policy gets its own subdirectory
    from collections import defaultdict
    by_policy: dict[str, list] = defaultdict(list)
    for ep in episodes:
        by_policy[ep.policy].append(ep)
    for policy_name, policy_eps in by_policy.items():
        pdir = out_dir / policy_name
        pdir.mkdir(parents=True, exist_ok=True)
        write_episodes_csv(policy_eps, pdir / "episodes.csv")
        write_summary_json(policy_eps, pdir / "summary.json")
    # Also write aggregate summary at root level (incremental: accumulates runs)
    merge_eval_summary(
        episodes,
        out_dir / "summary.json",
        seeds=seeds,
        meta={
            "episodes": args.episodes,
            "episode_steps": episode_steps,
            "start_mode": args.episode_start_mode,
        },
    )
    for policy, rows in step_rows.items():
        pdir = out_dir / policy
        pdir.mkdir(parents=True, exist_ok=True)
        write_steps_csv(rows, pdir / "steps.csv")

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
