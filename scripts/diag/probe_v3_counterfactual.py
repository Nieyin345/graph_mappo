"""Compare expert and v3 actions from the same live state over a short horizon.

The environment is replayed under the expert. At each sampled slot a branch
receives one v3 action; both branches then follow the expert. Sharing the
read-only scenario and rate provider keeps cloning small. Request arrivals
are checked slot by slot, so differences are attributable to the first action.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qkd_rl.baselines.path_greedy import PathScoreGreedy
from qkd_rl.baselines.serve_probe import ServeProbe
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.model_zoo import build_model
from qkd_rl.rl.algos.checkpoint import load_checkpoint
from qkd_rl.rl.algos.policy import MAPPOPolicy
from scripts.train.collect_pg_phased_trajectories import build_base_config


def expert_for(env):
    return PathScoreGreedy(
        weights=(1.0, 10.0, 1.0, 0.5, 0.2),
        phased=True,
        principles=False,
        router=ServeProbe(env),
    )


def clone_env(env):
    # These objects contain the static year-long topology/rate data. They are
    # never mutated by step(); all inventory, requests, RNGs, and graph caches
    # remain private to the cloned branch.
    memo = {
        id(env.scenario): env.scenario,
        id(env.rate_provider): env.rate_provider,
    }
    clone = copy.deepcopy(env, memo)
    # LazyEdgeWindows intentionally drops its H5 provider when copied/pickled.
    # Rebuild the current state through the shared read-only provider instead
    # of reusing the copied observation cache.
    clone._obs_cache = None
    return clone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--days", nargs="+", type=int, default=[5])
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--max-probes", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    config = build_base_config(
        "configs/supervised_train.yaml", ["qkd_rl/model_zoo/v3/config.yaml"]
    )
    ckpt = load_checkpoint(args.checkpoint, "cpu")
    model = build_model(build_env_from_config(config).action_resolver.action_space, ckpt.config)
    model.load_state_dict(ckpt.model_state)
    model.eval()
    policy = MAPPOPolicy(model, "cpu")
    rows = []
    for day in args.days:
        day_config = copy.deepcopy(config)
        day_config["env"]["episode_start_day"] = day
        env = build_env_from_config(day_config)
        expert = expert_for(env)
        obs = env.reset(seed=day)
        with gzip.open(ROOT / "outputs" / "trajs_v3_50d" / f"day_{day:04d}.pkl.gz", "rb") as fh:
            records = pickle.load(fh)
        step = 0
        probes = 0
        while step < min(len(records), args.max_probes * args.stride):
            if step % args.stride == 0 and probes < args.max_probes:
                alt = clone_env(env)
                alt_expert = expert_for(alt)
                with torch.no_grad():
                    policy_step = policy.act(obs, deterministic=True, build_scores=False)
                expert_actions, expert_scores = expert.act(obs)
                alt_obs, _, _, _, alt_info = alt.step(
                    policy_step.actions, policy_step.action_scores,
                    expected_matched_edges=list(policy_step.matched_edges or []),
                )
                obs, _, _, _, exp_info = env.step(expert_actions, expert_scores)
                if set(env.last_matched_arcs) != {tuple(a) for a in records[step]["arcs"]}:
                    raise RuntimeError(f"Expert replay diverged from saved trajectory at day={day} step={step}")
                if alt_info["arrived_keys"] != exp_info["arrived_keys"]:
                    raise RuntimeError("Counterfactual branches received different requests")
                first_expert = set(env.last_matched_arcs)
                first_policy = set(alt.last_matched_arcs)
                exp_served = float(exp_info["served_keys"])
                alt_served = float(alt_info["served_keys"])
                exp_generated = float(exp_info["generated_keys"])
                alt_generated = float(alt_info["generated_keys"])
                for offset in range(1, args.horizon):
                    exp_actions, exp_scores = expert.act(obs)
                    alt_actions, alt_scores = alt_expert.act(alt_obs)
                    obs, _, _, _, exp_info = env.step(exp_actions, exp_scores)
                    alt_obs, _, _, _, alt_info = alt.step(alt_actions, alt_scores)
                    if alt_info["arrived_keys"] != exp_info["arrived_keys"]:
                        raise RuntimeError("Counterfactual branches received different future requests")
                    if set(env.last_matched_arcs) != {tuple(a) for a in records[step + offset]["arcs"]}:
                        raise RuntimeError(f"Expert replay diverged at day={day} step={step + offset}")
                    exp_served += float(exp_info["served_keys"])
                    alt_served += float(alt_info["served_keys"])
                    exp_generated += float(exp_info["generated_keys"])
                    alt_generated += float(alt_info["generated_keys"])
                row = {
                    "day": day, "step": step, "horizon": args.horizon,
                    "expert_arcs": len(first_expert), "policy_arcs": len(first_policy),
                    "common_arcs": len(first_expert & first_policy),
                    "expert_only": len(first_expert - first_policy),
                    "policy_only": len(first_policy - first_expert),
                    "expert_served": exp_served, "policy_served": alt_served,
                    "served_delta_policy_minus_expert": alt_served - exp_served,
                    "expert_generated": exp_generated, "policy_generated": alt_generated,
                    "generated_delta_policy_minus_expert": alt_generated - exp_generated,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
                step += args.horizon
                probes += 1
            else:
                actions, scores = expert.act(obs)
                obs, _, _, _, _ = env.step(actions, scores)
                if set(env.last_matched_arcs) != {tuple(a) for a in records[step]["arcs"]}:
                    raise RuntimeError(f"Expert replay diverged at day={day} step={step}")
                step += 1
    result = {"checkpoint": args.checkpoint, "probes": rows}
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} probes to {args.out}", flush=True)


if __name__ == "__main__":
    main()
