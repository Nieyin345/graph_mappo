"""One-action v3 counterfactuals on frozen evaluation request seeds.

Each sampled expert state is cloned. One branch receives a deterministic v3
action; both branches then follow the same expert for a fixed horizon. The
expert branch continues to the next sample. This isolates the effect of one
action on service without assuming that matching the expert is inherently good.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qkd_rl.core.config import load_config
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.model_zoo import build_model
from qkd_rl.rl.algos.checkpoint import load_checkpoint
from qkd_rl.rl.algos.policy import MAPPOPolicy
from scripts.diag.probe_v3_counterfactual import clone_env, expert_for
from scripts.eval.frozen_eval import EPISODE_STEPS, WINDOWS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run directory with experiment_config.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--window", choices=WINDOWS, default="A")
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 105, 110])
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--stride", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    config = load_config([Path(args.run) / "experiment_config.yaml"])
    spec = WINDOWS[args.window]
    config["env"].update({
        "episode_start_mode": "random_day", "episode_steps": EPISODE_STEPS,
        "continuous": False, "activation_window_start_day": spec["start_day"],
        "activation_window_end_day": spec["end_day"],
        "activation_window_days": spec["end_day"] - spec["start_day"],
    })
    config["scenario"]["time_limit"]["days"] = spec["end_day"] + max(
        1, math.ceil(EPISODE_STEPS / int(config["env"].get("day_steps", 1440)))
    )
    env = build_env_from_config(config)
    model = build_model(env.action_resolver.action_space, config)
    model.load_state_dict(load_checkpoint(args.checkpoint, "cpu").model_state)
    model.eval()
    policy = MAPPOPolicy(model, "cpu")
    rows = []
    baselines = {}
    for seed in args.seeds:
        expert = expert_for(env)
        obs = env.reset(seed=seed, start_seed=seed)
        step = 0
        while step < EPISODE_STEPS:
            if step % args.stride == 0 and step + args.horizon <= EPISODE_STEPS:
                alt = clone_env(env)
                alt_expert = expert_for(alt)
                with torch.no_grad():
                    policy_step = policy.act(obs, deterministic=True, build_scores=False)
                exp_actions, exp_scores = expert.act(obs)
                alt_obs, _, _, _, alt_info = alt.step(
                    policy_step.actions, policy_step.action_scores,
                    expected_matched_edges=list(policy_step.matched_edges or []),
                )
                obs, _, _, _, exp_info = env.step(exp_actions, exp_scores)
                first_exp, first_alt = set(env.last_matched_arcs), set(alt.last_matched_arcs)
                exp_served = float(exp_info["served_keys"])
                alt_served = float(alt_info["served_keys"])
                exp_generated = float(exp_info["generated_keys"])
                alt_generated = float(alt_info["generated_keys"])
                if exp_info["arrived_keys"] != alt_info["arrived_keys"]:
                    raise RuntimeError("Counterfactual request streams differ at first step")
                for _ in range(1, args.horizon):
                    ea, es = expert.act(obs)
                    pa, ps = alt_expert.act(alt_obs)
                    obs, _, _, _, exp_info = env.step(ea, es)
                    alt_obs, _, _, _, alt_info = alt.step(pa, ps)
                    if exp_info["arrived_keys"] != alt_info["arrived_keys"]:
                        raise RuntimeError("Counterfactual request streams differ after first step")
                    exp_served += float(exp_info["served_keys"])
                    alt_served += float(alt_info["served_keys"])
                    exp_generated += float(exp_info["generated_keys"])
                    alt_generated += float(alt_info["generated_keys"])
                row = {
                    "seed": seed, "step": step, "horizon": args.horizon,
                    "expert_arcs": len(first_exp), "policy_arcs": len(first_alt),
                    "common_arcs": len(first_exp & first_alt),
                    "expert_only": len(first_exp - first_alt),
                    "policy_only": len(first_alt - first_exp),
                    "expert_served": exp_served, "policy_served": alt_served,
                    "served_delta_policy_minus_expert": alt_served - exp_served,
                    "expert_generated": exp_generated, "policy_generated": alt_generated,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
                step += args.horizon
            else:
                actions, scores = expert.act(obs)
                obs, _, _, _, _ = env.step(actions, scores)
                step += 1
        baselines[str(seed)] = env.metrics.episode_summary()
        print(f"baseline seed={seed} success={baselines[str(seed)]['success_rate']:.6f}", flush=True)
    result = {"window": args.window, "checkpoint": args.checkpoint,
              "probes": rows, "baseline": baselines}
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} probes to {args.out}", flush=True)


if __name__ == "__main__":
    main()
