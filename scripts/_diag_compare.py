"""One-shot: compare REAL env success rate of greedy (relay-diffusion) vs the
MILP plan on the same validation episode. Proves whether the MILP's ideal-flow
plan is the problem or the env's serve mechanism caps everyone.

Runs the greedy policy (V3 with importance_weight=10) on the same (seed,
start_seed) as a demo, and reports the real executed SR.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from qkd_rl.core.config import ConfigValidator
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.baselines.greedy_relay_diffusion import GreedyRelayDiffusionPolicyV3

import importlib.util
_tp_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_tp_spec)
_tp_spec.loader.exec_module(_tp)
build_validation_env_config = _tp.build_validation_env_config
load_validation_profile = _tp.load_validation_profile


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=str, default="outputs/milp_demos_global5/episode_0000.pt")
    parser.add_argument("--policy", type=str, default="greedy",
                        help="greedy | rate | random")
    args = parser.parse_args()

    demo = torch.load(str(ROOT / args.demo), map_location="cpu", weights_only=False)
    meta = demo["env"]
    profile = load_validation_profile(ROOT / "configs" / "global.yaml")
    config = build_validation_env_config(
        profile, include_baselines=False,
        episode_steps=meta["episode_steps"], start_mode=profile["start_mode"])
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    obs = env.reset(seed=meta["seed"], start_seed=meta["start_seed"])

    if args.policy == "greedy":
        pol = GreedyRelayDiffusionPolicyV3()
    elif args.policy == "rate":
        from qkd_rl.baselines.greedy_rate import GreedyRatePolicy
        pol = GreedyRatePolicy()
    else:
        from qkd_rl.baselines.random_policy import RandomPolicy
        pol = RandomPolicy(seed=meta["seed"])

    total_served = 0.0
    total_gen = 0.0
    from qkd_rl.env.action_space import NodeActionSpace
    for step in range(meta["episode_steps"]):
        actions, scores = pol.act(obs)
        # legacy baselines emit single-port {node: target}; under
        # mutual_choice the resolver needs BOTH endpoints to agree, so for each
        # arc u->v also set v's rx = u (dual-port proposals).
        first = next(iter(actions.values())) if actions else None
        if isinstance(first, str):
            dual = {n: [NodeActionSpace.IDLE, NodeActionSpace.IDLE] for n in actions}
            for n, t in actions.items():
                if t == NodeActionSpace.IDLE or t == n:
                    continue
                dual[n][0] = t
                dual[t][1] = n
            actions = {n: (tx, rx) for n, (tx, rx) in dual.items()}
        obs, reward, term, trunc, info = env.step(actions)
        total_served += info.get("served_keys", 0.0)
        total_gen += info.get("generated_keys", 0.0)
        if term or trunc:
            break
    arrived = float(env.metrics.arrived_keys)
    print(f"POLICY={args.policy} seed={meta['seed']}: gen={total_gen:,.0f} served={total_served:,.0f} "
          f"arrived={arrived:,.0f} REAL SR={total_served/arrived:.4f}", flush=True)
    print(f"  (vs MILP plan exec SR={demo['executed_sr']:.4f})", flush=True)


if __name__ == "__main__":
    main()
