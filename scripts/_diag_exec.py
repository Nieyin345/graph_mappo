"""One-shot diagnostic: replay a MILP demo's per-slot actions in the real env
and find WHERE the ideal-flow service breaks down vs the real serve mechanism.

Loads the gold actions from a demo file, steps the env with them, and prints
per-slot: activated edges, keys generated, keys served, pending request demand,
and (every K slots) the top pending request paths vs QKP levels on those paths.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from qkd_rl.core.config import ConfigValidator
from qkd_rl.env.factory import build_env_from_config

import importlib.util
_tp_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_tp_spec)
_tp_spec.loader.exec_module(_tp)
build_validation_env_config = _tp.build_validation_env_config
load_validation_profile = _tp.load_validation_profile
resolve_seeds = _tp.resolve_seeds


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=str, default="outputs/milp_demos_global5/episode_0000.pt")
    parser.add_argument("--every", type=int, default=20, help="print detail every N slots")
    args = parser.parse_args()

    demo = torch.load(str(ROOT / args.demo), map_location="cpu", weights_only=False)
    meta = demo["env"]
    profile = load_validation_profile(ROOT / "configs" / "global.yaml")
    config = build_validation_env_config(
        profile,
        include_baselines=False,
        episode_steps=meta["episode_steps"],
        start_mode=profile["start_mode"],
    )
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    obs = env.reset(seed=meta["seed"], start_seed=meta["start_seed"])

    steps = demo["trajectory"]
    total_gen = 0.0
    total_served = 0.0
    total_added = 0.0
    total_overflow = 0.0
    exp_keys = 0.0
    for i, step_data in enumerate(steps):
        actions = step_data["milp_actions"]
        obs, reward, term, trunc, info = env.step(actions)
        gen = info.get("generated_keys", 0.0)
        served = info.get("served_keys", 0.0)
        added = info.get("added_keys", 0.0)
        overflow = info.get("overflow_keys", 0.0)
        expired = info.get("expired_keys", 0.0)
        total_gen += gen
        total_served += served
        total_added += added
        total_overflow += overflow
        exp_keys += expired
        if i % args.every == 0 or i == len(steps) - 1:
            pend = env.requests.get_pending()
            pend_amt = sum(max(0.0, r.amount - r.served_amount) for r in pend)
            print(f"[t={int(obs.state.t)}] gen={gen:,.0f} added={added:,.0f} overflow={overflow:,.0f} "
                  f"served={served:,.0f} expired_keys={expired:,.0f} pending_amt={pend_amt:,.0f} "
                  f"#pending_req={len(pend)}", flush=True)
        if term or trunc:
            break

    arrived = float(env.metrics.arrived_keys)
    print(f"\nTOTAL: gen={total_gen:,.0f} added={total_added:,.0f} overflow={total_overflow:,.0f} "
          f"served={total_served:,.0f} expired_keys={exp_keys:,.0f} arrived={arrived:,.0f} "
          f"sr={total_served/arrived:.4f}", flush=True)
    print(f"demo metadata: plan_sr={demo['success_rate']:.4f} exec_sr={demo['executed_sr']:.4f}")


if __name__ == "__main__":
    main()
