"""One-shot diagnostic: WHY does the MILP plan fail to serve in the real env?

For a few sample slots, inspect:
  - the pending requests (EDF order) and their src/dst;
  - whether a positive-key path exists (routing.prepare_serve state);
  - the hop levels along the shortest path vs the request amount;
  - how many requests got zero service and the reason.
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
    parser.add_argument("--slots", type=str, default="0,40,80,120,160,200,239")
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

    watch = {int(s) for s in args.slots.split(",")}
    for i, step_data in enumerate(demo["trajectory"]):
        actions = step_data["milp_actions"]
        if i in watch:
            pend = env.requests.get_pending()
            pend_sorted = sorted(pend, key=lambda r: (r.deadline_t, r.arrival_t))
            print(f"\n=== slot {i} t={int(obs.state.t)} pending={len(pend)} "
                  f"pending_amt={sum(max(0,r.amount-r.served_amount) for r in pend):,.0f}", flush=True)
            # how many requests can currently be served (positive path + enough)?
            env.routing.prepare_serve(env.qkp)
            servable = 0
            for r in pend_sorted[:40]:
                rem = max(0.0, r.amount - r.served_amount)
                path = env.routing._usable_cached_path(r, env.qkp)
                if path is None:
                    comp = getattr(env.routing, "_pos_component", None)
                    reason = "no positive-key path (disconnected)" if (comp is not None and comp.get(r.src_gs, -1) != comp.get(r.dst_gs, -1)) else "no path"
                    if i % 40 == 0:
                        print(f"  REQ {r.request_id} {r.src_gs}->{r.dst_gs} rem={rem:,.0f} dl={r.deadline_t} -> {reason}", flush=True)
                    continue
                levels = [env.qkp.get_level(e) for e in path]
                bottleneck = min(levels) if levels else 0.0
                if bottleneck >= rem - 1e-9:
                    servable += 1
                elif i % 40 == 0:
                    print(f"  REQ {r.request_id} {r.src_gs}->{r.dst_gs} rem={rem:,.0f} path={path} "
                          f"bottleneck={bottleneck:,.0f} -> partial", flush=True)
            print(f"  -> fully servable among top-40 EDF: {servable}", flush=True)
            # QKP stats
            pos = env.qkp.positive
            n_pos = len(pos)
            total_level = sum(env.qkp.levels.values())
            total_cap = sum(env.qkp.capacities.values())
            print(f"  -> QKP: #positive_edges={n_pos} total_level={total_level:,.0f}/{total_cap:,.0f} "
                  f"util={total_level/total_cap:.3f}", flush=True)
        obs, reward, term, trunc, info = env.step(actions)
        if term or trunc:
            break


if __name__ == "__main__":
    main()
