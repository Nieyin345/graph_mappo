"""Diagnose the gap between MILP global plan and real-env execution.

Builds the env exactly like generate_milp_demos.py, solves ONE global MILP
over the whole episode (god's-eye view), then replays the plan step by step.
Per slot it prints:
  * planned arcs (what the MILP wants to activate)
  * arcs actually emitted by act() and actually activated by the resolver
  * generated / served / overflow keys
  * QKP utilisation and pending demand
At the end it compares the planned per-slot serve amounts (from the MILP
flow_detail) with what the env actually served, and counts how many planned
arcs were dropped.
"""
from __future__ import annotations
import argparse, importlib.util, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from qkd_rl.core.config import ConfigValidator
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.baselines.receding_horizon_milp import RecedingHorizonMILPPolicy
from qkd_rl.env.action_space import NodeActionSpace

_tp_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_tp_spec)
_tp_spec.loader.exec_module(_tp)
build_validation_env_config = _tp.build_validation_env_config
load_validation_profile = _tp.load_validation_profile
resolve_seeds = _tp.resolve_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start-seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--window", type=int, default=0, help="window steps (0 = full episode)")
    ap.add_argument("--replan", type=int, default=0, help="replan cadence (0 = open-loop once)")
    ap.add_argument("--switch-decay", type=float, default=0.5)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--max-requests", type=int, default=64)
    ap.add_argument("--relax", action="store_true", help="use the relaxed multi-path upper-bound model")
    ap.add_argument("--single-path", action="store_true", help="one path per request per slot (env semantics)")
    ap.add_argument("--print-req", action="store_true", help="print per-request failures")
    args = ap.parse_args()

    profile = load_validation_profile(ROOT / "configs" / "global.yaml")
    config = build_validation_env_config(
        profile, include_baselines=False,
        episode_steps=args.steps, start_mode=profile["start_mode"])
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    start_seed = args.start_seed if args.start_seed is not None else profile["start_seed"]
    obs = env.reset(seed=args.seed, start_seed=start_seed)
    t0 = int(obs.state.t)
    print(f"reset t={t0} (day {t0/1440:.1f}) seed={args.seed} start_seed={start_seed}", flush=True)

    pol = RecedingHorizonMILPPolicy(
        config, window_steps=(args.window if args.window > 0 else args.steps),
        max_requests=args.max_requests,
        max_paths_per_request=8, max_path_hops=6,
        time_limit_s=args.time_limit, replan_every_steps=(args.replan if args.replan > 0 else None),
        switch_decay=args.switch_decay, single_path=args.single_path,
        final_inventory_weight=0.0,
        relax_multi_path=args.relax,
    )
    pol._ensure_init(obs, gen=env.request_generator)
    actions, _ = pol.act(obs)  # triggers the single global solve
    oc = pol.last_outcome
    print(f"GLOBAL plan: served={oc.served_amount:,.0f} modeled={oc.modeled_amount:,.0f} "
          f"plan_sr={oc.served_amount/max(1e-9,oc.modeled_amount):.4f} "
          f"status={oc.status} gap={oc.mip_gap:.4f} solve={oc.solve_time_s:.1f}s", flush=True)

    # planned serve per slot summed over requests (for gap comparison)
    plan_serve_by_slot: dict[int, float] = {}
    plan_arcs_by_slot: dict[int, list[tuple[str, str]]] = oc.activation_plan
    for _req_id, _path, tau, amt in oc.flow_detail:
        plan_serve_by_slot[tau] = plan_serve_by_slot.get(tau, 0.0) + amt

    total_gen = 0.0
    total_served = 0.0
    total_overflow = 0.0
    total_plan_arcs = 0
    total_activated = 0
    dropped_arcs = 0
    failed_detail = {}

    for step in range(args.steps):
        actions, _ = pol.act(obs)
        # arcs act() actually emitted (this is the plan for the current slot from
        # whichever solve produced it — correct across re-plans)
        emitted: set[tuple[str, str]] = set()
        idle = NodeActionSpace.IDLE
        for node_id, (tx, rx) in actions.items():
            if tx != idle and tx != node_id:
                emitted.add((node_id, tx))
            if rx != idle and rx != node_id:
                emitted.add((rx, node_id))
        planned = set(emitted)
        obs, reward, term, trunc, info = env.step(actions, {})
        activated = set(getattr(env, "last_activated_edges", []) or [])
        # count arcs actually resolved (activated_edges are undirected pair ids)
        gen = info.get("generated_keys", 0.0)
        served = info.get("served_keys", 0.0)
        overflow = info.get("overflow_keys", 0.0)
        total_gen += gen
        total_served += served
        total_overflow += overflow
        total_plan_arcs += len(planned)
        total_activated += len(activated)
        # dropped = planned arc whose undirected pair was not activated
        for u, v in planned:
            if f"{u}__{v}" not in activated and f"{v}__{u}" not in activated:
                dropped_arcs += 1
        if step % args.every == 0 or step == args.steps - 1:
            pend = env.requests.get_pending()
            pend_amt = sum(max(0.0, r.amount - r.served_amount) for r in pend)
            qkp = env.qkp
            pos = qkp.positive
            print(f"[t={step}] plan_arcs={len(planned)} emitted={len(emitted)} "
                  f"activated={len(activated)} gen={gen:,.0f} served={served:,.0f} "
                  f"overflow={overflow:,.0f} pending_amt={pend_amt:,.0f} "
                  f"qkp_pos={len(pos)} qkp_level={sum(qkp.levels.values()):,.0f} "
                  f"/{sum(qkp.capacities.values()):,.0f}", flush=True)
        if term or trunc:
            break

    arrived = float(env.metrics.arrived_keys)
    print(f"\n==== SUMMARY seed={args.seed} ====", flush=True)
    print(f"total gen={total_gen:,.0f} served={total_served:,.0f} "
          f"overflow={total_overflow:,.0f} arrived={arrived:,.0f}", flush=True)
    print(f"EXEC SR = {total_served/max(1e-9,arrived):.4f}  (plan SR = "
          f"{oc.served_amount/max(1e-9,oc.modeled_amount):.4f})", flush=True)
    print(f"arcs: planned={total_plan_arcs} activated={total_activated} "
          f"dropped={dropped_arcs} ({dropped_arcs/max(1, total_plan_arcs)*100:.1f}%)", flush=True)
    if args.print_req:
        for req in env.requests.get_pending():
            rem = max(0.0, req.amount - req.served_amount)
            if rem > 0:
                print(f"  REMAIN {req.request_id} {req.src_gs}->{req.dst_gs} "
                      f"rem={rem:,.0f} dl_left={req.deadline_t - int(obs.state.t)}", flush=True)


if __name__ == "__main__":
    main()
