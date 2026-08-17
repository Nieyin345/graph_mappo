"""MILP upper bound: single sliding window, full topology, no env replay.
Usage: python scripts/compute_milp_upper_bound.py --window-steps 360 --time-limit 120
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import warnings
warnings.filterwarnings("ignore", message=".*iCCP.*")

from qkd_rl.core.config import (ConfigValidator, deep_merge, load_config)
from qkd_rl.env.factory import load_default_config, build_env_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start-step", type=int, default=475200)
    parser.add_argument("--window-steps", type=int, default=360, help="Window size (default 360 = 6h).")
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--out", type=str, default="outputs/eval/milp_ub")
    parser.add_argument("--max-requests", type=int, default=512)
    parser.add_argument("--max-paths", type=int, default=256)
    parser.add_argument("--max-hops", type=int, default=10)
    args = parser.parse_args()

    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config = deep_merge(config, load_config([ROOT / "configs" / "global.yaml"]))
    ConfigValidator().validate(config)

    env = build_env_from_config(config)
    obs = env.reset(seed=args.seed, start_seed=args.seed + 1000)

    from qkd_rl.baselines.receding_horizon_milp import RecedingHorizonMILPPolicy
    pol = RecedingHorizonMILPPolicy(
        config, window_steps=args.window_steps, max_requests=args.max_requests,
        max_paths_per_request=args.max_paths, max_path_hops=args.max_hops,
        time_limit_s=args.time_limit,
    )
    pol._ensure_init(obs)
    pol._gen_t = args.start_step
    horizon = args.window_steps

    # Generate all requests for the window (for total_arrived counting).
    # Then reset gen_t so solve_window's internal _preview_requests generates them.
    pol._preview_requests(args.start_step, horizon)
    total_arrived = sum(
        req.amount - getattr(req, "served_amount", 0.0)
        for at, req in pol._arrivals_cache
        if args.start_step <= at < args.start_step + horizon
    )
    # Deduplicate the arrivals cache (solve_window will add the same requests again)
    seen: set[str] = set()
    deduped: list[tuple[int, KeyRequest]] = []
    for at, req in pol._arrivals_cache:
        if req.request_id not in seen:
            seen.add(req.request_id)
            deduped.append((at, req))
    pol._arrivals_cache = deduped
    pol._gen_t = args.start_step  # reset again
    print(f"arrivals: {len(deduped)}, volume: {total_arrived:,.0f}")

    # Build holder
    class _Light: pass
    state = _Light()
    state.t = args.start_step
    state.pending_requests = []  # all requests arrive at or after start_step, handled by _preview_requests
    state.qkp_snapshot = env.qkp.snapshot() if hasattr(env, "qkp") else {}
    state.qkp_capacity = {}
    if hasattr(env, "qkp"):
        for eid in obs.physical_edge_ids:
            try: state.qkp_capacity[eid] = env.qkp.get_capacity(eid)
            except: pass
    state.edge_windows = obs.state.edge_windows
    holder = _Light()
    holder.state = state
    holder.node_ids = obs.node_ids
    holder.physical_edge_ids = obs.physical_edge_ids

    w0 = time.perf_counter()
    outcome = pol.solve_window(holder)
    solve_s = time.perf_counter() - w0

    total_served = sum(outcome.flow_by_request.values()) if outcome.flow_by_request else 0.0
    sr = total_served / max(1e-9, total_arrived)
    print(f"solve={solve_s:.1f}s status={outcome.status} served={total_served:,.0f} arrived={total_arrived:,.0f}")
    print(f"SR upper bound: {sr:.4f}")

    # ── Analysis ─────────────────────────────────────────────────
    # 1. Edge activation patterns
    activation_counts: dict[str, int] = {}
    activation_slots: dict[str, int] = {}  # edge_id -> number of slots activated
    for tau, edges_list in outcome.activation_plan.items():
        for eid in edges_list:
            activation_counts[eid] = activation_counts.get(eid, 0) + 1
            activation_slots[eid] = activation_slots.get(eid, 0) + 1
    # Top 10 most activated edges
    top_edges = sorted(activation_counts.items(), key=lambda x: -x[1])[:10]
    print(f"\nTop 10 most activated edges (slots):")
    for eid, count in top_edges:
        u, v = pol._parse_edge(eid)
        print(f"  {eid} ({u}--{v}): {count}/{args.window_steps} slots")

    # 2. Edge type breakdown
    type_counts: dict[str, int] = {t: 0 for t in ["GS-HAP", "GS-SAT", "HAP-SAT", "SAT-SAT", "HAP-HAP", "GS-GS", "other"]}
    def classify_edge(eid: str) -> str:
        u, v = pol._parse_edge(eid)
        if u is None: return "other"
        types = []
        for n in [u, v]:
            if n.startswith("GS_"): types.append("GS")
            elif n.startswith("HAP_") or n.startswith("HAP"): types.append("HAP")
            elif n.startswith("Sat_") or n.startswith("SAT_"): types.append("SAT")
            else: types.append("GS")  # City names = GS
        return f"{types[0]}-{types[1]}"
    for eid, count in activation_counts.items():
        ct = classify_edge(eid)
        if ct in type_counts:
            type_counts[ct] += count
        else:
            type_counts["other"] += count
    print(f"\nEdge activation by type (total slots):")
    for ct, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        if cnt > 0: print(f"  {ct}: {cnt}")

    # 3. Request flow analysis
    print(f"\nRequest flow detail ({len(outcome.flow_detail)} entries):")
    path_lengths: list[int] = []
    slot_delays: list[int] = []  # arrival -> service slot delta
    for req_id, path_indices, tau, amt in outcome.flow_detail:
        path_lengths.append(len(path_indices))
        # Find arrival time
        for at, req in pol._arrivals_cache:
            if req.request_id == req_id:
                if tau >= (at - args.start_step):
                    slot_delays.append(tau - (at - args.start_step))
                break
    if path_lengths:
        from collections import Counter
        pl = Counter(path_lengths)
        print(f"  Path lengths: {dict(sorted(pl.items()))}")
        if slot_delays:
            print(f"  Service delay (slots from arrival): avg={sum(slot_delays)/len(slot_delays):.1f} max={max(slot_delays)}")

    # 4. Activation persistence: how often edges stay active across consecutive slots
    consecutive_counts: list[int] = []
    for eid in activation_counts:
        active = [tau for tau, edges_list in outcome.activation_plan.items() if eid in edges_list]
        if len(active) > 1:
            # count consecutive runs
            runs = 1
            for i in range(1, len(active)):
                if active[i] == active[i-1] + 1:
                    runs += 1
                else:
                    consecutive_counts.append(runs)
                    runs = 1
            consecutive_counts.append(runs)
    if consecutive_counts:
        avg_run = sum(consecutive_counts) / len(consecutive_counts)
        print(f"  Avg consecutive activation: {avg_run:.1f} slots")

    summary = {"seed": args.seed, "start_step": args.start_step,
        "window_steps": args.window_steps, "time_limit_s": args.time_limit,
        "total_arrived": total_arrived, "total_served": total_served,
        "success_rate_upper_bound": sr, "solve_s": solve_s, "status": outcome.status}
    out_dir = Path(ROOT) / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()