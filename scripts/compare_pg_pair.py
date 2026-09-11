"""Compare Path-Greedy-Pair (GS-pair agg + stocked-edge graph) vs old
PathGreedy vs V3 vs Random, on identical deterministic env episodes.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import importlib.util
_tp_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp_ = importlib.util.module_from_spec(_tp_spec)
_tp_spec.loader.exec_module(_tp_)
build_validation_env_config = _tp_.build_validation_env_config
load_validation_profile = _tp_.load_validation_profile

from qkd_rl.core.config import ConfigValidator
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.baselines.greedy_relay_diffusion import GreedyRelayDiffusionPolicyV3
from qkd_rl.baselines._matching import greedy_matching_actions
from qkd_rl.baselines.path_greedy import PathGreedyPolicy
from qkd_rl.baselines.path_greedy_pair import PathGreedyPairPolicy


def v3_dual_actions(obs, w) -> dict[str, tuple]:
    scorer = GreedyRelayDiffusionPolicyV3(
        rate_weight=w[0], importance_weight=w[1], completion_weight=w[2],
        keep_weight=w[3], switch_weight=w[4])
    edge_scores = scorer.score_edges(obs)
    rates = {e: float(obs.state.edge_windows[e].rates[0]) for e in obs.physical_edge_ids}
    actions, _ = greedy_matching_actions(obs, edge_scores, tie_rates=rates)
    dual: dict[str, tuple] = {}
    for node_id in obs.node_ids:
        dual[node_id] = (NodeActionSpace.IDLE, NodeActionSpace.IDLE)
    for u, t in actions.items():
        if t == NodeActionSpace.IDLE or t == u:
            continue
        cur = list(dual[u]); cur[0] = t; dual[u] = tuple(cur)
        cur = list(dual[t]); cur[1] = u; dual[t] = tuple(cur)
    return dual


def random_dual_actions(obs, rng) -> dict[str, tuple]:
    """Independent per-node random Tx among valid candidates (Rx idle)."""
    actions = {}
    for node in obs.node_ids:
        cands = [c for c in obs.action_candidates.get(node, ()) if c != NodeActionSpace.IDLE]
        tx = rng.choice(cands) if cands else NodeActionSpace.IDLE
        actions[node] = (tx, NodeActionSpace.IDLE)
    return actions


def run(profile, seed, start_seed, steps, policy_type, w=None, rng_seed=0, trace=False):
    config = build_validation_env_config(
        profile, include_baselines=False, episode_steps=steps, start_mode=profile["start_mode"])
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    env.reset(seed=seed, start_seed=start_seed)
    obs = env._obs_cache[1] if env._obs_cache else None
    rw = w[7] if w else 1.0
    df = w[8] if (w and len(w) > 8) else 1.0
    pg_old = PathGreedyPolicy(rate_weight=rw, dense_fill=bool(df >= 0.5))
    ws = w[5] if w else 0.5
    wr = w[6] if w else 0.5
    pg_pair = PathGreedyPairPolicy(w_stock=ws, w_rate=wr)
    scorer = GreedyRelayDiffusionPolicyV3(
        rate_weight=w[0], importance_weight=w[1], completion_weight=w[2],
        keep_weight=w[3], switch_weight=w[4]) if w else None
    rng = random.Random(rng_seed)
    total_served = 0.0
    n_act = 0
    for step in range(steps):
        if policy_type == "pg_pair":
            actions, _ = pg_pair.act(obs)
        elif policy_type == "pg_old":
            actions, _ = pg_old.act(obs)
        elif policy_type == "v3":
            actions = v3_dual_actions(obs, w)
        else:
            actions = random_dual_actions(obs, rng)
        obs, _, term, trunc, info = env.step(actions)
        total_served += info.get("served_keys", 0.0)
        n_act += len(getattr(env, "last_activated_edges", []) or [])
        if trace and step % 40 == 0:
            print(f"[{policy_type} step={step}] activated_step="
                  f"{len(getattr(env,'last_activated_edges',[]) or [])} "
                  f"served_step={info.get('served_keys',0):,.0f}", flush=True)
        if term or trunc:
            break
    arrived = float(env.metrics.arrived_keys)
    sr = total_served / arrived if arrived > 0 else 0.0
    return sr, total_served, n_act / max(1, steps)


def main():
    ap = argparse.ArgumentParser(description="PG-Pair vs PG-Old vs V3 vs Random.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start-seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=90)
    ap.add_argument("--w", nargs=9, type=float, default=[1.0, 10.0, 1.0, 0.5, 0.2, 0.5, 0.5, 1.0, 1.0])
    ap.add_argument("--tau", type=float, default=0.8)
    args = ap.parse_args()
    profile = load_validation_profile(ROOT / "configs" / "global.yaml")

    pair = run(profile, args.seed, args.start_seed, args.steps, "pg_pair", trace=True)
    old = run(profile, args.seed, args.start_seed, args.steps, "pg_old", trace=True)
    v3 = run(profile, args.seed, args.start_seed, args.steps, "v3", w=args.w, trace=True)
    rnd = run(profile, args.seed, args.start_seed, args.steps, "random", trace=True)

    print(f"\n==== seed={args.seed} start_seed={args.start_seed} steps={args.steps} ====")
    print(f"PG-Pair  SR = {pair[0]:.4f} (served={pair[1]:,.0f}, activ/slot={pair[2]:.1f})")
    print(f"PG-Old+rate SR = {old[0]:.4f} (served={old[1]:,.0f}, activ/slot={old[2]:.1f})")
    print(f"V3       SR = {v3[0]:.4f} (served={v3[1]:,.0f}, activ/slot={v3[2]:.1f})")
    print(f"Random   SR = {rnd[0]:.4f} (served={rnd[1]:,.0f}, activ/slot={rnd[2]:.1f})")
    print(f"delta(PG-Pair - V3) = {pair[0]-v3[0]:+.4f}")
    print(f"delta(PG-Pair - PG-Old) = {pair[0]-old[0]:+.4f}")


if __name__ == "__main__":
    main()