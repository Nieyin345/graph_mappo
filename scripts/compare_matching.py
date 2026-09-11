"""Compare V3's greedy exclusive matching vs the env's Blossom max-weight
directional matching, using the SAME V3 edge_scores and the SAME env.

How
---
We let the env's real ActionResolver do the matching — that is the ground
truth that gets executed. We run the same episode twice, only switching
``env.action_resolver.mode``:

1. ``mutual_choice``      : greedy policy submits only mutual single-edge
                            proposals (current V3-mode matching).
2. ``max_weight_matching``: greedy policy submits every node's full dual-port
                            candidate set + the exact V3 ``edge_scores``, so
                            the resolver runs Blossom itself (global max-weight
                            directional matching).

Both consume identical V3 ``score_edges`` output, so the ONLY difference is
matching aggressiveness/optimality. Env is deterministic -> reproducible SRs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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


def greedy_mutual_actions(obs, edge_scores, tie_rates) -> dict[str, tuple]:
    """Current V3: exclusive greedy single-port, converted to dual-port."""
    actions, _ = greedy_matching_actions(obs, edge_scores, tie_rates=tie_rates)
    dual: dict[str, tuple] = {}
    for node_id in obs.node_ids:
        dual[node_id] = [NodeActionSpace.IDLE, NodeActionSpace.IDLE]
    for u, t in actions.items():
        if t == NodeActionSpace.IDLE or t == u:
            continue
        # u transmits to t; t receives from u (mutual agreement).
        dual[u][0] = t
        dual[t][1] = u
    for k in list(dual):
        dual[k] = tuple(dual[k])
    return dual


def full_candidate_actions(obs) -> dict[str, tuple]:
    """Submit every node's full candidate neighbours on both tx and rx; the
    resolver (max_weight_matching) then picks the best global set using the
    provided edge_scores. tx and rx left at IDLE is also fine, but filling the
    dual ports lets Blossom use the full directional graph."""
    actions: dict[str, tuple] = {}
    for node_id in obs.node_ids:
        neigh = sorted(set(obs.action_candidates[node_id]) - {NodeActionSpace.IDLE})
        tx = neigh[0] if neigh else NodeActionSpace.IDLE
        rx = neigh[-1] if neigh else NodeActionSpace.IDLE
        actions[node_id] = (tx, rx)
    return actions


def run_mode(profile, seed, start_seed, steps, weights, mode, scorer,
             trace=False) -> tuple[float, float, float]:
    config = build_validation_env_config(
        profile, include_baselines=False, episode_steps=steps,
        start_mode=profile["start_mode"])
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    env.action_resolver.mode = mode  # switch resolver matching strategy
    env.reset(seed=seed, start_seed=start_seed)
    obs = env._obs_cache[1] if env._obs_cache else None
    total_served = 0.0
    total_gen = 0.0
    n_activated = 0
    for step in range(steps):
        edge_scores = scorer.score_edges(obs)
        rates = {e: float(obs.state.edge_windows[e].rates[0]) for e in obs.physical_edge_ids}
        if mode == "mutual_choice":
            actions = greedy_mutual_actions(obs, edge_scores, rates)
            obs, _, term, trunc, info = env.step(actions)
        else:
            actions = full_candidate_actions(obs)
            obs, _, term, trunc, info = env.step(actions, edge_scores=edge_scores)
        total_served += info.get("served_keys", 0.0)
        total_gen += info.get("generated_keys", 0.0)
        n_activated += len(getattr(env, "last_activated_edges", []) or [])
        if trace and step % 40 == 0:
            print(f"[{mode} step={step}] activated_step="
                  f"{len(getattr(env,'last_activated_edges',[]) or [])} "
                  f"gen={info.get('generated_keys',0):,.0f} "
                  f"served={info.get('served_keys',0):,.0f}", flush=True)
        if term or trunc:
            break
    arrived = float(env.metrics.arrived_keys)
    sr = total_served / arrived if arrived > 0 else 0.0
    return sr, total_served, n_activated / max(1, steps)


def main():
    ap = argparse.ArgumentParser(
        description="Same V3 scores; greedy-mutual vs Blossom-directional matching.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start-seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--w", nargs=5, type=float,
                    default=[1.0, 10.0, 1.0, 0.5, 0.2],
                    help="rate importance completion keep switch")
    args = ap.parse_args()

    profile = load_validation_profile(ROOT / "configs" / "global.yaml")
    scorer = GreedyRelayDiffusionPolicyV3(
        rate_weight=args.w[0], importance_weight=args.w[1],
        completion_weight=args.w[2], keep_weight=args.w[3], switch_weight=args.w[4])
    gr = run_mode(profile, args.seed, args.start_seed, args.steps,
                  args.w, "mutual_choice", scorer, trace=True)
    bl = run_mode(profile, args.seed, args.start_seed, args.steps,
                  args.w, "max_weight_matching", scorer, trace=True)

    print(f"\n==== seed={args.seed} start_seed={args.start_seed} steps={args.steps} ====")
    print(f"Greedy-mutual      SR = {gr[0]:.4f} (served={gr[1]:,.0f}, avg_activated/slot={gr[2]:.1f})")
    print(f"Blossom-directional SR = {bl[0]:.4f} (served={bl[1]:,.0f}, avg_activated/slot={bl[2]:.1f})")
    print(f"delta = {bl[0] - gr[0]:+.4f}  ({((bl[0]-gr[0])/max(1e-9,gr[0]))*100:+.1f}%)")


if __name__ == "__main__":
    main()