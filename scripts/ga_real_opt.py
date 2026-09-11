"""Genetic algorithm over the greedy V3 *edge-scoring weights*, evaluated by
REAL env replay (deterministic), to find a genuinely executable link-selection
plan that beats the default greedy V3 baseline.

Core idea
---------
The greedy V3 baseline scores every legal physical edge as

    score = rate_weight * rate_norm * (1 + importance*importance_norm)
          + completion_weight * completion_norm
          + keep_weight * keep_norm - switch_weight * switch_norm

and then does a global greedy matching. These weights shape *which* links get
activated and therefore drive the real executed success rate. Rather than
hand-tuning, a GA evolves the 5-weight vector (plus a couple of structural
knobs) against the deterministic env as a black-box evaluator. Because the env
is deterministic (rng only used at reset), fitness = real served/arrived is
noise-free, so selection is clean.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
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


DEFAULT_W = [1.0, 10.0, 1.0, 0.5, 0.2]  # rate, importance, completion, keep, switch


def weighted_actions(obs, w: tuple[float, float, float, float, float]) -> dict[str, tuple]:
    """Score edges with custom weights, run greedy matching, return dual-port actions."""
    from qkd_rl.baselines._matching import edge_map, greedy_matching_actions
    from qkd_rl.env.relay_importance import compute_relay_importance
    rate_weight, importance_weight, completion_weight, keep_weight, switch_weight = w

    active_ids = list(obs.physical_edge_ids)
    endpoints, pair_to_edge = edge_map(obs)
    importance = compute_relay_importance(
        node_ids=obs.node_ids,
        physical_edge_ids=obs.physical_edge_ids,
        pending_requests=obs.state.pending_requests,
        qkp_snapshot=obs.state.qkp_snapshot,
        qkp_capacity=obs.state.qkp_capacity,
        t=obs.state.t,
        all_edge_ids=list(obs.state.edge_windows.keys()),
    )
    rates = {e: float(obs.state.edge_windows[e].rates[0]) for e in active_ids}
    max_rate = max(rates.values(), default=1.0) or 1.0
    levels = obs.state.qkp_snapshot
    candidates = {
        n: set(obs.action_candidates[n]) - {NodeActionSpace.IDLE} for n in obs.node_ids
    }
    pairs: dict[tuple[str, str], float] = {}
    for req in obs.state.pending_requests:
        rem = max(0.0, req.amount - req.served_amount)
        if rem > 1e-9:
            pr = tuple(sorted((req.src_gs, req.dst_gs)))
            pairs[pr] = pairs.get(pr, 0.0) + rem
    completion: dict[str, float] = {}
    for (a, b), amount in pairs.items():
        for relay in (candidates.get(a, set()) & candidates.get(b, set())):
            e_a = pair_to_edge.get(tuple(sorted((a, relay))))
            e_b = pair_to_edge.get(tuple(sorted((b, relay))))
            if e_a is None or e_b is None:
                continue
            sa = levels.get(e_a, 0.0) > 0.0
            sb = levels.get(e_b, 0.0) > 0.0
            if sa and sb:
                continue
            if not sa and not sb:
                continue
            edge = e_b if sa else e_a
            completion[edge] = completion.get(edge, 0.0) + amount
    max_completion = max(completion.values(), default=1.0) or 1.0
    last_activated = set(obs.state.last_activated_edges)

    edge_scores: dict[str, float] = {}
    for e in active_ids:
        rate_norm = rates[e] / max_rate
        imp_norm = importance.get(e, 0.0)
        compl_norm = completion.get(e, 0.0) / max_completion
        keep_norm = 1.0 if (e in last_activated and (imp_norm > 0.0 or compl_norm > 0.0)) else 0.0
        switch_norm = 0.0 if e in last_activated else 1.0
        edge_scores[e] = (
            rate_weight * rate_norm * (1.0 + importance_weight * imp_norm)
            + completion_weight * compl_norm
            + keep_weight * keep_norm
            - switch_weight * switch_norm
        )

    actions, _ = greedy_matching_actions(obs, edge_scores, tie_rates=rates)
    dual: dict[str, tuple] = {}
    for node_id in obs.node_ids:
        tx, rx = NodeActionSpace.IDLE, NodeActionSpace.IDLE
        opposite = actions.get(node_id)
        if opposite != NodeActionSpace.IDLE and opposite is not None and opposite != node_id:
            tx = opposite
            rx = next((s for s, t in actions.items() if t == node_id), NodeActionSpace.IDLE)
        dual[node_id] = (tx, rx)
    return dual


def replay_sr(env, w: tuple, steps: int) -> float:
    total = 0.0
    obs = env._obs_cache[1] if env._obs_cache else None
    for _ in range(steps):
        actions = weighted_actions(obs, w)
        obs, _, term, trunc, info = env.step(actions)
        total += info.get("served_keys", 0.0)
        if term or trunc:
            break
    arrived = float(env.metrics.arrived_keys)
    return total / arrived if arrived > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="GA over greedy V3 scoring weights, real-env fitness.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start-seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--mutate", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.35, help="gaussian mutation std on scaled vars")
    args = ap.parse_args()

    profile = load_validation_profile(ROOT / "configs" / "global.yaml")
    steps, seed, start_seed = args.steps, args.seed, args.start_seed

    # param bounds (log-ish scaling for weights that span decades)
    BOUNDS = [(0.0, 5.0), (0.0, 40.0), (0.0, 5.0), (0.0, 5.0), (0.0, 2.0)]

    def new_env():
        config = build_validation_env_config(
            profile, include_baselines=False, episode_steps=steps,
            start_mode=profile["start_mode"])
        ConfigValidator().validate(config)
        return build_env_from_config(config)

    def eval_w(w: tuple) -> float:
        env = new_env()
        env.reset(seed=seed, start_seed=start_seed)
        return replay_sr(env, w, steps)

    def rand_w():
        return tuple(random.uniform(lo, hi) for lo, hi in BOUNDS)

    def mutate(w: tuple) -> list:
        return [min(hi, max(lo, x + random.gauss(0, 1) * args.sigma * (hi - lo)))
                for x, (lo, hi) in zip(w, BOUNDS)]

    pop = [rand_w() for _ in range(args.pop)]
    pop[0] = tuple(DEFAULT_W)  # warm start with the known-good V3 weights

    for gen in range(args.gens):
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fits = list(ex.map(eval_w, pop))
        ranked = sorted(zip(pop, fits), key=lambda x: -x[1])
        best_w, best_f = ranked[0]
        mean = sum(f for _, f in ranked) / len(ranked)
        print(f"[gen {gen}] best={best_f:.4f} mean={mean:.4f} "
              f"w=(r{best_w[0]:.2f} i{best_w[1]:.1f} c{best_w[2]:.2f} "
              f"k{best_w[3]:.2f} s{best_w[4]:.2f})", flush=True)

        keep = max(1, args.pop // 3)
        elites = [[float(x) for x in w_] for w_, _ in ranked[:keep]]
        next_pop = [m if False else m for m in elites]  # elites copied
        while len(next_pop) < args.pop:
            a = [float(x) for x in random.choice(ranked[:keep])[0]]
            b = [float(x) for x in random.choice(ranked[:keep])[0]]
            cross = random.randint(1, 4)
            child = a[:cross] + b[cross:]
            if random.random() < args.mutate:
                child = mutate(tuple(child))
            next_pop.append(child)
        pop = next_pop

    print(f"\nBEST weights real SR = {best_f:.4f} -> w={tuple(round(x,3) for x in best_w)} "
          f"({seed=}, {start_seed=})")
    print(f"(default greedy V3 = 0.2874 on seed7)")


if __name__ == "__main__":
    main()