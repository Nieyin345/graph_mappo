"""GA over PathScoreGreedy (PG-User) V3-style edge-scoring weights, evaluated by
REAL env replay (deterministic, fixed environment) as a black-box fitness.

Fitness = executed served/arrived over ``--steps`` (default 2160 = one day).
The 5 weights (rate, importance, completion, keep, switch) shape which links
get activated, so the GA evolves them against the env directly. Because the
env is deterministic (rng only at reset), fitness is noise-free.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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
from qkd_rl.baselines.path_greedy import PathScoreGreedy


DEFAULT_W = (1.0, 10.0, 1.0, 0.5, 0.2)  # rate, importance, completion, keep, switch
BOUNDS = [(0.0, 5.0), (0.0, 40.0), (0.0, 5.0), (0.0, 5.0), (0.0, 2.0)]


def replay_sr(env, w: tuple, steps: int) -> float:
    pol = PathScoreGreedy(weights=tuple(w))
    obs = env._obs_cache[1] if env._obs_cache else None
    total = 0.0
    for _ in range(steps):
        actions, _ = pol.act(obs)
        obs, _, term, trunc, info = env.step(actions)
        total += info.get("served_keys", 0.0)
        if term or trunc:
            break
    arrived = float(env.metrics.arrived_keys)
    return total / arrived if arrived > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(
        description="GA over PathScoreGreedy scoring weights, real-env fitness.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start-seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=2160, help="one day = 2160 slots")
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--mutate", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.35)
    args = ap.parse_args()

    profile = load_validation_profile(ROOT / "configs" / "global.yaml")
    steps, seed, start_seed = args.steps, args.seed, args.start_seed

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
    pop[0] = list(DEFAULT_W)  # warm start with the known-good weights

    best_w, best_f = list(DEFAULT_W), -1.0
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
        next_pop = [list(m) for m in elites]
        while len(next_pop) < args.pop:
            a = [float(x) for x in random.choice(ranked[:keep])[0]]
            b = [float(x) for x in random.choice(ranked[:keep])[0]]
            cross = random.randint(1, 4)
            child = a[:cross] + b[cross:]
            if random.random() < args.mutate:
                child = mutate(tuple(child))
            next_pop.append(child)
        pop = next_pop

    print(f"\nBEST PG-User real SR = {best_f:.4f} -> w={tuple(round(x, 3) for x in best_w)} "
          f"({seed=}, {start_seed=}, steps={steps})")


if __name__ == "__main__":
    main()
