"""Diagnose why PG-Pair activates fewer edges than PG-Old on a clean slot."""
from __future__ import annotations

import os
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
from qkd_rl.baselines.path_greedy import PathGreedyPolicy
from qkd_rl.baselines.path_greedy_pair import PathGreedyPairPolicy


def make_env(profile, seed):
    cfg = build_validation_env_config(profile, include_baselines=False, episode_steps=90,
                                      start_mode=profile["start_mode"])
    ConfigValidator().validate(cfg)
    env = build_env_from_config(cfg)
    env.reset(seed=seed, start_seed=seed)
    return env


def non_idle(actions):
    return [(n, a) for n, a in actions.items() if a[0] != NodeActionSpace.IDLE or a[1] != NodeActionSpace.IDLE]


def run_steps(env, policy, steps=30):
    per_slot = []
    obs = env._obs_cache[1]
    for _ in range(steps):
        a, _ = policy.act(obs)
        obs, *_ = env.step(a)
        per_slot.append(len(env.last_activated_edges))
    return per_slot


def main():
    profile = load_validation_profile(ROOT / "configs" / "global.yaml")
    pg_old = PathGreedyPolicy()
    pg_pair = PathGreedyPairPolicy()

    for seed in (7, 3, 5, 8, 11):
        env_o = make_env(profile, seed)
        env_p = make_env(profile, seed)
        so = run_steps(env_o, pg_old)
        sp = run_steps(env_p, pg_pair)
        diff = [i for i in range(len(so)) if so[i] != sp[i]]
        print(f"\n===== seed={seed} (30 slots) PG-Old total={sum(so)} PG-Pair total={sum(sp)} "
              f"| slots where counts differ: {len(diff)}/{len(so)} =====")
        print(f"  PG-Old  per-slot activated: {so}")
        print(f"  PG-Pair per-slot activated: {sp}")
        for i in diff[:8]:
            print(f"  slot t={i+1}: O={so[i]} P={sp[i]}")


if __name__ == "__main__":
    main()