"""Compare PG-Dense / PG-NoDense / V3 on the current (rebuilt 1978-edge) dataset."""
from __future__ import annotations

import argparse
import os
import sys
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
from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.baselines.path_greedy import PathGreedyPolicy, PathScoreGreedy
from qkd_rl.baselines.serve_probe import ServeProbe
from qkd_rl.baselines.greedy_relay_diffusion import GreedyRelayDiffusionPolicyV3
from qkd_rl.baselines._matching import greedy_matching_actions


def run_pg(profile, seed, start_seed, steps, dense_fill, persist_kept=True,
           v3_mix=0.0, w=(1.0, 10.0, 1.0, 0.5, 0.2), fill_mode="rate",
           rate_weight=1.0, v3_expert=False, max_hops=None):
    config = build_validation_env_config(profile, include_baselines=False,
                                         episode_steps=steps, start_mode=profile["start_mode"])
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    env.reset(seed=seed, start_seed=start_seed)
    obs = env._obs_cache[1] if env._obs_cache else None
    pol = PathGreedyPolicy(rate_weight=rate_weight, dense_fill=dense_fill,
                           persist_kept=persist_kept, v3_mix=v3_mix,
                           fill_mode=fill_mode, v3_expert=v3_expert,
                           v3_weights=w, max_hops=max_hops)
    scorer = (GreedyRelayDiffusionPolicyV3(
        rate_weight=w[0], importance_weight=w[1], completion_weight=w[2],
        keep_weight=w[3], switch_weight=w[4]) if v3_mix > 0.0 else None)
    total_served = 0.0
    n_act = 0
    for _ in range(steps):
        v3s = scorer.score_edges(obs) if scorer is not None else None
        actions, _ = pol.act(obs, v3_scores=v3s)
        obs, _, term, trunc, info = env.step(actions)
        total_served += info.get("served_keys", 0.0)
        n_act += len(getattr(env, "last_activated_edges", []) or [])
        if term or trunc:
            break
    arrived = float(env.metrics.arrived_keys)
    sr = total_served / arrived if arrived > 0 else 0.0
    sm = env.metrics.episode_summary()
    return sr, total_served, n_act / max(1, steps), sm.get("completed_requests", 0)


def run_user(profile, seed, start_seed, steps, w, phased=False, principles=False):
    config = build_validation_env_config(profile, include_baselines=False,
                                         episode_steps=steps, start_mode=profile["start_mode"])
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    env.reset(seed=seed, start_seed=start_seed)
    obs = env._obs_cache[1] if env._obs_cache else None
    pol = PathScoreGreedy(weights=tuple(w), phased=phased,
                          principles=principles,
                          router=ServeProbe(env) if (phased or principles) else None)
    total_served = 0.0
    n_act = 0
    for _ in range(steps):
        actions, _ = pol.act(obs)
        obs, _, term, trunc, info = env.step(actions)
        total_served += info.get("served_keys", 0.0)
        n_act += len(getattr(env, "last_activated_edges", []) or [])
        if term or trunc:
            break
    arrived = float(env.metrics.arrived_keys)
    sr = total_served / arrived if arrived > 0 else 0.0
    sm = env.metrics.episode_summary()
    return sr, total_served, n_act / max(1, steps), sm.get("completed_requests", 0)


def run_v3(profile, seed, start_seed, steps, w):
    config = build_validation_env_config(profile, include_baselines=False,
                                         episode_steps=steps, start_mode=profile["start_mode"])
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    env.reset(seed=seed, start_seed=start_seed)
    obs = env._obs_cache[1] if env._obs_cache else None
    scorer = GreedyRelayDiffusionPolicyV3(
        rate_weight=w[0], importance_weight=w[1], completion_weight=w[2],
        keep_weight=w[3], switch_weight=w[4])
    total_served = 0.0
    n_act = 0
    for _ in range(steps):
        edge_scores = scorer.score_edges(obs)
        rates = {e: float(obs.state.edge_windows[e].rates[0]) for e in obs.generation_edge_ids}
        actions, _ = greedy_matching_actions(obs, edge_scores, tie_rates=rates)
        dual = {n: (NodeActionSpace.IDLE, NodeActionSpace.IDLE) for n in obs.node_ids}
        for u, t in actions.items():
            if t == NodeActionSpace.IDLE or t == u:
                continue
            cur = list(dual[u]); cur[0] = t; dual[u] = tuple(cur)
            cur = list(dual[t]); cur[1] = u; dual[t] = tuple(cur)
        obs, _, term, trunc, info = env.step(dual)
        total_served += info.get("served_keys", 0.0)
        n_act += len(getattr(env, "last_activated_edges", []) or [])
        if term or trunc:
            break
    arrived = float(env.metrics.arrived_keys)
    sr = total_served / arrived if arrived > 0 else 0.0
    sm = env.metrics.episode_summary()
    return sr, total_served, n_act / max(1, steps), sm.get("completed_requests", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 7, 8, 11])
    ap.add_argument("--w", nargs=5, type=float, default=[1.0, 10.0, 1.0, 0.5, 0.2])
    args = ap.parse_args()
    profile = load_validation_profile(ROOT / "configs" / "global.yaml")

    # PG 变体：PG-Dense vs PG-User（V3 打分+最短路径整体激活+V3 抢占填充）
    variants = [
        ("PG-Dense",     dict(dense_fill=True, persist_kept=True, rate_weight=1.0)),
    ]
    results: dict[str, list[float]] = {name: [] for name, _ in variants}
    v3_srs: list[float] = []
    user_srs: list[float] = []
    skip_srs: list[float] = []
    prin_srs: list[float] = []

    for seed in args.seeds:
        for name, kw in variants:
            kw = dict(kw)
            kw.setdefault("w", tuple(args.w))
            sr, served, n_act, done = run_pg(profile, seed, seed, args.steps, **kw)
            results[name].append(sr)
            print(f"seed={seed} {name:<12} SR={sr:.4f}", flush=True)
        user_srs.append(run_user(profile, seed, seed, args.steps, args.w)[0])
        print(f"seed={seed} {'PG-User':<12} SR={user_srs[-1]:.4f}", flush=True)
        skip_srs.append(run_user(profile, seed, seed, args.steps, args.w,
                                 phased=True)[0])
        print(f"seed={seed} {'PG-User+Phased':<16} SR={skip_srs[-1]:.4f}", flush=True)
        prin_srs.append(run_user(profile, seed, seed, args.steps, args.w,
                                 principles=True)[0])
        print(f"seed={seed} {'PG-User+Prin':<16} SR={prin_srs[-1]:.4f}", flush=True)
        v3_srs.append(run_v3(profile, seed, seed, args.steps, args.w)[0])
        print(f"seed={seed} {'V3':<12} SR={v3_srs[-1]:.4f}\n", flush=True)

    print("==== summary (240 steps) ====")
    print(f"{'variant':<14}" + "".join(f"{s:>9}" for s in args.seeds) + f"{'avg':>9}")
    for name, _ in variants:
        row = results[name]
        print(f"{name:<14}" + "".join(f"{v:>9.4f}" for v in row) + f"{sum(row)/len(row):>9.4f}")
    print(f"{'PG-User':<16}" + "".join(f"{v:>9.4f}" for v in user_srs) + f"{sum(user_srs)/len(user_srs):>9.4f}")
    print(f"{'PG-User+Phased':<16}" + "".join(f"{v:>9.4f}" for v in skip_srs) + f"{sum(skip_srs)/len(skip_srs):>9.4f}")
    print(f"{'PG-User+Prin':<16}" + "".join(f"{v:>9.4f}" for v in prin_srs) + f"{sum(prin_srs)/len(prin_srs):>9.4f}")
    print(f"{'V3':<14}" + "".join(f"{v:>9.4f}" for v in v3_srs) + f"{sum(v3_srs)/len(v3_srs):>9.4f}")


if __name__ == "__main__":
    main()