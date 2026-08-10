"""Audit the training data pipeline for breaks / unreasonable inputs.

Checks the H5-backed environment observation, masks, rate windows, action
legality, reward finiteness, and the matching policy action over a few steps.

Usage:
    python scripts/audit_data_pipeline.py
"""

from __future__ import annotations

import math
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


def check(ok: bool, name: str, detail: str = "") -> None:
    print(f"[{'OK ' if ok else 'BAD'}] {name}" + (f" | {detail}" if detail else ""))


def finite_array(arr) -> bool:
    a = np.asarray(arr)
    return bool(np.isfinite(a).all())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config["rate_provider"]["provider"] = "h5"
    config["env"]["episode_start_mode"] = "fixed"
    config["env"]["episode_steps"] = int(10**9)
    config["scenario"]["time_limit"]["days"] = 2
    ConfigValidator().validate(config)

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, device="cuda" if torch.cuda.is_available() else "cpu")
    policy.model.eval()

    dims = config["features"]["dims"]
    node_dim = int(dims["node_dim_resolved"])
    edge_dim = int(dims["edge_dim_resolved"])
    horizon = int(config["features"]["edge"]["prediction_horizon"])
    print(f"resolved dims: node={node_dim} edge={edge_dim} horizon={horizon}")

    obs = env.reset(seed=1234)
    print(f"\nreset t={env.t} scenario start_t={env.scenario.start_t} end_t={env.scenario.end_t}")

    check(
        obs.node_features.shape[1] == node_dim,
        "node feature dim",
        f"obs={obs.node_features.shape} resolved={node_dim}",
    )
    check(
        obs.edge_features.shape[1] == edge_dim,
        "edge feature dim",
        f"obs={obs.edge_features.shape} resolved={edge_dim}",
    )
    check(finite_array(obs.node_features), "node features finite")
    check(finite_array(obs.edge_features), "edge features finite")
    check(
        len(obs.physical_edge_ids) == obs.edge_features.shape[0] - len(obs.demand_edge_ids),
        "physical edge count",
        f"phys={len(obs.physical_edge_ids)} rows={obs.edge_features.shape[0]} demand={len(obs.demand_edge_ids)}",
    )

    node_ids = set(obs.node_ids)
    check(len(node_ids) == len(obs.node_ids), "node ids unique")
    edge_ids = set(obs.physical_edge_ids)
    check(len(edge_ids) == len(obs.physical_edge_ids), "edge ids unique")

    # Mask / candidate consistency.
    mask_ok = True
    mask_detail = []
    for node_id in obs.node_ids:
        if len(obs.action_masks[node_id]) != len(obs.action_candidates[node_id]):
            mask_ok = False
            mask_detail.append(f"{node_id}: mask/len mismatch")
            continue
        if not all(obs.action_masks[node_id]):
            mask_ok = False
            mask_detail.append(f"{node_id}: candidate not legal")
    check(mask_ok, "action masks align with candidates", "; ".join(mask_detail[:3]))

    # Every candidate action maps to a graph edge or idle.
    cand_edge_ok = True
    for node_id in obs.node_ids:
        for action in obs.action_candidates[node_id]:
            if action == env.action_resolver.action_space.IDLE:
                continue
            edge_id = env.action_resolver.action_space.action_to_edge(node_id, action)
            if edge_id is None or edge_id not in edge_ids:
                cand_edge_ok = False
    check(cand_edge_ok, "candidate actions reference active graph edges")

    # Rate windows.
    windows = obs.state.edge_windows
    window_len_ok = True
    window_finite = True
    for edge_id in obs.physical_edge_ids[:5] or obs.physical_edge_ids:
        w = windows[edge_id]
        if len(w.rates) != horizon + 1 or len(w.available) != horizon + 1:
            window_len_ok = False
        if not all(math.isfinite(v) for v in w.rates):
            window_finite = False
    check(window_len_ok, "rate window length == horizon+1")
    check(window_finite, "sampled rate windows finite")

    # QKP state sanity.
    check(all(v >= 0.0 for v in env.qkp.levels.values()), "qkp levels non-negative")
    check(all(v > 0.0 for v in env.qkp.capacities.values()), "qkp capacities positive")

    # Step a few ticks with the deterministic matching policy and verify the
    # whole action/reward loop stays sane.
    print("\nstepping with deterministic matching policy...")
    step_ok = True
    first_arrival = None
    first_nonzero_reward = None
    max_pending = 0
    demand_seen = 0
    for i in range(args.steps):
        arrived_before = env.metrics.arrived_keys
        step = policy.act(obs, deterministic=True)
        illegal = []
        for node_id, action in step.actions.items():
            candidates = obs.action_candidates.get(node_id)
            if candidates is None or action not in candidates:
                illegal.append(f"{node_id}->{action}")
        if illegal:
            step_ok = False
            print(f"  step {i}: ILLEGAL ACTIONS {illegal}")
            break
        if not step.matched_edges:
            step_ok = False
            print(f"  step {i}: empty matching")
            break
        next_obs, reward, term, trunc, info = env.step(
            step.actions,
            step.action_scores,
            edge_scores=step.edge_scores,
        )
        if not math.isfinite(reward):
            step_ok = False
            print(f"  step {i}: non-finite reward {reward}")
            break
        if reward != 0.0 and first_nonzero_reward is None:
            first_nonzero_reward = env.t
        max_pending = max(max_pending, len(env.requests.get_pending()))
        if len(obs.demand_edge_ids) > 0:
            demand_seen += 1
        if term or trunc:
            print(f"  step {i}: terminated early (term={term} trunc={trunc})")
            break
        if first_arrival is None and env.metrics.arrived_keys > arrived_before:
            first_arrival = env.t - 1
        print(
            f"  step {i}: t={env.t} matched={len(step.matched_edges)} "
            f"reward={reward:.3f} sr_so_far={env.metrics.episode_summary()['success_rate']:.3f}"
        )
        obs = next_obs
    check(step_ok, f"{args.steps}-step deterministic rollout sanity")
    print(f"first request arrival t={first_arrival}, first nonzero reward t={first_nonzero_reward}, "
          f"max pending={max_pending}, steps with demand edges={demand_seen}")

    # Observation cache: the next obs state must match env.t.
    check(obs.state.t == env.t, "observation state t aligns with env.t", f"obs={obs.state.t} env={env.t}")

    # Feature value ranges (first obs) for a quick reasonableness scan.
    nf = np.asarray(obs.node_features)
    ef = np.asarray(obs.edge_features)
    print(f"\nfeature ranges: node [{nf.min():.4f}, {nf.max():.4f}] edge [{ef.min():.4f}, {ef.max():.4f}]")
    rates_now = np.asarray([windows[e].rates[0] for e in obs.physical_edge_ids])
    print(f"active edge raw rates: min={rates_now.min():.2f} max={rates_now.max():.2f} "
          f"n_active={len(rates_now)}")
    pending = len(env.requests.get_pending())
    print(f"pending requests at t={env.t}: {pending}")

    # Column-wise max to find which feature groups have an unreasonable scale.
    node_top = np.argsort(nf.max(axis=0))[-5:][::-1]
    edge_top = np.argsort(ef.max(axis=0))[-5:][::-1]
    print("node feature max by column (top 5):", [(int(i), float(nf[:, i].max())) for i in node_top])
    print("edge feature max by column (top 5):", [(int(i), float(ef[:, i].max())) for i in edge_top])


if __name__ == "__main__":
    main()
