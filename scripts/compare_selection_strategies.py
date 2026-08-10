"""Compare decision strategies for a trained checkpoint without training.

Usage:
    python scripts/compare_selection_strategies.py outputs/run/checkpoint_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.algos.checkpoint import load_checkpoint
from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.core.types import NodeType
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


def _config(days: int):
    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config = deep_merge(config, load_config([ROOT / "configs" / "train_demand_edge.yaml"]))
    config["rate_provider"]["provider"] = "h5"
    config["env"]["episode_start_mode"] = "fixed"
    config["env"]["episode_steps"] = int(10**9)
    config["scenario"]["time_limit"]["days"] = days
    ConfigValidator().validate(config)
    return config


def _matching_actions(
    env,
    scores: dict[str, float],
    endpoints: dict[str, tuple[str, str]],
    relay: dict[str, float],
    strategy: str,
    device: str,
):
    idle = env.action_resolver.action_space.IDLE
    actions = {n: idle for n in env.scenario.node_ids}
    action_scores = {n: {idle: 0.0} for n in env.scenario.node_ids}
    edge_arg = dict(scores)

    if strategy == "greedy":
        tensors = {eid: torch.tensor(s, dtype=torch.float32, device=device) for eid, s in scores.items()}
        acts, ascores, _matched, _, _ = policy._sample_matching(tensors, deterministic=True)
        return acts, ascores, scores

    cand = dict(scores)
    if "filter" in strategy or strategy.startswith("threshold"):
        threshold = 1.5 if strategy.startswith("threshold") else 0.0
        cand = {
            eid: s
            for eid, s in scores.items()
            if relay.get(eid, 0.0) > 1e-9 and s > threshold
        }

    if strategy == "greedy_filter":
        tensors = {eid: torch.tensor(s, dtype=torch.float32, device=device) for eid, s in cand.items()}
        acts, ascores, _matched, _, _ = policy._sample_matching(tensors, deterministic=True)
        return acts, ascores, cand

    if strategy in ("mwm_all", "mwm_filter", "threshold_mwm"):
        g = nx.Graph()
        g.add_nodes_from(env.scenario.node_ids)
        for eid, s in cand.items():
            u, v = endpoints[eid]
            g.add_edge(u, v, weight=s, edge_id=eid)
        mw = nx.max_weight_matching(g, maxcardinality=False, weight="weight")
        for u, v in mw:
            eid = g.edges[u, v].get("edge_id")
            actions[u] = v
            actions[v] = u
            action_scores[u] = {v: cand[eid]}
            action_scores[v] = {u: cand[eid]}
        return actions, action_scores, cand

    if strategy == "beam4":
        beams = [([], set(), 0.0)]
        while True:
            new_beams = []
            for matched, used, total in beams:
                avail = [
                    (eid, s)
                    for eid, s in cand.items()
                    if endpoints[eid][0] not in used and endpoints[eid][1] not in used
                ]
                if not avail:
                    new_beams.append((matched, used, total))
                    continue
                for eid, s in avail:
                    u, v = endpoints[eid]
                    new_beams.append((matched + [eid], used | {u, v}, total + s))
            if not new_beams or len(new_beams) == len(beams):
                beams = new_beams or beams
                break
            beams = sorted(new_beams, key=lambda item: -item[2])[:4]
        best = max(beams, key=lambda item: item[2])
        for eid in best[0]:
            u, v = endpoints[eid]
            actions[u] = v
            actions[v] = u
            action_scores[u] = {v: cand[eid]}
            action_scores[v] = {u: cand[eid]}
        return actions, action_scores, cand

    if strategy == "regret_greedy":
        used = set()
        while True:
            best = None
            for eid, s in cand.items():
                u, v = endpoints[eid]
                if u in used or v in used:
                    continue
                blocked = 0.0
                for f, fs in cand.items():
                    if f == eid:
                        continue
                    fu, fv = endpoints[f]
                    if (fu in (u, v) or fv in (u, v)) and fu not in used and fv not in used:
                        blocked = max(blocked, fs)
                value = s - blocked
                if best is None or value > best[0]:
                    best = (value, eid)
            if best is None:
                break
            _, eid = best
            u, v = endpoints[eid]
            actions[u] = v
            actions[v] = u
            action_scores[u] = {v: cand[eid]}
            action_scores[v] = {u: cand[eid]}
            used.update((u, v))
        return actions, action_scores, cand

    if strategy == "gs_hap_sat":
        node_type = {n.node_id: n.node_type for n in env.scenario.nodes}
        used = set()
        for typ in (NodeType.GS, NodeType.HAP, NodeType.SAT):
            nodes = sorted(n for n in env.scenario.node_ids if node_type[n] == typ and n not in used)
            for node in nodes:
                best = None
                for eid, s in cand.items():
                    u, v = endpoints[eid]
                    if node not in (u, v):
                        continue
                    other = v if u == node else u
                    if other in used or node in used:
                        continue
                    if best is None or s > best[1]:
                        best = (eid, s)
                if best is not None:
                    eid, s = best
                    u, v = endpoints[eid]
                    actions[u] = v
                    actions[v] = u
                    action_scores[u] = {v: s}
                    action_scores[v] = {u: s}
                    used.update((u, v))
        return actions, action_scores, cand

    raise ValueError(f"Unknown strategy: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    global policy
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = _config(args.days)
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, device)
    data = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(data.model_state)
    model.eval()
    endpoints = {e.edge_id: (e.src, e.dst) for e in env.scenario.edges}
    phys_dim = int(config["features"]["dims"]["physical_edge_dim_resolved"])
    relay_col = phys_dim - 2

    all_strategies = [
        "greedy",
        "mwm_all",
        "mwm_filter",
        "greedy_filter",
        "beam4",
        "regret_greedy",
        "threshold_mwm",
        "gs_hap_sat",
    ]
    strategies = args.strategies or all_strategies
    results = {}

    for strategy in strategies:
        env = build_env_from_config(_config(args.days))
        obs = env.reset(seed=args.seed)
        done = False
        total_reward = 0.0
        while not done:
            output = model(obs, device, build_logits_dict=False)
            scores_t = output.edge_scores or {}
            scores = {eid: float(s.detach().cpu()) for eid, s in scores_t.items()}
            relay = {}
            for eid in obs.physical_edge_ids:
                row = obs.edge_ids.index(eid)
                relay[eid] = float(obs.edge_features[row, relay_col])
            actions, action_scores, edge_arg = _matching_actions(
                env, scores, endpoints, relay, strategy, device
            )
            obs, reward, terminated, truncated, _info = env.step(
                actions, action_scores, edge_scores=edge_arg
            )
            total_reward += reward
            done = terminated or truncated
        s = env.metrics.episode_summary()
        entry = {
            "strategy": strategy,
            "success_rate": round(float(s["success_rate"]), 6),
            "request_completion_rate": round(float(s["request_completion_rate"]), 6),
            "total_reward": round(float(total_reward), 3),
            "conflict_count": int(s["conflict_count"]),
        }
        results[strategy] = entry
        print(json.dumps(entry, ensure_ascii=False))

    if args.out:
        out_path = Path(ROOT) / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"written: {out_path}")


if __name__ == "__main__":
    main()
