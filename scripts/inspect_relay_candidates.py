"""Inspect relay-candidate hop counts on one fixed time slot.

Usage:
    python scripts/inspect_relay_candidates.py [--seed 0] [--days 30]

Prints the number of currently legal physical edges and how many of them can
form 2/3/4-hop relay paths between GS pairs in the current visible graph.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--advance-steps", type=int, default=60)
    args = parser.parse_args()

    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config["rate_provider"]["provider"] = "h5"
    config["env"]["episode_start_mode"] = "fixed"
    config["scenario"]["time_limit"]["days"] = args.days
    ConfigValidator().validate(config)

    env = build_env_from_config(config)
    env.reset(seed=args.seed)
    idle = {node_id: env.action_resolver.action_space.IDLE for node_id in env.scenario.node_ids}
    for _ in range(args.advance_steps):
        env.step(idle, {})
    state = env._build_state()
    masks = env.mask_builder.build(state, env.qkp, env.requests)
    obs = env.graph_builder.build(state, env.requests, env.request_history, masks)

    edge_by_id = {edge.edge_id: edge for edge in env.graph_builder.edges}
    active_edges = [edge_by_id[eid] for eid in obs.physical_edge_ids]
    adj: dict[str, list[str]] = {node_id: [] for node_id in env.graph_builder._node_ids_list}
    for edge in active_edges:
        adj[edge.src].append(edge.dst)
        adj[edge.dst].append(edge.src)

    gs_dist: dict[str, dict[str, int]] = {}
    for gs in env.graph_builder.gs_ids:
        dist = {gs: 0}
        queue = deque([gs])
        while queue:
            node = queue.popleft()
            for nxt in adj.get(node, ()):
                if nxt in dist:
                    continue
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
        gs_dist[gs] = dist

    max_links = int(
        config["features"]["edge"].get("relay_importance", {}).get("max_path_links", 4)
    )
    inf = 10**6
    pending_pairs = set(env.requests.demand_by_pair())
    all_pairs = [
        (src, dst)
        for i, src in enumerate(env.graph_builder.gs_ids)
        for dst in env.graph_builder.gs_ids[i + 1 :]
    ]
    target_pairs = sorted(pending_pairs)[:1] if pending_pairs else [all_pairs[0]]

    edges_by_hops: dict[int, set[str]] = defaultdict(set)
    for src, dst in target_pairs:
        src_dist = gs_dist[src]
        dst_dist = gs_dist[dst]
        for edge in active_edges:
            a = min(src_dist.get(edge.src, inf), src_dist.get(edge.dst, inf))
            b = min(dst_dist.get(edge.src, inf), dst_dist.get(edge.dst, inf))
            if a >= inf or b >= inf:
                continue
            total = a + b + 1
            if total <= max_links:
                edges_by_hops[total].add(edge.edge_id)

    print("t:", env.t)
    print("active_edges:", len(active_edges))
    print("target_pair:", target_pairs[0])
    for hops in range(2, max_links + 1):
        print(
            f"hop{hops}: shortest_path_edges={len(edges_by_hops[hops])} "
            "(edges whose best path through them is exactly this many hops)"
        )
        type_counts = Counter(
            edge_by_id[edge_id].link_type.value for edge_id in edges_by_hops[hops]
        )
        print(f"  link_types={dict(sorted(type_counts.items()))}")


if __name__ == "__main__":
    main()
