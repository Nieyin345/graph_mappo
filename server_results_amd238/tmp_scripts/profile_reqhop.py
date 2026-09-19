"""拆解 _compute_req_hop_features 内部：adj 构建 / 两次 BFS / 输出组装。

先看这 0.145ms（占 build_edge_features 13%）到底怎么分布，
再决定是否值得动。历史教训：relay 那块三次读代码的猜测都被实测推翻。

同时做一件关键的事：检查这个函数的图规模（活跃边 vs 全候选边），
因为它只用 active_edges 建 adj —— 如果边上很少，手写 BFS 其实很快，
换成 shortest_path 反而可能更慢。
"""

from __future__ import annotations

import statistics
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config


def build(day: int):
    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config = deep_merge(
        config,
        {
            "env": {"episode_steps": 60, "episode_start_mode": "fixed", "episode_start_day": day},
            "runtime": {"device": "cpu"},
        },
    )
    ConfigValidator().validate(config)
    return build_env_from_config(config)


def main() -> int:
    env = build(0)
    gb = env.graph_builder
    obs = env.reset(seed=0)
    acc: dict[str, list[float]] = {}

    def rec(k: str, dt: float) -> None:
        acc.setdefault(k, []).append(dt * 1000)

    n_edges_hist = []
    n_pending_hist = []
    n_src_hist = []
    for _ in range(40):
        state = env._build_state()
        masks = env.mask_builder.build(state, env.qkp, env.requests)
        active_edges, _ = gb._active_edges(masks, env.mask_builder.last_flat_legal)

        # a) adj 构建
        t = time.perf_counter()
        adj: dict[str, list[str]] = {}
        for edge in active_edges:
            adj.setdefault(edge.src, []).append(edge.dst)
            adj.setdefault(edge.dst, []).append(edge.src)
        rec("a. adj 构建", time.perf_counter() - t)

        pending = env.requests.get_pending()
        srcs = {req.src_gs for req in pending}
        dsts = {req.dst_gs for req in pending}
        max_hop = float(len(adj) + 1)

        def bfs(sources):
            dist = {node: max_hop for node in adj}
            queue = deque()
            for s in sources:
                if s in dist:
                    dist[s] = 0.0
                    queue.append(s)
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] > dist[u] + 1.0:
                        dist[v] = dist[u] + 1.0
                        queue.append(v)
            return dist

        t = time.perf_counter()
        d_src = bfs(srcs)
        rec("b. BFS(src)", time.perf_counter() - t)
        t = time.perf_counter()
        d_dst = bfs(dsts)
        rec("c. BFS(dst)", time.perf_counter() - t)

        t = time.perf_counter()
        out = {}
        for edge in active_edges:
            out[edge.edge_id] = (
                d_src[edge.src] / max_hop, d_src[edge.dst] / max_hop,
                d_dst[edge.src] / max_hop, d_dst[edge.dst] / max_hop,
            )
        rec("d. 输出组装", time.perf_counter() - t)

        # 对照：整个函数（产品实现）
        t = time.perf_counter()
        gb._compute_req_hop_features(active_edges, env.requests)
        rec("e. 产品实现整体", time.perf_counter() - t)

        n_edges_hist.append(len(active_edges))
        n_pending_hist.append(len(pending))
        n_src_hist.append(len(srcs))
        obs, *_ = env.step({nid: (None, None) for nid in obs.node_ids}, {}, edge_scores=None)  # type: ignore[arg-type]

    print(f"\n{'步骤':<24}{'中位 ms':>10}{'占 e 比':>10}")
    print("-" * 46)
    med = {k: statistics.median(v) for k, v in acc.items()}
    base = med.get("e. 产品实现整体", 1.0)
    for k in sorted(acc):
        print(f"{k:<24}{med[k]:>10.4f}{med[k] / base * 100:>9.1f}%")
    print("-" * 46)
    m = lambda v: statistics.median(v)  # noqa: E731
    print(f"\n图规模: 活跃边中位 {m(n_edges_hist):.0f}（adj 节点数 ≈ {m(n_edges_hist) * 2 / max(m(n_edges_hist), 1):.1f}×边）")
    print(f"        pending 请求 {m(n_pending_hist):.0f}，distinct 源 {m(n_src_hist):.0f}")
    print(f"        总候选边（参考）= {len(gb.edges)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
