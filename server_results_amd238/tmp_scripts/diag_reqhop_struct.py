"""req_hop 的结构性问题检查：

BFS 每次遍历 adj 里的全部节点，但真正需要的只是"每条活跃边的两个端点"。
活跃边 226 条 -> 涉及的端点去重后远少于全部节点。检查：
  1) adj 节点数 vs 活跃边端点数（BFS 是否在白跑无关节点）
  2) max_hop 归一化用的是 len(adj)（全图），不是真实最大 hop
  3) 两次 BFS 能否合并成一次多源 BFS（src 与 dst 分开但同图）
  4) 能否用 scipy shortest_path 一次算完（对照）
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
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    env = build(0)
    gb = env.graph_builder
    obs = env.reset(seed=0)

    stats = {"adj_nodes": [], "edge_nodes": [], "n_edges": [], "n_src": [], "n_dst": []}
    t_bfs = []
    t_scipy = []

    for i in range(30):
        state = env._build_state()
        masks = env.mask_builder.build(state, env.qkp, env.requests)
        active_edges, _ = gb._active_edges(masks, env.mask_builder.last_flat_legal)

        adj: dict[str, list[str]] = {}
        for edge in active_edges:
            adj.setdefault(edge.src, []).append(edge.dst)
            adj.setdefault(edge.dst, []).append(edge.src)
        edge_nodes = set()
        for e in active_edges:
            edge_nodes.add(e.src); edge_nodes.add(e.dst)

        pending = env.requests.get_pending()
        srcs = {r.src_gs for r in pending}
        dsts = {r.dst_gs for r in pending}
        max_hop = float(len(adj) + 1)

        def bfs(sources):
            dist = {node: max_hop for node in adj}
            q = deque()
            for s in sources:
                if s in dist:
                    dist[s] = 0.0
                    q.append(s)
            while q:
                u = q.popleft()
                du = dist[u]
                for v in adj[u]:
                    if dist[v] > du + 1.0:
                        dist[v] = du + 1.0
                        q.append(v)
            return dist

        t0 = time.perf_counter()
        d_src = bfs(srcs); d_dst = bfs(dsts)
        t1 = time.perf_counter()

        # scipy 对照：一次算所有 (src ∪ dst) 源
        nodes = list(adj)
        idx = {n: j for j, n in enumerate(nodes)}
        rows, cols = [], []
        for u, nbrs in adj.items():
            for v in nbrs:
                rows.append(idx[u]); cols.append(idx[v])
        g = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                       shape=(len(nodes), len(nodes)))
        all_src = sorted(srcs | dsts)
        all_src = [s for s in all_src if s in idx]
        t2 = time.perf_counter()
        if all_src and len(nodes):
            shortest_path(g, method="D", unweighted=True, directed=False,
                          indices=np.asarray([idx[s] for s in all_src], dtype=np.int64))
        t3 = time.perf_counter()

        if i >= 5:
            t_bfs.append((t1 - t0) * 1000)
            t_scipy.append((t3 - t2) * 1000)
        stats["adj_nodes"].append(len(adj))
        stats["edge_nodes"].append(len(edge_nodes))
        stats["n_edges"].append(len(active_edges))
        stats["n_src"].append(len(srcs))
        stats["n_dst"].append(len(dsts))

        obs, *_ = env.step({nid: (None, None) for nid in obs.node_ids}, {}, edge_scores=None)  # type: ignore[arg-type]

    m = lambda v: statistics.median(v)  # noqa: E731
    print(f"\n结构诊断（中位）")
    print(f"  adj 节点数            {m(stats['adj_nodes']):.0f}")
    print(f"  活跃边端点数(去重)      {m(stats['edge_nodes']):.0f}")
    print(f"  活跃边数              {m(stats['n_edges']):.0f}")
    print(f"  distinct src / dst    {m(stats['n_src']):.0f} / {m(stats['n_dst']):.0f}")
    print(f"\n计时（{len(t_bfs)} 次）")
    print(f"  手写 BFS 两次          {m(t_bfs):.4f} ms")
    print(f"  scipy 一次多源         {m(t_scipy):.4f} ms")
    print(f"  -> {'scipy 更快' if m(t_scipy) < m(t_bfs) else '手写 BFS 更快'}"
          f"  ({abs(m(t_bfs) / m(t_scipy) - 1) * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
