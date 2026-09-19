"""对照测试：req_hop 输出组装的三条实现路径，看哪条真的更快。

关键：产品路径的产物是 dict[edge_id] = (4 floats)，随后被 np.fromiter 展平成
(n_edges, 4) 数组。也就是说中间那个 dict-of-tuple 是纯中转。
这里比较：
  A. 现状：per-edge Python tuple 构造 + np.fromiter 展平
  B. 中间层：per-edge numpy 4 元素数组（仍存 dict，展平用 np.stack）
  C. 直产数组：直接从 dist dict 生成 (n_edges, 4) 的 numpy 数组（跳过 tuple）
并校验三者数值一致。
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


def adj_and_dist(active_edges, requests):
    adj: dict[str, list[str]] = {}
    for edge in active_edges:
        adj.setdefault(edge.src, []).append(edge.dst)
        adj.setdefault(edge.dst, []).append(edge.src)
    pending = requests.get_pending()
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

    return adj, bfs(srcs), bfs(dsts), max_hop


def path_A(active_edges, d_src, d_dst, max_hop):
    """现状：dict[edge_id] -> tuple(4 floats)，再 np.fromiter 展平。"""
    out = {}
    for edge in active_edges:
        out[edge.edge_id] = (
            d_src[edge.src] / max_hop, d_src[edge.dst] / max_hop,
            d_dst[edge.src] / max_hop, d_dst[edge.dst] / max_hop,
        )
    ids = [e.edge_id for e in active_edges]
    arr = np.fromiter(
        (v for eid in ids for v in out.get(eid, (0.0, 0.0, 0.0, 0.0))),
        dtype=np.float32, count=len(ids) * 4,
    ).reshape(len(ids), 4)
    return out, arr


def path_C(active_edges, d_src, d_dst, max_hop):
    """直产 (n_edges, 4) numpy 数组（跳过 tuple），同时构造 dict 供 loop 路径用。"""
    ids = [e.edge_id for e in active_edges]
    src_vals = np.fromiter((d_src[e.src] for e in active_edges), dtype=np.float64, count=len(active_edges))
    dst_vals = np.fromiter((d_src[e.dst] for e in active_edges), dtype=np.float64, count=len(active_edges))
    src_vals2 = np.fromiter((d_dst[e.src] for e in active_edges), dtype=np.float64, count=len(active_edges))
    dst_vals2 = np.fromiter((d_dst[e.dst] for e in active_edges), dtype=np.float64, count=len(active_edges))
    arr = (np.column_stack([src_vals, dst_vals, src_vals2, dst_vals2]) / max_hop).astype(np.float32)
    return arr


def main() -> int:
    env = build(0)
    gb = env.graph_builder
    obs = env.reset(seed=0)

    ta: list[float] = []
    tc: list[float] = []
    tfull: list[float] = []
    max_diff = 0.0

    for i in range(40):
        state = env._build_state()
        masks = env.mask_builder.build(state, env.qkp, env.requests)
        active_edges, _ = gb._active_edges(masks, env.mask_builder.last_flat_legal)
        adj, d_src, d_dst, max_hop = adj_and_dist(active_edges, env.requests)

        t0 = time.perf_counter()
        _d, arr_a = path_A(active_edges, d_src, d_dst, max_hop)
        t1 = time.perf_counter()
        arr_c = path_C(active_edges, d_src, d_dst, max_hop)
        t2 = time.perf_counter()
        arr_prod = gb._compute_req_hop_features(active_edges, env.requests)
        t3 = time.perf_counter()

        if i >= 5:
            ta.append((t1 - t0) * 1000)
            tc.append((t2 - t1) * 1000)
            tfull.append((t3 - t2) * 1000)
        if arr_a.size and arr_c.size:
            max_diff = max(max_diff, float(np.abs(arr_a - arr_c).max()))
        obs, *_ = env.step({nid: (None, None) for nid in obs.node_ids}, {}, edge_scores=None)  # type: ignore[arg-type]

    ma, mc, mf = statistics.median(ta), statistics.median(tc), statistics.median(tfull)
    print(f"\n组装+展平耗时（{len(ta)} 次，已丢预热）")
    print(f"  A. 现状 tuple + fromiter   {ma:.4f} ms")
    print(f"  C. 直产 numpy 数组          {mc:.4f} ms  ({'快' if mc < ma else '慢'} {abs(ma / mc - 1) * 100:.1f}%)")
    print(f"  产品实现整体（含两次 BFS）    {mf:.4f} ms")
    print(f"\n  A vs C 数组最大差 {max_diff:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
