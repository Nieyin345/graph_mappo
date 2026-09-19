"""拆解 relay_importance 的 0.70ms 到底花在哪。

关键怀疑：_distances_to_sources 里每次调用都
  - np.concatenate 两个数组 + 建 csr_matrix
  - scipy shortest_path(method="D", indices=...)
其中 CSR 的"图结构"只取决于边集，而边集每步变化不大；
sources 每步都变。要确认哪个是可省的。
"""

from __future__ import annotations

import statistics
import sys
import time
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
            "env": {"episode_steps": 40, "episode_start_mode": "fixed", "episode_start_day": day},
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

    acc: dict[str, list[float]] = {}

    def rec(k: str, dt: float) -> None:
        acc.setdefault(k, []).append(dt * 1000)

    N = 30
    for _ in range(N):
        state = env._build_state()
        masks = env.mask_builder.build(state, env.qkp, env.requests)
        active_edges, active_pos = gb._active_edges(masks, env.mask_builder.last_flat_legal)
        n_nodes = len(gb._node_ids_list)

        # 复刻 _distances_to_sources 的输入构造
        t = time.perf_counter()
        act_src = np.asarray(gb._inc_src_pos[active_pos], dtype=np.int64)
        act_dst = np.asarray(gb._inc_dst_pos[active_pos], dtype=np.int64)
        stocked = env.qkp.positive
        sp = np.fromiter((gb._edge_pos[e] for e in stocked), dtype=np.int64, count=len(stocked))
        extra_src = np.asarray(gb._inc_src_pos[sp], dtype=np.int64)
        extra_dst = np.asarray(gb._inc_dst_pos[sp], dtype=np.int64)
        all_src = np.concatenate([act_src, extra_src]) if extra_src.size else act_src
        all_dst = np.concatenate([act_dst, extra_dst]) if extra_dst.size else act_dst
        valid = (all_src >= 0) & (all_dst >= 0)
        src_pos = all_src[valid]; dst_pos = all_dst[valid]
        rec("a. 输入构造(concat+mask)", time.perf_counter() - t)

        # CSR 构造
        t = time.perf_counter()
        rows = np.concatenate([src_pos, dst_pos])
        cols = np.concatenate([dst_pos, src_pos])
        data = np.ones(rows.size, dtype=np.int8)
        graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        rec("b. CSR 构造", time.perf_counter() - t)

        # shortest_path: sources 数量扫一遍
        gs_needed = []
        for req in env.requests.get_pending():
            for gs in (req.src_gs, req.dst_gs):
                if gs not in gs_needed:
                    gs_needed.append(gs)
        for k in (5, 10, 20, 34):
            srcs = np.arange(min(k, n_nodes), dtype=np.int64)
            t = time.perf_counter()
            shortest_path(graph, method="D", unweighted=True, directed=False, indices=srcs)
            rec(f"c. shortest_path({k}源)", time.perf_counter() - t)

        obs, *_ = env.step({nid: (None, None) for nid in obs.node_ids}, {}, edge_scores=None)  # type: ignore[arg-type]

    print(f"\n{'步骤':<30}{'中位 ms':>10}")
    print("-" * 42)
    for k in sorted(acc):
        print(f"{k:<30}{statistics.median(acc[k]):>10.4f}")
    print("-" * 42)
    print(f"实际 gs_needed 数量 = {len(gs_needed)}（每步读一次）")
    print(f"活跃边 = {len(active_edges)}, 节点 = {n_nodes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
