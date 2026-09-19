"""拆解 _build_physical_edge_rows_vectorized 内部各步耗时。

重点怀疑：
  a) rn.mean(axis=0) / rn.max(axis=0)  —— 对全年 52 万行做归约，只取 ~150 列
  b) np.isin(active_np, ...) + list(last_set) —— 每步建 set/array
  c) rn[1:horizon+1, link_idx].T 花式索引
  d) np.column_stack(cols) 拼接
  e) np.concatenate 最终拼装
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
        flat = env.mask_builder.last_flat_legal
        active_edges, active_pos = gb._active_edges(masks, flat)
        ewindows = state.edge_windows
        blocks = ewindows.blocks
        rn, av, _ = blocks
        active_ids = [e.edge_id for e in active_edges]
        if gb._edge_list_link_ids is None:
            gb._edge_list_link_ids = ewindows.link_ids(gb._edge_list)
        link_idx = gb._edge_list_link_ids[active_pos]

        t = time.perf_counter()
        _ = rn.mean(axis=0)
        _ = rn.max(axis=0)
        rec("a. rn.mean+max(全年归约)", time.perf_counter() - t)

        t = time.perf_counter()
        active_np = np.asarray(active_ids)
        last_set = set(state.last_activated_edges)
        _ = np.isin(active_np, np.asarray(list(last_set)))
        rec("b. np.isin(+set/list)", time.perf_counter() - t)

        t = time.perf_counter()
        _ = rn[1:7, link_idx].T
        rec("c. 窗口花式索引(6跳)", time.perf_counter() - t)

        t = time.perf_counter()
        _ = av[1:7, link_idx].T
        rec("c2. avail窗口索引", time.perf_counter() - t)

        t = time.perf_counter()
        cols = [rn[0][link_idx], rn.mean(axis=0)[link_idx], rn.max(axis=0)[link_idx]]
        _ = np.column_stack(cols)
        rec("d. column_stack(3列)", time.perf_counter() - t)

        t = time.perf_counter()
        cap = gb._edge_capacity_arr[active_pos]
        lvl = np.fromiter(
            (gb.qkp.levels.get(e.edge_id, 0.0) for e in active_edges),
            dtype=np.float64, count=len(active_edges),
        )
        _ = np.divide(cap - lvl, cap, out=np.zeros_like(cap), where=cap > 0)
        rec("e. qkp_capacity_left(fromiter)", time.perf_counter() - t)

        obs, *_ = env.step({nid: (None, None) for nid in obs.node_ids}, {}, edge_scores=None)  # type: ignore[arg-type]

    print(f"\n{'内部步骤':<30}{'中位 ms':>10}{'占比':>8}")
    print("-" * 50)
    med = {k: statistics.median(v) for k, v in acc.items()}
    tot = sum(med.values())
    for k in sorted(acc):
        print(f"{k:<30}{med[k]:>10.4f}{med[k] / tot * 100:>7.1f}%")
    print("-" * 50)
    print(f"{'合计':<30}{tot:>10.4f}{100.0:>7.1f}%")
    print(f"\n参考: 全年行数 = {rn.shape[0]}, 活跃边数 = {len(active_ids)}")
    print(f"      mean/max 归约 = {(rn.shape[0] / max(len(active_ids), 1)):.0f} 倍于实际需要的列数")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
