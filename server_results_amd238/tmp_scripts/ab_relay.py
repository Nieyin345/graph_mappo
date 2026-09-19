"""A/B 基准：同一进程内交替测量"批量版"与"循环版"relay_importance，消除机器噪声。

思路：不改产品代码，直接 import 两个实现，在同一个 env 状态上交替调用，
各测 N 次取中位。这样两边共享完全相同的机器状态与数据，差异只来自实现。

用法：python .tmp/ab_relay.py
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
from qkd_rl.env import relay_importance as RI


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


def loop_acc(pair_demand, gs_row, dist_matrix, act_src, act_dst, scarcity,
             valid_scarcity, max_path_links, hop_decay_factor):
    """优化前的逐 pair 循环（复刻历史实现）。"""
    n_active = act_src.size
    totals = np.zeros(n_active, dtype=np.float64)
    src_ok = act_src >= 0
    dst_ok = act_dst >= 0
    for pair, budget in pair_demand.items():
        row_s = gs_row.get(pair[0])
        row_d = gs_row.get(pair[1])
        if row_s is None or row_d is None or budget <= 0.0:
            continue
        d_s = dist_matrix[row_s]
        d_d = dist_matrix[row_d]
        a = np.minimum(d_s[act_src], d_s[act_dst])
        b = np.minimum(d_d[act_src], d_d[act_dst])
        total_hops = a + b + 1.0
        mask = (
            (a < RI._INF) & (b < RI._INF) & (total_hops <= max_path_links)
            & src_ok & dst_ok & valid_scarcity
        )
        if not np.any(mask):
            continue
        decay = np.power(hop_decay_factor, np.maximum(0.0, total_hops - 2.0))
        totals += np.where(mask, budget * decay * scarcity, 0.0)
    return totals


def batch_acc(pair_demand, gs_row, dist_matrix, act_src, act_dst, scarcity,
              valid_scarcity, max_path_links, hop_decay_factor):
    """优化后的批量实现（与产品代码同逻辑）。"""
    rows_s, rows_d, budgets = [], [], []
    for pair, budget in pair_demand.items():
        rs, rd = gs_row.get(pair[0]), gs_row.get(pair[1])
        if rs is None or rd is None or budget <= 0.0:
            continue
        rows_s.append(rs); rows_d.append(rd); budgets.append(budget)
    if not rows_s:
        return np.zeros(act_src.size, dtype=np.float64)
    src_ok = act_src >= 0
    dst_ok = act_dst >= 0
    d_s = dist_matrix[np.asarray(rows_s, dtype=np.int64)]
    d_d = dist_matrix[np.asarray(rows_d, dtype=np.int64)]
    a = np.minimum(d_s[:, act_src], d_s[:, act_dst])
    b = np.minimum(d_d[:, act_src], d_d[:, act_dst])
    total_hops = a + b + 1.0
    m = ((a < RI._INF) & (b < RI._INF) & (total_hops <= max_path_links)
         & src_ok & dst_ok & valid_scarcity)
    if not np.any(m):
        return np.zeros(act_src.size, dtype=np.float64)
    decay = np.power(hop_decay_factor, np.maximum(0.0, total_hops - 2.0))
    bud = np.asarray(budgets, dtype=np.float64)[:, None]
    return np.where(m, bud * decay * scarcity, 0.0).sum(axis=0)


def main() -> int:
    env = build(0)
    gb = env.graph_builder
    obs = env.reset(seed=0)

    loop_t, batch_t = [], []
    max_diff = 0.0
    N = 40

    for i in range(N):
        state = env._build_state()
        masks = env.mask_builder.build(state, env.qkp, env.requests)
        active_edges, active_pos = gb._active_edges(masks, env.mask_builder.last_flat_legal)
        act_src = np.asarray(gb._inc_src_pos[active_pos], dtype=np.int64)
        act_dst = np.asarray(gb._inc_dst_pos[active_pos], dtype=np.int64)
        node_index = {n: j for j, n in enumerate(gb._node_ids_list)}
        n_nodes = len(gb._node_ids_list)

        pair_demand = {}
        for req in env.requests.get_pending():
            rem = max(0.0, req.amount - req.served_amount)
            if rem <= 1e-9:
                continue
            age = max(0, int(state.t) - int(req.arrival_t))
            dl = max(1, int(req.deadline_t) - int(req.arrival_t))
            urg = float(np.exp(age / max(1.0, dl * 0.8)))
            pair = tuple(sorted((req.src_gs, req.dst_gs)))
            pair_demand[pair] = pair_demand.get(pair, 0.0) + rem * urg
        if not pair_demand:
            obs, *_ = env.step({nid: (None, None) for nid in obs.node_ids}, {}, edge_scores=None)  # type: ignore[arg-type]
            continue

        gs_needed = []
        for pair in pair_demand:
            for g in pair:
                if g not in gs_needed and g in node_index:
                    gs_needed.append(g)
        gs_pos = [node_index[g] for g in gs_needed]
        gs_row = {g: j for j, g in enumerate(gs_needed)}

        valid0 = (act_src >= 0) & (act_dst >= 0)
        dist_matrix = RI._distances_to_sources(
            n_nodes, act_src[valid0], act_dst[valid0], gs_pos, gb._node_ids_list
        )

        ids = [e.edge_id for e in active_edges]
        caps = np.fromiter((float(env.qkp.capacities.get(e, 0.0)) for e in ids),
                           dtype=np.float64, count=len(ids))
        lvls = np.fromiter((float(env.qkp.levels.get(e, 0.0)) for e in ids),
                           dtype=np.float64, count=len(ids))
        scarcity = np.zeros_like(caps)
        np.divide(caps - lvls, caps, out=scarcity, where=caps > 0.0)
        scarcity = np.maximum(0.0, scarcity)
        valid_scarcity = (caps > 0.0) & (scarcity > 0.0)

        # 交替测量（同状态、同数据）
        t0 = time.perf_counter()
        r1 = loop_acc(pair_demand, gs_row, dist_matrix, act_src, act_dst, scarcity,
                      valid_scarcity, 8, 0.8)
        t1 = time.perf_counter()
        r2 = batch_acc(pair_demand, gs_row, dist_matrix, act_src, act_dst, scarcity,
                       valid_scarcity, 8, 0.8)
        t2 = time.perf_counter()
        if i >= 5:  # 丢掉预热
            loop_t.append((t1 - t0) * 1000)
            batch_t.append((t2 - t1) * 1000)
        max_diff = max(max_diff, float(np.abs(r1 - r2).max()))

        obs, *_ = env.step({nid: (None, None) for nid in obs.node_ids}, {}, edge_scores=None)  # type: ignore[arg-type]

    ml, mb = statistics.median(loop_t), statistics.median(batch_t)
    print(f"\n配对测量（同状态交替，{len(loop_t)} 对，已丢预热）")
    print(f"  循环版 中位 {ml:.4f} ms")
    print(f"  批量版 中位 {mb:.4f} ms")
    print(f"  差     {ml - mb:+.4f} ms  ({'快 ' + format((ml / mb - 1) * 100, '.1f') + '%' if mb < ml else '慢'})")
    print(f"  数值最大绝对差 {max_diff:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
