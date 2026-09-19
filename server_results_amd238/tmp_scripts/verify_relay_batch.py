"""验证：把 relay_importance 的 pair 累加循环批量化后，结果是否逐位/近位一致。

原实现：对每个 pair 逐次算 a/b/total_hops/mask/decay，再累加进 totals。
批量版：把所有 pair 的 row_s/row_d/budget 堆成矩阵，一次算完再按 pair 求和。

只要两者在浮点容差内一致，就能安全替换（且能大幅减少 Python 循环与临时数组）。
本脚本不改产品代码，只做等价性验证。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config

INF = 10**6


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


def loop_version(pair_demand, gs_row, dist_matrix, act_src, act_dst, scarcity,
                 valid_scarcity, src_ok, dst_ok, max_path_links, hop_decay_factor):
    n_active = act_src.size
    totals = np.zeros(n_active, dtype=np.float64)
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
            (a < INF) & (b < INF) & (total_hops <= max_path_links)
            & src_ok & dst_ok & valid_scarcity
        )
        if not np.any(mask):
            continue
        decay = np.power(hop_decay_factor, np.maximum(0.0, total_hops - 2.0))
        totals += np.where(mask, budget * decay * scarcity, 0.0)
    return totals


def batch_version(pair_demand, gs_row, dist_matrix, act_src, act_dst, scarcity,
                  valid_scarcity, src_ok, dst_ok, max_path_links, hop_decay_factor):
    """把 pair 循环变成矩阵运算：形状 (n_pairs, n_active)。"""
    rows_s, rows_d, budgets = [], [], []
    for pair, budget in pair_demand.items():
        rs, rd = gs_row.get(pair[0]), gs_row.get(pair[1])
        if rs is None or rd is None or budget <= 0.0:
            continue
        rows_s.append(rs); rows_d.append(rd); budgets.append(budget)
    if not rows_s:
        return np.zeros(act_src.size, dtype=np.float64)

    rs = np.asarray(rows_s, dtype=np.int64)      # (P,)
    rd = np.asarray(rows_d, dtype=np.int64)
    bud = np.asarray(budgets, dtype=np.float64)[:, None]  # (P,1)

    d_s = dist_matrix[rs]                                # (P, n_nodes)
    d_d = dist_matrix[rd]
    a = np.minimum(d_s[:, act_src], d_s[:, act_dst])      # (P, n_active)
    b = np.minimum(d_d[:, act_src], d_d[:, act_dst])
    total_hops = a + b + 1.0
    mask = (
        (a < INF) & (b < INF) & (total_hops <= max_path_links)
        & src_ok & dst_ok & valid_scarcity
    )
    decay = np.power(hop_decay_factor, np.maximum(0.0, total_hops - 2.0))
    contrib = np.where(mask, bud * decay * scarcity, 0.0)
    return contrib.sum(axis=0)


def main() -> int:
    env = build(0)
    gb = env.graph_builder
    obs = env.reset(seed=0)

    max_abs = 0.0
    checked = 0
    for _ in range(20):
        state = env._build_state()
        masks = env.mask_builder.build(state, env.qkp, env.requests)
        active_edges, active_pos = gb._active_edges(masks, env.mask_builder.last_flat_legal)

        act_src = np.asarray(gb._inc_src_pos[active_pos], dtype=np.int64)
        act_dst = np.asarray(gb._inc_dst_pos[active_pos], dtype=np.int64)
        node_index = {n: i for i, n in enumerate(gb._node_ids_list)}
        n_nodes = len(gb._node_ids_list)

        # pair_demand（复刻 relay 内部）
        pair_demand: dict[tuple[str, str], float] = {}
        for req in env.requests.get_pending():
            rem = max(0.0, req.amount - req.served_amount)
            if rem <= 1e-9:
                continue
            age = max(0, int(state.t) - int(req.arrival_t))
            dl = max(1, int(req.deadline_t) - int(req.arrival_t))
            urg = np.exp(age / max(1.0, dl * 0.8))
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
        # dist_matrix 的行序 == gs_needed/gs_pos 的顺序（_distances_to_sources 按 indices 顺序返回）
        gs_row = {g: i for i, g in enumerate(gs_needed)}

        # 距离矩阵
        from qkd_rl.env.relay_importance import _distances_to_sources
        valid0 = (act_src >= 0) & (act_dst >= 0)
        dist_matrix = _distances_to_sources(n_nodes, act_src[valid0], act_dst[valid0], gs_pos, gb._node_ids_list)

        caps = np.fromiter((float(env.qkp.capacities.get(e, 0.0)) for e in [x.edge_id for x in active_edges]),
                           dtype=np.float64, count=len(active_edges))
        lvls = np.fromiter((float(env.qkp.snapshot().get(e, 0.0)) for e in [x.edge_id for x in active_edges]),
                           dtype=np.float64, count=len(active_edges))
        scarcity = np.zeros_like(caps)
        np.divide(caps - lvls, caps, out=scarcity, where=caps > 0.0)
        scarcity = np.maximum(0.0, scarcity)
        valid_scarcity = (caps > 0.0) & (scarcity > 0.0)
        src_ok = act_src >= 0
        dst_ok = act_dst >= 0

        a1 = loop_version(pair_demand, gs_row, dist_matrix, act_src, act_dst, scarcity,
                          valid_scarcity, src_ok, dst_ok, 8, 0.8)
        a2 = batch_version(pair_demand, gs_row, dist_matrix, act_src, act_dst, scarcity,
                           valid_scarcity, src_ok, dst_ok, 8, 0.8)
        d = float(np.abs(a1 - a2).max())
        max_abs = max(max_abs, d)
        checked += 1
        obs, *_ = env.step({nid: (None, None) for nid in obs.node_ids}, {}, edge_scores=None)  # type: ignore[arg-type]

    print(f"对比了 {checked} 步的累加结果")
    print(f"  循环版 vs 批量版 最大绝对差 = {max_abs:.3e}")
    print(f"  {'✅ 位级/近位一致，可安全替换' if max_abs < 1e-9 else '❌ 有差异，需重新设计'}")
    return 0 if max_abs < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
