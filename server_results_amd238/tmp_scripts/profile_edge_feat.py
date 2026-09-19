"""拆解 build_edge_features 的真实构成：relay / req_hop / physical rows / demand rows。

上一轮 profile 显示 build_edge_features 占总 graph build 的 75.9%（1.10ms），
但 _build_physical_edge_rows_vectorized 内部各步加起来只有 0.109ms。
差额必然在 relay_importance / _compute_req_hop_features / demand rows 里。
本脚本直接对这几块计时，定位真正的大头。
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
from qkd_rl.env.relay_importance import compute_relay_importance


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

    relay_cfg = env.config["features"]["edge"].get("relay_importance", {})
    N = 30
    for _ in range(N):
        state = env._build_state()
        masks = env.mask_builder.build(state, env.qkp, env.requests)
        flat = env.mask_builder.last_flat_legal
        active_edges, active_pos = gb._active_edges(masks, flat)

        # 1) relay_importance（最大嫌疑）
        t = time.perf_counter()
        stocked = env.qkp.positive
        stocked_pos = np.fromiter((gb._edge_pos[e] for e in stocked), dtype=np.int64, count=len(stocked))
        ri = compute_relay_importance(
            node_ids=gb._node_ids_list,
            physical_edge_ids=[e.edge_id for e in active_edges],
            pending_requests=env.requests.get_pending(),
            qkp_snapshot=env.qkp.snapshot(),
            qkp_capacity=env.qkp.capacities,
            t=state.t,
            max_path_links=int(relay_cfg.get("max_path_links", 3)),
            hop_decay_factor=float(relay_cfg.get("hop_decay_factor", 0.25)),
            capacity_strength=float(relay_cfg.get("capacity_decay_strength", 1.0)),
            min_scarcity=float(relay_cfg.get("min_scarcity", 0.0)),
            wait_urgency_tau_ratio=float(relay_cfg.get("wait_urgency_tau_ratio", 0.8)),
            ignore_consumption=bool(relay_cfg.get("ignore_consumption", False)),
            include_stocked_unavailable=True,
            link_type_bonus=relay_cfg.get("link_type_bonus", None),
            active_src_pos=gb._inc_src_pos[active_pos],
            active_dst_pos=gb._inc_dst_pos[active_pos],
            stocked_src_pos=gb._inc_src_pos[stocked_pos],
            stocked_dst_pos=gb._inc_dst_pos[stocked_pos],
        )
        rec("1. relay_importance", time.perf_counter() - t)

        # 2) req_hop（第二个嫌疑）
        t = time.perf_counter()
        rh = gb._compute_req_hop_features(active_edges, env.requests)
        rec("2. _compute_req_hop_features", time.perf_counter() - t)

        # 3) physical rows
        blocks = state.edge_windows.blocks
        pdim = int(env.config["features"]["dims"]["physical_edge_dim_resolved"])
        ddim = int(env.config["features"]["dims"]["demand_edge_dim_resolved"])
        t = time.perf_counter()
        _ = gb._build_physical_edge_rows_vectorized(
            active_edges, active_pos, state, blocks, pdim, ddim,
            relay_importance=ri, req_hop=rh,
        )
        rec("3. physical rows", time.perf_counter() - t)

        # 4) demand rows
        ds = env.requests.stats_by_pair(state.t, env.request_history, None, 10, 960)
        pairs = gb.build_demand_pairs(state, env.requests, env.request_history, demand_stats=ds)
        t = time.perf_counter()
        _ = gb.build_demand_edge_features(
            state, env.requests, env.request_history, pairs, demand_stats_by_pair=ds
        )
        rec("4. demand rows", time.perf_counter() - t)

        obs, *_ = env.step({nid: (None, None) for nid in obs.node_ids}, {}, edge_scores=None)  # type: ignore[arg-type]

    print(f"\n{'build_edge_features 内部':<32}{'中位 ms':>10}{'占比':>8}")
    print("-" * 52)
    med = {k: statistics.median(v) for k, v in acc.items()}
    tot = sum(med.values())
    for k in sorted(acc):
        print(f"{k:<32}{med[k]:>10.4f}{med[k] / tot * 100:>7.1f}%")
    print("-" * 52)
    print(f"{'合计':<32}{tot:>10.4f}{100.0:>7.1f}%")
    print(f"\n活跃边 {len(active_edges)}，pending 请求 {len(env.requests.get_pending())}，"
          f"有库存边 {len(env.qkp.positive)}，需求对 {len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
