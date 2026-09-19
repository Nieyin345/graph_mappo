"""细粒度拆解 graph_builder.build 各子函数的真实耗时占比。

cProfile 会把 C 扩展（numpy/bincount/fromiter）的时间挂到调用者身上，容易误导。
这里用 perf_counter 直接对每个子步骤包一层计时，得到"墙上时间"口径的占比。
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

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
    N = 30

    def rec(key: str, dt: float) -> None:
        acc.setdefault(key, []).append(dt * 1000)

    for _ in range(N):
        state = env._build_state()
        masks = env.mask_builder.build(state, env.qkp, env.requests)
        flat = env.mask_builder.last_flat_legal

        t0 = time.perf_counter()
        demand_stats = env.requests.stats_by_pair(state.t, env.request_history, None, 10, 960)
        t1 = time.perf_counter(); rec("1.stats_by_pair", t1 - t0)

        pairs = gb.build_demand_pairs(state, env.requests, env.request_history, demand_stats=demand_stats)
        t2 = time.perf_counter(); rec("2.build_demand_pairs", t2 - t1)

        active_edges, active_pos = gb._active_edges(masks, flat)
        t3 = time.perf_counter(); rec("3._active_edges", t3 - t2)

        nf = gb.build_node_features(state, env.requests, env.request_history)
        t4 = time.perf_counter(); rec("4.build_node_features", t4 - t3)

        ei = gb.build_edge_index(active_edges, pairs)
        t5 = time.perf_counter(); rec("5.build_edge_index", t5 - t4)

        ef = gb.build_edge_features(
            active_edges, active_pos, state, env.requests, env.request_history, pairs,
            demand_stats_by_pair=demand_stats,
        )
        t6 = time.perf_counter(); rec("6.build_edge_features", t6 - t5)

        # 拆 6 的两大块：relay_importance 与 _compute_req_hop_features
        t6a = time.perf_counter()
        gb._last_relay_importance = gb._last_relay_importance  # noqa: B018
        t6b = time.perf_counter(); rec("6a.(relay 已在上一行内)", t6b - t6a)

        obs, *_ = env.step({nid: (None, None) for nid in obs.node_ids}, {}, edge_scores=None)  # type: ignore[arg-type]

    print(f"\n{'子步骤':<28}{'中位 ms':>10}{'均值 ms':>10}{'占比':>8}")
    print("-" * 58)
    totals = {k: sum(v) / len(v) for k, v in acc.items()}
    grand = sum(totals.values())
    for k in sorted(acc):
        med = statistics.median(acc[k])
        avg = totals[k]
        print(f"{k:<28}{med:>10.3f}{avg:>10.3f}{avg / grand * 100:>7.1f}%")
    print("-" * 58)
    print(f"{'合计(不含 env.step)':<28}{'':>10}{grand:>10.3f}{100.0:>7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
