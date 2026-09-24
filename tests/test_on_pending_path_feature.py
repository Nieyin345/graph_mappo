"""`include_on_pending_path`（v2：规范路瓶颈跳）这一列的造反证测试。

v1 曾表示“有存量子图上的请求路径成员资格”；2026-09-20 的 v2 设计
**刻意换了语义**：对每条 pending 请求取 `routing.shortest_path` 的规范路，只把
QKP 存量最小的一跳标为 1。这样恰好能指出“哪一跳补货才能解除吞吐下界”。

因此测试必须验证 v2 合同，而不能继续拿 v1 的“无存量边永远为 0”约束来卡它：
1. 规范路上只有一个最小存量跳 ⟹ 只标该跳 1；同路其它跳为 0。
2. 不在任何请求规范路上的边 ⟹ 0。
3. 规范路全为 0 时，仍必须指出一个瓶颈；当前确定性 tie-break 是第一跳。
4. 没有 pending 请求 ⟹ 全 0。
5. 关开关只增加末尾 1 维，已有列逐位不动。

另外仍保留“真实运行中既见 0 又见 1”的反证，防止恒 0 / 恒 1 静默失效。
"""
from __future__ import annotations

import numpy as np
import pytest

from qkd_rl.core.config import ConfigValidator
from tests.helpers import build_test_env


def _apply(env, on_pending_path: bool):
    """开/关这一列并**重算维度**。

    ★ 必须重算：维度是 `resolve_feature_dims` 按开关现算的，
    只翻开关不重算会在 build 时报 `dim mismatch`（`test-harness-must-use-real-launch-path`
    的同族坑）。
    """
    env.config["features"]["edge"]["include_on_pending_path"] = bool(on_pending_path)
    ConfigValidator().resolve_feature_dims(env.config)
    env.reset()
    return env


def _build(on_pending_path: bool):
    env = build_test_env(".")
    return _apply(env, on_pending_path)


def _col(env, edge_id: str) -> float:
    """该边新列的取值（末列 = on_pending_path）。"""
    cfg = env.config["features"]
    pdim = int(cfg["dims"]["physical_edge_dim_resolved"])
    obs = env._build_observation()
    row = obs.edge_ids.index(edge_id)
    return float(obs.edge_features[row, pdim - 1])


def _zero_positive(env) -> None:
    for eid in list(env.qkp.positive):
        env.qkp.consume_path([eid], env.qkp.get_level(eid))


def test_column_is_last_and_shifts_nothing():
    """关/开只差 1 维，且已有列逐位不变（新列是**追加**）。"""
    off = _build(False)
    on = _build(True)
    off_dim = int(off.config["features"]["dims"]["physical_edge_dim_resolved"])
    on_dim = int(on.config["features"]["dims"]["physical_edge_dim_resolved"])
    assert on_dim == off_dim + 1, (on_dim, off_dim)

    eid = off._build_observation().physical_edge_ids[0]
    obs_off = off._build_observation()
    obs_on = on._build_observation()
    r_off = obs_off.edge_ids.index(eid)
    r_on = obs_on.edge_ids.index(eid)
    np.testing.assert_array_equal(
        obs_on.edge_features[r_on, :off_dim - 1],
        obs_off.edge_features[r_off, :off_dim - 1],
    )


def test_zero_when_no_pending_requests():
    """没有在挂请求 ⟹ 全 0（不是全 1，也不是报错）。"""
    env = _build(True)
    env.requests.reset()
    for eid in env._build_observation().physical_edge_ids:
        assert _col(env, eid) == 0.0


def test_marks_only_the_minimum_stock_hop():
    """v2：规范路上只标 QKP 存量最小的一跳，而不是整条路径。"""
    env = _build(True)
    edges = env.scenario.edges
    from qkd_rl.core.types import KeyRequest

    # 找一条至少两跳的 GS→GS 规范路，才能同时验证“瓶颈=1 / 同路非瓶颈=0”。
    src = dst = None
    path = None
    gs_ids = list(env.graph_builder.gs_ids)
    for candidate_src in gs_ids:
        for candidate_dst in gs_ids:
            if candidate_src == candidate_dst:
                continue
            candidate = env.routing.shortest_path(candidate_src, candidate_dst)
            if candidate and len(candidate) >= 2:
                src, dst, path = candidate_src, candidate_dst, candidate
                break
        if path:
            break
    assert path and src is not None and dst is not None, "测试前提：应存在至少两跳的 GS 规范路"

    env.requests.reset()
    env.requests.add_arrivals([KeyRequest(
        request_id="t1", src_gs=src, dst_gs=dst, amount=1.0e6,
        arrival_t=int(env.t), deadline_t=int(env.t) + 100, priority=1.0,
    )])
    _zero_positive(env)
    for e in edges:
        env.qkp.add_keys(e.edge_id, 1.0e5, env.t)

    # 人为把最后一跳压成唯一最小值。
    bottleneck = path[-1]
    level = env.qkp.get_level(bottleneck)
    env.qkp.consume_path([bottleneck], level * 0.9)

    indicator = env.graph_builder._compute_on_pending_path(env.requests)
    for eid in path:
        value = float(indicator[env.graph_builder._edge_pos[eid]])
        assert value == float(eid == bottleneck), (eid, bottleneck, value)
    off_path = [e.edge_id for e in edges if e.edge_id not in set(path)]
    assert all(float(indicator[env.graph_builder._edge_pos[eid]]) == 0.0 for eid in off_path)


def test_all_zero_path_still_marks_one_bottleneck():
    """v2：规范路全为 0 时仍指出瓶颈；并列最小值按规范路第一跳破平。"""
    env = _build(True)
    edges = env.scenario.edges
    src, dst = edges[0].src, edges[0].dst
    from qkd_rl.core.types import KeyRequest

    env.requests.reset()
    env.requests.add_arrivals([KeyRequest(
        request_id="t2", src_gs=src, dst_gs=dst, amount=1.0e6,
        arrival_t=int(env.t), deadline_t=int(env.t) + 100, priority=1.0,
    )])
    _zero_positive(env)
    path = env.routing.shortest_path(src, dst)
    assert path, "测试前提：src/dst 之间应有规范路"

    indicator = env.graph_builder._compute_on_pending_path(env.requests)
    marked = {
        eid for eid in path
        if float(indicator[env.graph_builder._edge_pos[eid]]) == 1.0
    }
    assert marked == {path[0]}, (path, marked)
    assert all(
        float(indicator[env.graph_builder._edge_pos[e.edge_id]]) == 0.0
        for e in edges if e.edge_id not in set(path)
    )


def test_not_constant_across_a_real_run():
    """★ 造反证：真跑几步，这一列必须**既有 0 也有 1**。

    这一条是防"恒 0 / 恒 1"的：前四条的失败模式都是静默的
    （占了一维却什么都没表达），只有"取值真的在变"能排除它。
    """
    env = _build(True)
    seen = set()
    for _ in range(40):
        obs = env._build_observation()
        cfg = env.config["features"]
        pdim = int(cfg["dims"]["physical_edge_dim_resolved"])
        if obs.edge_features.size:
            seen.update(np.unique(obs.edge_features[:, pdim - 1]).tolist())
        acts = {n: (n, n) for n in obs.node_ids}          # 全 idle，只推进时间
        env.step(acts)
    assert 0.0 in seen and 1.0 in seen, (
        "列取值只有 %s ⟹ 它没有表达任何区别（恒 0/恒 1 是静默失效）" % sorted(seen))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
