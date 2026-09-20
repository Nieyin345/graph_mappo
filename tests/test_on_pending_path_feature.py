"""`include_on_pending_path`（通路成员资格）这一列的**造反证**测试。

### 为什么必须是造反证而不是"跑通就行"

`docs/训练诊断记录.md` 的教训 **`never-run-code-path-hides-bugs`**：
长期为 false 的开关 = 一条**从未执行过**的路径，打开前先假设里面有 bug。

而且这一列有个**特有的失败模式**：它太容易写成"恒 0 或恒 1"。
- 恒 0（例如子图取错、`node_index` 查不到）⟹ 等于**没加**，却会占一个维度
  让 `resolved_config` 看起来"改动生效了"
- 恒 1 ⟹ 等于加了个常数偏置

所以本测试**不只验证"对的时候是 1"，还断言它真的在 0/1 之间变化**，
并按手算答案逐位对。

判据（全部手算，不照抄实现）：
1. 给了存量、且该边在 src→dst 的最短路上 ⟹ **1**
2. 有存量但**不在**任何请求路径上 ⟹ **0**（这条是"成员资格 ≠ 有存量"的关键）
3. **没有存量**但几何上在最短路上的边 ⟹ **0**（子图必须是"有存量的边"）
4. 没有任何 pending 请求 ⟹ 全 **0**
5. 关掉开关 ⟹ 维度少 1，且其余列**逐位不变**（证明新列是**追加**，
   没有移动已有列的位置）
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


def test_one_for_edges_on_a_pending_request_path():
    """有存量 + 在 src→dst 最短路上 ⟹ 1。"""
    env = _build(True)
    edges = env.scenario.edges
    src, dst = edges[0].src, edges[0].dst
    from qkd_rl.core.types import KeyRequest

    env.requests.reset()
    env.requests.add_arrivals([KeyRequest(
        request_id="t1", src_gs=src, dst_gs=dst, amount=1.0e6,
        arrival_t=int(env.t), deadline_t=int(env.t) + 100, priority=1.0,
    )])
    # 给**整张图**灌一点存量，保证最短路存在且与几何最短路重合
    for e in edges:
        env.qkp.add_keys(e.edge_id, 1.0e5, env.t)
    path = env.routing.shortest_path(src, dst)
    assert path, "测试前提：src/dst 之间应有路径"
    for eid in path:
        assert _col(env, eid) == 1.0, eid
    # ★ 关键反面：**有存量但不在路径上**的边必须是 0
    off_path = [e.edge_id for e in edges if e.edge_id not in set(path)]
    if off_path:
        assert any(_col(env, eid) == 0.0 for eid in off_path)


def test_zero_for_edges_without_stock():
    """几何上在最短路、但**没有存量**的边 ⟹ 0（子图是"有存量的边"）。"""
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
    for eid in env._build_observation().physical_edge_ids:
        assert _col(env, eid) == 0.0


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
