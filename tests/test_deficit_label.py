"""缺额预测标签 label_mode="deficit"（改进项 2，2026-09-28）。

动机：\"new\" 标签 = 边 e 在 (t, t+K] 内新落在待服务请求路径上。它不区分 e 现在
有没有存货——已经囤够密钥的边不需要再提前生成，把它们也当正例，偏置会把激活预算
浪费在不缺货的边上。\"deficit\" = \"new\" ∧ 当前库存 < 阈值（默认 = 平均请求量 1e5 bit）
= "很快要用、但还没备好"的边。

实现约定（只有一个实现）：
  * 阈值判断只在 v8 GraphBuilder 里做（唯一读配置的地方），结果放在
    GraphObservation.physical_edge_low_stock（与 physical_edge_ids 对齐的 bool 列表）；
    只有 usage_forecast.label_mode == "deficit" 时才填，否则为 None ⟹ 既有臂零变化。
  * RolloutBuffer 的 deficit 模式 = new 窗口标签再与 step t 的 low_stock 取与；
    缺该字段时报错（不静默退化成 new）。

钉住：
  1. deficit ≤ new 逐元素，且只保留 t 时刻低库存的边（用 t 的库存，不看未来）；
  2. 缺 physical_edge_low_stock 报错；
  3. 真实环境：deficit 配置下字段与 qkp.get_level(e) < 阈值逐边一致；默认配置下为 None。
"""

from __future__ import annotations

import copy

import pytest
import torch

from qkd_rl.rl.algos.rollout_buffer import USAGE_LABEL_MODES, RolloutBuffer, RolloutStep


class _Obs:
    def __init__(self, physical_edge_ids, mask, low=None):
        self.physical_edge_ids = physical_edge_ids
        self.request_path_edge_mask = mask
        self.physical_edge_low_stock = low


def _step(obs):
    return RolloutStep(
        obs=obs, actions={}, log_probs={}, entropies={}, value=torch.zeros(()),
        reward=0.0, terminated=False, truncated=False,
    )


def _run(obs_list, horizon, mode):
    steps = [_step(o) for o in obs_list]
    buf = RolloutBuffer(0.99, 0.95, torch.device("cpu"))
    buf.usage_horizon = horizon
    buf.usage_label_mode = mode
    buf.steps.extend(steps)
    buf.finish_episode(0.0)
    return [s.usage_label.tolist() for s in steps]


def _episode(low_by_step):
    ids = ["a", "b", "c", "d"]
    F, T = False, True
    masks = [
        [F, F, F, T],   # t0: d on path now
        [T, T, F, T],   # t1: a, b newly on path
        [F, F, T, F],   # t2: c newly on path
        [F, F, F, F],
    ]
    return [_Obs(list(ids), m, low) for m, low in zip(masks, low_by_step)]


def test_deficit_mode_registered():
    assert "deficit" in USAGE_LABEL_MODES


def test_deficit_is_new_and_low_stock_at_t():
    F, T = False, True
    # t0 时 a 低库存、b 有存货、c 低库存、d 低库存（但 d 此刻已在路径上 ⟹ 不是 new）
    low = [[T, F, T, T], [F, F, F, F], [T, T, T, T], [T, T, T, T]]
    new = _run(_episode(low), horizon=2, mode="new")
    deficit = _run(_episode(low), horizon=2, mode="deficit")
    # new@t0 窗口 (0,2]: a,b,c 新出现，d 在 t0 已在路径 ⟹ [1,1,1,0]
    assert new[0] == [1.0, 1.0, 1.0, 0.0]
    # deficit@t0 = new ∧ low@t0 ⟹ b 有存货被去掉
    assert deficit[0] == [1.0, 0.0, 1.0, 0.0]
    # t1 全部有存货 ⟹ deficit 全 0，而 new 不是
    assert sum(new[1]) > 0 and deficit[1] == [0.0, 0.0, 0.0, 0.0]
    # 逐元素 deficit ≤ new
    for n_row, d_row in zip(new, deficit):
        assert all(d <= n for d, n in zip(d_row, n_row))


def test_deficit_uses_ids_not_positions():
    # step t 的边顺序与未来步不同：标签按 id 对齐、low 按 step t 的位置对齐
    F, T = False, True
    o0 = _Obs(["b", "a"], [F, F], [T, F])        # b 低库存, a 有存货
    o1 = _Obs(["a", "b"], [T, T], [T, T])
    lab = _run([o0, o1], horizon=1, mode="deficit")
    assert lab[0] == [1.0, 0.0]


def test_deficit_without_low_stock_field_is_an_error():
    obs = _episode([None, None, None, None])
    with pytest.raises(ValueError, match="physical_edge_low_stock"):
        _run(obs, horizon=2, mode="deficit")


# ---- 真实环境（服务器 H5）端到端 ----

BASE = ["env_full_demand_v2.yaml", "no_attention.yaml"]


def _env(extra):
    from qkd_rl.env.factory import build_env_from_config
    from scripts.experiment import resolve_experiment

    cfg = copy.deepcopy(resolve_experiment("v8", 42, 1, BASE + extra, "v8_deficit_test"))
    env = build_env_from_config(cfg)
    obs = env.reset(seed=100)
    return cfg, env, obs


def test_builder_low_stock_matches_qkp_levels():
    cfg, env, obs = _env(["aux_deficit.yaml"])
    ucfg = cfg["model"]["actor"]["usage_forecast"]
    assert ucfg["label_mode"] == "deficit"
    thr = float(ucfg["deficit_threshold"])
    ids = list(obs.physical_edge_ids)
    assert len(ids) >= 2
    # 给一条边存 2×阈值，另一条存 0.5×阈值
    env.qkp.add_keys(ids[0], 2.0 * thr, env.t)
    env.qkp.add_keys(ids[1], 0.5 * thr, env.t)
    obs = env._build_observation()
    low = obs.physical_edge_low_stock
    assert low is not None and len(low) == len(obs.physical_edge_ids)
    expect = [float(env.qkp.get_level(e)) < thr for e in obs.physical_edge_ids]
    assert list(low) == expect
    pos = {e: i for i, e in enumerate(obs.physical_edge_ids)}
    assert low[pos[ids[0]]] is False and low[pos[ids[1]]] is True


def test_builder_default_config_leaves_field_none():
    _cfg, _env_, obs = _env([])
    assert obs.physical_edge_low_stock is None
