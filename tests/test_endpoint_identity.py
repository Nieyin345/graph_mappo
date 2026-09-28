"""节点失效评测：scenario.active_nodes.fail 与 model.endpoint_identity: reference（改进项 4，2026-09-28）。

完整模型的路径注意力 token 里拼了 endpoint_identity[节点位置]（位置 one-hot，
n ≤ hidden 时为 eye）。训练场景是 smart 选择 + exclude（88 节点，地理排序，不是注册表顺序）。
删节点后位置前移 ⟹ 默认模式下后面每个节点都拿到"别的节点"的身份 —— 失效评测被身份错位混淆。

  fail：在 smart 选择与排序**之后**删节点（默认空 = 行为不变），连带删掉其边；未知名字报错。
  reference：身份按无失效场景（reference_node_ids = 去掉 fail 的场景顺序）取行。

钉住：
  1. reference_node_ids == 无失效场景的节点顺序；
  2. 无失效时 reference 与 position 逐位相同 ⟹ 既有结果不受影响；
  3. fail 后每个存活节点的身份行 = 无失效模型里同名节点的身份行；
  4. position 模式确实会错位（否则开关没有意义）；
  5. 无失效训练的 state_dict 能 strict 加载到失效网络模型并前向；
  6. fail 删除节点及其边、未知名字报错、未知 endpoint_identity 报错。
"""

from __future__ import annotations

import copy

import pytest
import torch

BASE = ["env_full_demand_v2.yaml", "load_x20.yaml", "v8_bias.yaml"]
FAIL = ["Sat_LEO_1d", "Sat_LEO_2e", "HAP_Chicago"]


def _cfg(fail=None, ident=None):
    from scripts.experiment import resolve_experiment

    cfg = copy.deepcopy(resolve_experiment("v8", 42, 1, BASE, "v8_ident_test"))
    if fail:
        cfg.setdefault("scenario", {}).setdefault("active_nodes", {})["fail"] = list(fail)
    if ident:
        cfg["model"]["endpoint_identity"] = ident
    return cfg


def _model(fail=None, ident=None):
    from qkd_rl.env.factory import build_env_from_config
    from qkd_rl.model_zoo import build_model

    cfg = _cfg(fail, ident)
    env = build_env_from_config(cfg)
    obs = env.reset(seed=100)
    torch.manual_seed(0)
    model = build_model(env.action_resolver.action_space, cfg)
    model.eval()
    return cfg, env, obs, model


def _rows(model):
    return {n: model.endpoint_identity[i].clone() for i, n in enumerate(model.action_space.node_ids)}


def test_reference_ids_match_failure_free_scenario_order():
    from qkd_rl.data.scenario_builder import reference_node_ids

    cfg, env, _obs, model = _model()
    ids = [n.node_id for n in env.scenario.nodes]
    assert reference_node_ids(cfg) == ids == list(model.action_space.node_ids)
    # fail 不改变参照顺序
    assert reference_node_ids(_cfg(fail=FAIL)) == ids


def test_no_failure_reference_equals_position():
    _c, _e, _o, pos = _model()
    _c, _e, _o, ref = _model(ident="reference")
    assert torch.equal(pos.endpoint_identity, ref.endpoint_identity)


def test_failure_keeps_each_node_identity():
    _c, _e, _o, full = _model()
    _c, env, _o, red = _model(fail=FAIL, ident="reference")
    names = list(red.action_space.node_ids)
    assert len(names) == len(full.action_space.node_ids) - len(FAIL)
    assert not set(FAIL) & set(names)
    ref = _rows(full)
    for n, row in _rows(red).items():
        assert torch.equal(row, ref[n]), n


def test_position_mode_shifts_identities_after_failure():
    _c, _e, _o, full = _model()
    _c, _e, _o, red = _model(fail=FAIL)
    ref = _rows(full)
    assert [n for n, row in _rows(red).items() if not torch.equal(row, ref[n])]


def test_full_state_dict_loads_into_failed_network_and_forwards():
    _c, _e, _o, full = _model()
    _c, _e, obs, red = _model(fail=FAIL, ident="reference")
    red.load_state_dict(full.state_dict(), strict=True)
    with torch.no_grad():
        out = red(obs, device="cpu")
    assert out.edge_scores
    assert torch.isfinite(out.value).all()


def test_fail_removes_nodes_and_their_edges():
    from qkd_rl.env.factory import build_env_from_config

    full = build_env_from_config(_cfg())
    red = build_env_from_config(_cfg(fail=FAIL))
    assert {n.node_id for n in full.scenario.nodes} - {n.node_id for n in red.scenario.nodes} == set(FAIL)
    touching = [e for e in full.scenario.edges if e.src in FAIL or e.dst in FAIL]
    assert touching
    assert len(red.scenario.edges) == len(full.scenario.edges) - len(touching)
    assert not [e for e in red.scenario.edges if e.src in FAIL or e.dst in FAIL]


def test_fail_unknown_name_is_an_error():
    from qkd_rl.env.factory import build_env_from_config

    with pytest.raises(ValueError, match="fail"):
        build_env_from_config(_cfg(fail=["Sat_LEO_9z"]))


def test_unknown_identity_mode_is_an_error():
    with pytest.raises(ValueError, match="endpoint_identity"):
        _model(ident="bogus")
