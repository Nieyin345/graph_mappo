"""注意力消融臂（configs/no_attention.yaml）必须真的去掉注意力，且单图/批量两条路径一致。

为什么要测：v8 的 `attention_only: false` 分支此前从未在训练里用过
（v3/v8 所有 run 都是 attention_only）。v8 历史上两次静默失效都出在
"单图路径（rollout 采样）与批量路径（PPO 重打分）不一致"，比率失真但
run 照常跑完。这里在无注意力配置下重钉同样的不变量。
"""

from __future__ import annotations

import copy

import pytest
import torch

from qkd_rl.env.factory import build_env_from_config
from qkd_rl.model_zoo import build_model
from scripts.experiment import resolve_experiment

BASE = ["env_full_demand_v2.yaml", "no_attention.yaml"]


def _build(extra=(), bias_weight=None, seed_model=0):
    cfg = copy.deepcopy(resolve_experiment("v8", 42, 1, BASE + list(extra), "v8_noatt_test"))
    if bias_weight is not None:
        cfg["model"]["actor"]["usage_forecast"]["bias_weight"] = bias_weight
    env = build_env_from_config(cfg)
    torch.manual_seed(seed_model)
    model = build_model(env.action_resolver.action_space, cfg)
    model.eval()
    obs = env.reset(seed=100)
    return cfg, env, model, obs


@pytest.fixture(scope="module")
def built():
    return _build(bias_weight=0.2)


def test_attention_is_really_off(built):
    cfg, _env, model, _obs = built
    actor = model.actor
    assert actor.pair_path_scorer is None, "no_attention 下仍构造了 pair-path 注意力"
    assert not actor.attention_only
    assert actor.edge_scorer is not None, "无注意力时必须有 MLP 边打分器"
    assert actor.usage_forecast is not None, "no_attention 不应顺带关掉预测头"


def test_forecast_off_turns_v8_into_v3_shape():
    cfg, _env, model, _obs = _build(extra=["forecast_off.yaml"])
    assert model.actor.usage_forecast is None
    assert not cfg["features"]["edge"].get("include_path_recency", False)
    assert model.actor.pair_path_scorer is None


def _single_scores(model, obs):
    out = model(obs, device="cpu")
    assert out.edge_scores, "单图路径没有 edge_scores"
    return torch.sort(torch.stack([v.detach() for v in out.edge_scores.values()])).values


def _batched_scores(model, obs):
    out = model.batched_forward([obs], torch.device("cpu"), want_edge_maps=True)
    return torch.sort(out.edge_arrays[0][2].detach()).values


def test_single_and_batched_arc_scores_agree(built):
    """rollout 用单图路径采样、PPO 用批量路径重打分：两者必须给出同一组弧分数。"""
    _cfg, _env, model, obs = built
    with torch.no_grad():
        s = _single_scores(model, obs)
        b = _batched_scores(model, obs)
    assert s.numel() and s.shape == b.shape
    torch.testing.assert_close(b, s, atol=1e-5, rtol=1e-4)


def test_bias_reaches_both_paths_without_attention():
    """偏置在注意力之后统一加；关掉注意力后两条路径都必须仍受 bias_weight 影响。"""
    _c0, _e0, m0, obs0 = _build(bias_weight=0.0)
    _c1, _e1, m1, obs1 = _build(bias_weight=5.0)
    m1.load_state_dict(m0.state_dict())
    with torch.no_grad():
        s0, s1 = _single_scores(m0, obs0), _single_scores(m1, obs1)
        b0, b1 = _batched_scores(m0, obs0), _batched_scores(m1, obs1)
    assert not torch.allclose(s0, s1), "单图路径上 bias 无效"
    assert not torch.allclose(b0, b1), "批量路径上 bias 无效"
