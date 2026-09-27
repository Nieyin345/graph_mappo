"""端口冲突注意力（model.actor.conflict_attention，2026-09-27）。

动机（实验日志 §11.12 失败解剖）：前瞻/囤货方向测不出效应（§11.13–11.14）后，剩下最大的可控损失是
端口冲突——需要的跳可见时，其端点端口被别的候选弧占着（2× 2.8 点、3× 6.7 点）。现有打分对每条候选弧
独立打分，采样器只在可行性层面删掉冲突弧，分数本身看不到竞争者。本模块让每条候选弧 attend 到
与它争同一端口的候选弧（同发端 = 争 Tx，同收端 = 争 Rx，另加反向弧 = 同一条物理边），
输出一个零初始化的残差分数。

不变量：
  1. 掩码按手算值（每节点 1 Tx + 1 Rx）；
  2. 弧 a 的残差只依赖与它冲突的弧；没有冲突弧时残差为 0（不是 NaN）；
  3. 零初始化：刚构造时残差恒为 0 ⟹ 分数与去掉该模块完全相同（可直接从旧 BC 起点 resume）；
  4. 单图路径（rollout 采样）与多图批量路径（PPO 重打分）逐图给出同一组弧分数；
  5. 构造该模块不推进全局 RNG ⟹ 其余所有参数与不开该模块时逐位相同（预测头不在 BC 里、靠随机初始化，
     否则冲突臂会比对照多一个“预测头初始化不同”的变量；b9 首轮实测 update-0 rollout 已分叉）。
"""

from __future__ import annotations

import copy

import pytest
import torch

from qkd_rl.model_zoo.v8.model_impl import ConflictResidual, conflict_mask

# 弧：0: 0→1 | 1: 0→2（与 0 争 Tx@0）| 2: 3→1（与 0 争 Rx@1）| 3: 1→0（0 的反向弧）| 4: 2→3（与谁都不争）
SRC = torch.tensor([0, 0, 3, 1, 2])
DST = torch.tensor([1, 2, 1, 0, 3])


def _randomize_out(module: ConflictResidual, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    last = module.out[-1]
    with torch.no_grad():
        last.weight.copy_(torch.randn(last.weight.shape, generator=g))
        last.bias.copy_(torch.randn(last.bias.shape, generator=g))


def test_conflict_mask_hand_computed():
    m = conflict_mask(SRC, DST)
    assert m.dtype == torch.bool and m.shape == (5, 5)
    assert m[0].tolist() == [False, True, True, True, False]
    assert m[1].tolist() == [True, False, False, False, False]
    assert m[2].tolist() == [True, False, False, False, False]
    assert m[3].tolist() == [True, False, False, False, False]
    assert m[4].tolist() == [False] * 5
    assert torch.equal(m, m.T)


def test_residual_depends_only_on_conflicting_arcs():
    torch.manual_seed(0)
    mod = ConflictResidual(arc_dim=12, hidden_dim=8, num_heads=2, activation="relu", dropout=0.0)
    _randomize_out(mod)
    x = torch.randn(5, 12)
    base = mod(x, SRC, DST)
    assert base.shape == (5,)
    assert torch.isfinite(base).all()
    assert base[4].item() == 0.0, "没有冲突弧的弧残差必须为 0"
    # 改动弧 4（与弧 0 不冲突）→ 弧 0 不变；改动弧 1（冲突）→ 弧 0 变
    x2 = x.clone(); x2[4] += 5.0
    assert torch.allclose(mod(x2, SRC, DST)[0], base[0])
    x3 = x.clone(); x3[1] += 5.0
    assert not torch.allclose(mod(x3, SRC, DST)[0], base[0])


def test_residual_zero_at_init_and_empty():
    torch.manual_seed(0)
    mod = ConflictResidual(arc_dim=12, hidden_dim=8, num_heads=2, activation="relu", dropout=0.0)
    assert torch.equal(mod(torch.randn(5, 12), SRC, DST), torch.zeros(5))
    empty = mod(torch.zeros(0, 12), SRC[:0], DST[:0])
    assert empty.shape == (0,)


# ---- 真实环境端到端（服务器 H5） ----

BASE = ["env_full_demand_v2.yaml", "no_attention.yaml", "v8_bias.yaml", "conflict_attention.yaml"]


@pytest.fixture(scope="module")
def built():
    from qkd_rl.env.factory import build_env_from_config
    from qkd_rl.model_zoo import build_model
    from scripts.experiment import resolve_experiment

    cfg = copy.deepcopy(resolve_experiment("v8", 42, 1, BASE, "v8_conflict_test"))
    env = build_env_from_config(cfg)
    torch.manual_seed(0)
    model = build_model(env.action_resolver.action_space, cfg)
    model.eval()
    obs_list = [env.reset(seed=s) for s in (100, 101, 102, 103, 104)]
    return cfg, env, model, obs_list


def _single(model, obs):
    out = model(obs, device="cpu")
    return torch.sort(torch.stack([v.detach() for v in out.edge_scores.values()])).values


def _batched(model, obs_list):
    out = model.batched_forward(obs_list, torch.device("cpu"), want_edge_maps=True)
    return [torch.sort(out.edge_arrays[i][2].detach()).values for i in range(len(obs_list))]


def test_module_is_built(built):
    cfg, _env, model, _obs = built
    assert cfg["model"]["actor"]["conflict_attention"]["enabled"]
    assert isinstance(model.actor.conflict_residual, ConflictResidual)
    assert model.actor.pair_path_scorer is None


def test_zero_init_equals_model_without_module(built):
    _cfg, _env, model, obs_list = built
    with torch.no_grad():
        with_mod = [_single(model, o) for o in obs_list]
        saved = model.actor.conflict_residual
        model.actor.conflict_residual = None
        try:
            without = [_single(model, o) for o in obs_list]
        finally:
            model.actor.conflict_residual = saved
    for a, b in zip(with_mod, without):
        assert torch.equal(a, b)


def test_single_and_multigraph_batched_agree(built):
    """rollout 单图采样、PPO 多图批量重打分：残差非零时两条路径逐图一致（v8 两次静默失效都在这里）。"""
    _cfg, _env, model, obs_list = built
    m = copy.deepcopy(model)
    _randomize_out(m.actor.conflict_residual, seed=1)
    with torch.no_grad():
        singles = [_single(m, o) for o in obs_list]
        batched = _batched(m, obs_list)
        plain = [_single(model, o) for o in obs_list]
    assert any(not torch.allclose(s, p) for s, p in zip(singles, plain)), "残差非零后分数应当改变"
    assert len({s.numel() for s in singles}) > 1, "批里的图尺寸应当不同，才能测到逐图偏移"
    for s, b in zip(singles, batched):
        assert s.shape == b.shape
        torch.testing.assert_close(b, s, atol=1e-5, rtol=1e-4)


def test_construction_leaves_other_params_identical(built):
    from qkd_rl.model_zoo import build_model
    from scripts.experiment import resolve_experiment

    _cfg, env, model_cf, _obs = built
    cfg0 = copy.deepcopy(resolve_experiment("v8", 42, 1, BASE[:-1], "v8_conflict_test0"))
    torch.manual_seed(0)
    model0 = build_model(env.action_resolver.action_space, cfg0)
    sd_cf, sd0 = model_cf.state_dict(), model0.state_dict()
    extra = sorted(k for k in sd_cf if k not in sd0)
    assert extra and all("conflict_residual" in k for k in extra)
    assert set(sd0) <= set(sd_cf)
    diff = [k for k in sd0 if not torch.equal(sd0[k], sd_cf[k])]
    assert not diff, f"开启冲突注意力改变了其他参数的初始化：{diff[:5]}"
