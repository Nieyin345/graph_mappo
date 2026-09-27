"""接触计划前瞻特征（features.edge.contact_lookahead）。

为什么有这个特征（实验日志 §11.12 失败解剖）：可控损失里最大的一块是"没提前囤货"——
请求到达前 30–120 步那些边可见过但没存货，而模型只看当前可见边未来 6 步
（prediction_horizon: 6）。卫星接触计划是确定、可预知的，本特征把未来 H 步的可见性
按桶汇总后喂给边编码器，用来检验"这类信息有没有用"。

不变量：
  1. 纯函数按手算值输出（分桶可见比例、当前接触剩余长度、之后的间隙长度）；
  2. provider.get_avail_lookahead 与 get_window_blocks 的可见性逐元素一致（同一实现），
     越过数据末尾的时隙为不可见；
  3. 真实环境里特征维度被 resolve_feature_dims 计入，obs 的最后几列与直接计算一致，模型能前向。
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from qkd_rl.env.contact_features import contact_lookahead_features
from tests.test_h5_rate_provider import _provider, _write_dataset


def test_features_hand_computed():
    T, F = True, False
    avail = np.array([
        # col0 全可见 | col1 两段 | col2 全不可见 | col3 先断一步再连
        [T, T, F, F],
        [T, T, F, T],
        [T, F, F, T],
        [T, F, F, T],
        [T, F, F, T],
        [T, T, F, T],
        [T, T, F, T],
        [T, F, F, T],
    ])
    out = contact_lookahead_features(avail, buckets=4)
    assert out.shape == (4, 6)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0], [1, 1, 1, 1, 1.0, 0.0])
    np.testing.assert_allclose(out[1], [1, 0, 0.5, 0.5, 2 / 8, 3 / 8])
    np.testing.assert_allclose(out[2], [0, 0, 0, 0, 0.0, 1.0])
    np.testing.assert_allclose(out[3], [0.5, 1, 1, 1, 0.0, 1 / 8])


def test_features_gap_without_reappearance_is_one():
    avail = np.array([[True], [True], [False], [False]])
    out = contact_lookahead_features(avail, buckets=2)
    np.testing.assert_allclose(out[0], [1.0, 0.0, 0.5, 1.0])


def test_features_reject_indivisible_buckets():
    with pytest.raises(ValueError):
        contact_lookahead_features(np.zeros((10, 3), dtype=bool), buckets=4)


def test_provider_lookahead_matches_window_blocks(tmp_path):
    _write_dataset(tmp_path, t_steps=500)
    provider = _provider(tmp_path, horizon=60)
    for t in (0, 90, 130, 300):
        la = provider.get_avail_lookahead(t, 40)
        blocks = provider.get_window_blocks(t)
        assert la.shape == (40, blocks[1].shape[1])
        np.testing.assert_array_equal(la, blocks[1][1:41])
    provider.close()


def test_provider_lookahead_outage_and_dataset_end(tmp_path):
    data = _write_dataset(tmp_path, t_steps=500)
    provider = _provider(tmp_path, horizon=6)
    la = provider.get_avail_lookahead(95, 10)  # 时隙 96..105；link 2 在 100–149 中断
    assert la[:4, 2].all() and not la[4:, 2].any()
    assert la[:, 1].all()
    end = provider.get_avail_lookahead(data["T"] - 3, 5)  # 时隙 T-2, T-1 在数据内，其余越界
    assert end.shape == (5, data["L"])
    assert end[:2, 1].all()
    assert not end[2:].any()
    provider.close()


def test_block_availability_requires_los(tmp_path):
    """LOS=0 但速率>0 的时隙必须不可见（原夹具里两者总是同时为 0，测不出漏掉 LOS）。"""
    import h5py

    _write_dataset(tmp_path, t_steps=500)
    with h5py.File(tmp_path / "link_data.h5", "r+") as f:
        f["los"][200:210, 1] = 0
        assert (f["k_max"][200:210, 1] > 0).all()
    provider = _provider(tmp_path, horizon=6)
    la = provider.get_avail_lookahead(195, 20)  # 时隙 196..215
    assert la[:4, 1].all() and not la[4:14, 1].any() and la[14:, 1].all()
    assert not provider.get_window_blocks(200)[1][0, 1]
    provider.close()


# ---- 真实环境（服务器 H5）端到端 ----

BASE = ["env_full_demand_v2.yaml", "no_attention.yaml"]


@pytest.fixture(scope="module")
def la_env():
    from qkd_rl.env.factory import build_env_from_config
    from scripts.experiment import resolve_experiment

    cfg = copy.deepcopy(resolve_experiment("v8", 42, 1, BASE + ["contact_lookahead.yaml"], "v8_la_test"))
    env = build_env_from_config(cfg)
    obs = env.reset(seed=100)
    return cfg, env, obs


def test_dims_include_lookahead(la_env):
    from scripts.experiment import resolve_experiment

    cfg, _env, _obs = la_env
    base = resolve_experiment("v8", 42, 1, BASE, "v8_base_test")
    la = cfg["features"]["edge"]["contact_lookahead"]
    assert la["enabled"]
    assert (cfg["features"]["dims"]["physical_edge_dim_resolved"]
            == base["features"]["dims"]["physical_edge_dim_resolved"] + int(la["buckets"]) + 2)


def test_obs_columns_match_direct_computation(la_env):
    cfg, env, obs = la_env
    la = cfg["features"]["edge"]["contact_lookahead"]
    steps, nb = int(la["steps"]), int(la["buckets"])
    pdim = cfg["features"]["dims"]["physical_edge_dim_resolved"]
    n_phys = len(obs.physical_edge_ids)
    feats = np.asarray(obs.edge_features, dtype=np.float32)[:n_phys, pdim - (nb + 2):pdim]
    idx = env.rate_provider.edge_link_ids(list(obs.physical_edge_ids))
    expect = contact_lookahead_features(env.rate_provider.get_avail_lookahead(obs.state.t, steps)[:, idx], nb)
    np.testing.assert_allclose(feats, expect, atol=1e-6)
    # 特征不是常数（否则检验无意义）
    assert feats[:, nb].std() > 0 and feats[:, :nb].std() > 0


def test_model_forward_with_lookahead(la_env):
    import torch
    from qkd_rl.model_zoo import build_model

    cfg, env, obs = la_env
    torch.manual_seed(0)
    model = build_model(env.action_resolver.action_space, cfg)
    model.eval()
    with torch.no_grad():
        out = model(obs, device="cpu")
    assert out.edge_scores
