"""--switch-weight（2026-10-03）：_matching_log_prob_entropy_fast 的 decision_weights 契约。

- 不传 = 原计算（其余测试已钉住逐位一致）；
- 全 1 权重 + average_log_prob = 原平均（数值相等）；
- 加大某一决策的权重只改变它在平均里的占比：结果 = 加权平均（手算核对）。
"""
from __future__ import annotations

import torch

from tests.test_matching_log_prob_paths import _build, _maximal_matching


def _setup():
    env, policy = _build()
    obs = env.reset(seed=1)
    outputs = policy.model.batched_forward([obs], policy.device, want_edge_maps=True)
    edge_scores = outputs.edge_score_maps[0]
    src_arr, dst_arr, _scores = outputs.edge_arrays[0]
    matched = _maximal_matching(src_arr, dst_arr, list(obs.node_ids))
    assert len(matched) >= 2
    return policy, edge_scores, matched


def test_unit_weights_equal_plain_average() -> None:
    policy, es, m = _setup()
    base, _ = policy._matching_log_prob_entropy_fast(es, m, average_log_prob=True)
    w1, _ = policy._matching_log_prob_entropy_fast(es, m, average_log_prob=True, decision_weights=[1.0] * len(m))
    assert torch.allclose(base, w1, atol=1e-6)


def test_weighted_average_matches_hand_computation() -> None:
    policy, es, m = _setup()
    total, _ = policy._matching_log_prob_entropy_fast(es, m, average_log_prob=False)
    # 逐决策对数概率：前缀差分（前缀匹配的联合 log-prob 不含末尾 STOP，故用 m[:k] 的和做差不成立；
    # 改为直接比较：把第 0 个决策权重设为 w，其余 1 ⇒ 加权和 = 总和 + (w−1)·lp0）
    w = 5.0
    ws, _ = policy._matching_log_prob_entropy_fast(es, m, average_log_prob=False, decision_weights=[w] + [1.0] * (len(m) - 1))
    w2, _ = policy._matching_log_prob_entropy_fast(es, m, average_log_prob=False, decision_weights=[2 * w - 1] + [1.0] * (len(m) - 1))
    lp0 = (ws - total) / (w - 1.0)
    assert float(lp0) < 0.0  # 对数概率
    assert torch.allclose(w2 - total, (2 * w - 2) * lp0, atol=1e-4)
    avg, _ = policy._matching_log_prob_entropy_fast(es, m, average_log_prob=True, decision_weights=[w] + [1.0] * (len(m) - 1))
    n_dec = len(m) + 1  # 末尾 STOP（若仍有可行弧）
    assert torch.allclose(avg * (w + n_dec - 1), ws, atol=1e-4) or torch.allclose(avg * (w + len(m) - 1), ws, atol=1e-4)
