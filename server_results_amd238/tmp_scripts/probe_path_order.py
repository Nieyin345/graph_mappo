#!/usr/bin/env python
"""判定：rollout 路径与 update 路径的 1.19e-07 差异是**真差异**还是**我测试里的顺序伪影**。

关键问题：
  1. edge_score_maps 的键顺序 与 edge_arrays 的顺序是否一致？
  2. 三个实现是否**依赖 dict 的插入顺序**（浮点累加顺序不同 → 末位不同）？
  3. 真实 rollout 调用 log_prob_entropy_for_matching 时，传的 dict 是什么顺序？

若(2)成立且(1)不成立，那么差异是我构造 dict 时用了另一种顺序造成的，
**真实训练里两条路径仍是逐位一致的**——不能报成"PPO ratio 有偏"。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/opt/qkd/graph_mappo")))
sys.path.insert(0, str(Path("/opt/qkd/graph_mappo/tests")))

import torch
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.rl.algos.policy import MAPPOPolicy
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
from tests.helpers import ROOT, point_config_to_h5

cfg = point_config_to_h5(load_default_config(ROOT))
env = build_env_from_config(cfg)
model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
policy = MAPPOPolicy(model, "cpu")

for seed in (1, 2, 3, 5):
    obs = env.reset(seed=seed)
    out = policy.model.batched_forward([obs], policy.device, want_edge_maps=True)
    esm = out.edge_score_maps[0]
    src_arr, dst_arr, scores = out.edge_arrays[0]
    node_ids = list(obs.node_ids)
    node_pos = {n: i for i, n in enumerate(node_ids)}

    order_maps = list(esm.keys())
    order_arrs = [(node_ids[int(s)], node_ids[int(d)]) for s, d in zip(src_arr.tolist(), dst_arr.tolist())]
    same_order = order_maps == order_arrs
    same_set = set(order_maps) == set(order_arrs)

    print(f"--- seed {seed}: n_maps={len(order_maps)} n_arrays={len(order_arrs)}")
    print(f"    键集合相同: {same_set}   键顺序相同: {same_order}")
    if not same_order and same_set:
        for i, (a, b) in enumerate(zip(order_maps, order_arrs)):
            if a != b:
                print(f"    首个顺序分歧 @{i}: maps={a}  arrays={b}")
                break

    # 值是否逐位相同（键相同的意义下）
    val_mismatch = 0
    for k, v in esm.items():
        j = order_arrs.index(k) if k in order_arrs else None
        if j is None:
            continue
        if float(v) != float(scores[j]):
            val_mismatch += 1
    print(f"    同一键上的浮点值不等个数: {val_mismatch}")

    # 三路比较：同一个 matched_edges，比较 fast / arrays / rollout
    sampled = list(policy.act_batched([obs], deterministic=True, build_scores=True)[0].matched_edges or [])
    if not sampled:
        print("    (无匹配，跳过)")
        continue
    me = sampled

    fast_lp, fast_ent = policy._matching_log_prob_entropy_fast(esm, me)
    arr_lp, arr_ent = policy._matching_log_prob_entropy_arrays(src_arr, dst_arr, scores, node_pos, me)

    # rollout 路径：用 **maps 顺序** 构造 float dict（复现真实调用）
    fs_maps = {k: float(v) for k, v in esm.items()}
    r_maps_lp, r_maps_ent = policy.log_prob_entropy_for_matching(fs_maps, me)

    # rollout 路径：用 **arrays 顺序** 构造（我原来测试的写法）
    fs_arr = {k: float(v) for k, v in zip(order_arrs, scores.tolist())}
    r_arr_lp, r_arr_ent = policy.log_prob_entropy_for_matching(fs_arr, me)

    print(f"    fast    lp={fast_lp.item():.10f}")
    print(f"    arrays  lp={arr_lp.item():.10f}")
    print(f"    rollout(maps顺序)  lp={r_maps_lp.item():.10f}  equal_to_arrays={torch.equal(r_maps_lp, arr_lp)}")
    print(f"    rollout(arrays顺序) lp={r_arr_lp.item():.10f}  equal_to_arrays={torch.equal(r_arr_lp, arr_lp)}")
    print(f"    fast==arrays: {torch.equal(fast_lp, arr_lp)}")
    print()
