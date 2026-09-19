#!/usr/bin/env python
"""把"三条路径是否逐位一致"从模型 RNG 中隔离出来：直接喂随机分数。

动机：新加的 test_rollout_path_matches_array_path_bit_for_bit 在**全量测试**里
失败（1.19e-07 ≈ 1 ulp），单独跑却通过；探针里四个种子还给出一模一样的值
（说明测试 env 对种子几乎不敏感）。→ 失败依赖**模型权重的 RNG 状态**，
说明这是"某些分数下才出现"的末位差，而不是稳定结构差。

本探针不建模型，直接构造候选弧 + 随机分数 + 随机可行匹配，比较：
  fast(dict, 来自 edge_score_maps 的语义) vs arrays vs rollout
并统计最大绝对/相对差。若最大差恒为 ~1e-7（1 ulp at float32），
说明只是归约顺序；若出现大差，说明是结构差（弧集/可行行不同）——那才危险。
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/opt/qkd/graph_mappo")))

import torch
from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.rl.algos.policy import MAPPOPolicy
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic
from tests.helpers import ROOT, point_config_to_h5

cfg = point_config_to_h5(load_default_config(ROOT))
env = build_env_from_config(cfg)
model = GraphMAPPOActorCritic(env.action_resolver.action_space, cfg)
policy = MAPPOPolicy(model, "cpu")

obs = env.reset(seed=1)
node_ids = list(obs.node_ids)
node_pos = {n: i for i, n in enumerate(node_ids)}

# 用真实候选弧结构（保证可行行有内容），但分数随机化
out = policy.model.batched_forward([obs], policy.device, want_edge_maps=True)
src_arr, dst_arr, _ = out.edge_arrays[0]
src_l, dst_l = src_arr.tolist(), dst_arr.tolist()
arcs = [(node_ids[s], node_ids[d]) for s, d in zip(src_l, dst_l)]
print(f"候选弧数 = {len(arcs)}")

# 构造一条可行匹配（双端约束）
def feasible_matching(rng, k):
    used_tx, used_rx, used_pair, me = set(), set(), set(), []
    idx = list(range(len(arcs)))
    rng.shuffle(idx)
    for i in idx:
        if len(me) >= k:
            break
        s, d = src_l[i], dst_l[i]
        pair = (s, d) if s <= d else (d, s)
        if s in used_tx or d in used_rx or pair in used_pair:
            continue
        used_tx.add(s); used_rx.add(d); used_pair.add(pair)
        me.append(arcs[i])
    return me

rng = random.Random(0)
worst_abs = 0.0
worst_rel = 0.0
n_compared = 0
struct_fail = 0
scales = [0.1, 1.0, 5.0, 20.0]

for trial in range(300):
    scale = scales[trial % len(scales)]
    vals = [rng.gauss(0, scale) for _ in arcs]
    scores = torch.tensor(vals, dtype=torch.float32)

    # arrays 路径的输入
    a_src = torch.tensor(src_l, dtype=torch.long)
    a_dst = torch.tensor(dst_l, dtype=torch.long)

    me = feasible_matching(rng, rng.randint(1, 5))
    if not me:
        continue

    # dict 形式（两种顺序都试，看顺序是否有影响）
    d_maps = {a: float(v) for a, v in zip(arcs, vals)}
    d_rev = {a: float(v) for a, v in zip(reversed(arcs), reversed(vals))}

    arr_lp, arr_ent = policy._matching_log_prob_entropy_arrays(
        a_src, a_dst, scores, node_pos, me)
    rl_lp, rl_ent = policy.log_prob_entropy_for_matching(d_maps, me)
    rr_lp, rr_ent = policy.log_prob_entropy_for_matching(d_rev, me)

    for name, a, b in (("rollout(fwd)", rl_lp, arr_lp),
                       ("rollout(rev)", rr_lp, arr_lp),
                       ("rollout_ent(fwd)", rl_ent, arr_ent)):
        av, bv = a.item(), b.item()
        d = abs(av - bv)
        rel = d / max(abs(bv), 1e-12)
        n_compared += 1
        if d > worst_abs:
            worst_abs = d
        if rel > worst_rel:
            worst_rel = rel
        if d > 1e-4:
            struct_fail += 1
            if struct_fail <= 3:
                print(f"  !! 大差异 trial={trial} {name}: {av!r} vs {bv!r} d={d:.3e}")

print()
print(f"比较次数          = {n_compared}")
print(f"最大绝对差        = {worst_abs:.3e}")
print(f"最大相对差        = {worst_rel:.3e}")
print(f"结构级差异(>1e-4) = {struct_fail}")
print()
print(f"float32 的 1 ulp 在量级 1.0 处约为 {2**-23:.3e}")
print(f"float32 的 1 ulp 在量级 1.6 处约为 {1.6*2**-23:.3e}")
if struct_fail == 0 and worst_abs < 1e-5:
    print("→ 只有末位归约差，**不是结构差**：两路径数学相同，浮点结合律不同。")
else:
    print("→ 存在结构级差异，需要当 bug 处理。")
