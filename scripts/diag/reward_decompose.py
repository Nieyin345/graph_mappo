#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""奖励分解核实：生效奖励里到底哪几项非零？RL 省密钥是有意的还是副产品？

### 为什么查这个
记忆 `reward-is-the-metric` 说「生效奖励里只有 served(101.4%) 与 failed(1.7%)
非零，其余全恒 0」，且「reward ≈ 1e-6 × arrived × sr」。
记忆 `rl-key-efficiency-is-unconstrained-drift` 说 RL 比专家省 36.5% 密钥，
但**奖励对生成量完全无感** ⟹ 那是**副产品不是能力**，且「正因无人看守才可能漂回去」。

如果这两条都成立，就指着一个**真正的算法改进**：
把「密钥效率」写进奖励 ⟹ 把无人看守的副产品变成**有意学到的能力**。
所以先要**独立核实**这两条——记忆是快照，要以数据为准。

`rollout_debug.jsonl` 里有全部奖励分量（每轮一行，免费）。
"""
import json
import os
import statistics as st

R = "/opt/qkd/graph_mappo/outputs"

# 覆盖不同配置的臂，避免只看一个配置就下结论
RUNS = ["ent01_rerun_s42", "ent01_rerun_s43", "gae90_s42",
        "vcoef075_s42", "mini512_s42"]

COMPONENTS = ["mean_reward_served", "mean_reward_generated", "mean_reward_dense",
              "mean_reward_storage", "mean_reward_keep_active",
              "mean_reward_failed", "mean_reward_waiting",
              "mean_reward_switch", "mean_reward_expired",
              "mean_reward_conflict"]


def load(run):
    p = os.path.join(R, run, "rollout_debug.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


print("=" * 84)
print("奖励分量核实：哪些恒 0、哪些是主项（只取 u5+ 稳态段）")
print("=" * 84)

allrows = {}
for run in RUNS:
    rows = [x for x in load(run) if x.get("update", 0) >= 5]
    if rows:
        allrows[run] = rows

if not allrows:
    print("没有 rollout_debug 数据")
    raise SystemExit(1)

print()
print("%-22s %13s %13s %12s %10s" % ("分量", "跨臂均值", "最大绝对值", "非零臂数", "占比"))
print("-" * 84)

# 先算总量级：用所有分量绝对值之和做分母
base = {}
for run, rows in allrows.items():
    base[run] = st.mean([abs(x.get("mean_reward", 0.0)) for x in rows])
grand = st.mean(list(base.values()))

for comp in COMPONENTS:
    vals, mx, nz = [], 0.0, 0
    for run, rows in allrows.items():
        v = [x.get(comp, 0.0) for x in rows]
        m = st.mean(v)
        vals.append(m)
        mx = max(mx, max(abs(x) for x in v))
        if max(abs(x) for x in v) > 1e-12:
            nz += 1
    gm = st.mean(vals)
    print("%-22s %13.6f %13.6f %8d/%-3d %9.1f%%"
          % (comp, gm, mx, nz, len(allrows),
             100 * abs(gm) / grand if grand else 0))

print()
print("mean_reward（总）跨臂均值 = %.6f" % grand)
print()

# 密钥效率：RL 到底省不省密钥
print("=" * 84)
print("密钥效率：生成 vs 服务（RL 相对专家省 36.5% 的说法）")
print("=" * 84)
print()
print("%-18s %14s %14s %10s" % ("run", "生成(万)", "服务(万)", "服务/生成"))
print("-" * 84)
for run, rows in sorted(allrows.items()):
    g = st.mean([x.get("mean_generated_keys", 0.0) for x in rows])
    s = st.mean([x.get("mean_served_keys", 0.0) for x in rows])
    print("%-18s %14.2f %14.2f %10.4f"
          % (run, g / 1e4, s / 1e4, (s / g) if g else 0))

print()
print("=== 判读 ===")
print("· 若 mean_reward_generated / dense / storage / keep_active 等全恒 0，")
print("  而 served 是唯一主项 ⟹ 记忆 `reward-is-the-metric` 成立，")
print("  且『省密钥』确实是**奖励看不见的方向** ⟹ 副产品。")
print("· 若某一项非零 ⟹ 记忆该订正，且省密钥可能已有激励（要看符号）。")
