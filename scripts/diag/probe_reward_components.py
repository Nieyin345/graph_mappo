# -*- coding: utf-8 -*-
"""奖励分解还在涨吗？—— 用来区分"优化到顶"与"目标错位"。

动机：`success_rate` 在 u30 已近饱和（0.8551→0.8555，50 轮不动），而
密钥效率（生成/服务）却掉了 41%。两个问题：
  1. reward 本身还涨不涨？若 reward 也平 → 真的优化到顶；若 reward 还涨而
     success_rate 平 → 策略在改善**代理目标**，而代理已与指标脱钩。
  2. 涨/平的是**哪一项**？奖励有 served/storage/keep_active/failed/dense 等多项。

数据全在已有 `rollout_debug.jsonl` 里，不跑新实验。

env_full.yaml 的 shaped 配置（实测生效值）：
  served_weight 50, failed_weight 5, expired_key_weight 0.01,
  switch_weight 0.001, keep_active_weight 0.001,
  generated_weight 0, waiting_weight 0, waiting_stock_weight 0,
  dense_generation_importance_weight 0.02, dense_normalize_by_added true

用法（服务器上）：python3 /tmp/probe_reward_components.py
"""
import json
import statistics as st
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
KEYS = [
    "mean_reward", "mean_reward_served", "mean_reward_storage",
    "mean_reward_keep_active", "mean_reward_failed", "mean_reward_generated",
    "mean_reward_dense", "mean_reward_waiting", "mean_reward_expired",
    "mean_reward_switch", "mean_reward_conflict",
    # 顺带把"看不见的"几维也放进来对照
    "mean_generated_keys", "mean_served_keys", "mean_activated_edges",
]
TOTAL = "mean_reward"

RUNS = ["ent01_s42", "ent01_s43", "ent01_s44"]

print("=" * 78)
print("奖励分项：前 1/3 轮 vs 后 1/3 轮（每轮是 8 集的日均值）")
print("=" * 78)

agg = {}
for run in RUNS:
    p = OUT / run / "rollout_debug.jsonl"
    if not p.exists():
        print(f"{run}: 缺 rollout_debug.jsonl")
        continue
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    n3 = max(1, len(rows) // 3)
    print(f"\n{run}  轮数={len(rows)}")
    print(f"  {'key':<24}{'前1/3':>14}{'后1/3':>14}{'变化':>12}")
    for k in KEYS:
        v = [r[k] for r in rows if r.get(k) is not None]
        if not v:
            continue
        a = st.mean(v[:n3])
        b = st.mean(v[-n3:])
        pct = (b / a - 1) * 100 if a else 0.0
        agg.setdefault(k, []).append((a, b))
        print(f"  {k:<24}{a:>14.4f}{b:>14.4f}{pct:>11.1f}%")

print("\n" + "=" * 78)
print("三种子合并（各分项先按种子平均）")
print("=" * 78)
print(f"  {'key':<24}{'前1/3':>14}{'后1/3':>14}{'变化':>12}")
for k in KEYS:
    if k not in agg or not agg[k]:
        continue
    a = st.mean(x[0] for x in agg[k])
    b = st.mean(x[1] for x in agg[k])
    pct = (b / a - 1) * 100 if a else 0.0
    print(f"  {k:<24}{a:>14.4f}{b:>14.4f}{pct:>11.1f}%")

# 分项占 |total| 的比重，判断哪一项在数值上主导。
print("\n" + "=" * 78)
print("分项量级（后 1/3，三种子均值）—— 看谁在数值上主导")
print("=" * 78)
tot = st.mean(x[1] for x in agg.get(TOTAL, [(0, 0)]))
for k in KEYS:
    if k == TOTAL or k not in agg or not agg[k]:
        continue
    b = st.mean(x[1] for x in agg[k])
    share = b / tot * 100 if tot else 0.0
    print(f"  {k:<24}{b:>14.4f}{share:>10.1f}% of total")
print(f"  {TOTAL:<24}{tot:>14.4f}")
