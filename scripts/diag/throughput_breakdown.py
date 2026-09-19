#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""吞吐分解：每环境步耗时 + 各配置的代价倍数。只读，零内存成本。

为什么现在做：节点的并发上限**由内存定**（实测 6 run / 225.7G，CPU 只用 36/128 核）。
所以"再快一点"有两条路，必须分清是哪一条：

  (A) 每个 run 更快      —— 改代码/线程数（记忆说代码层已无油水、线程数是最优 8）
  (B) 同时跑更多 run      —— 只能靠**降单 run 内存**，因为内存才是约束

本脚本量的是 (A) 里唯一还没在本节点量过的东西：
**每环境步的真实耗时**，以及各配置相对基准的**代价倍数**。
"""
import json
import os
import statistics as st

R = "/opt/qkd/graph_mappo/outputs"

# 配置 → 该跑哪个 run 作代表（都用种子 42，同代码同节点）
CASES = [
    ("基准 ent01 (mb=256)", "ent01_rerun_s42"),
    ("gae90  (λ=0.90)", "gae90_s42"),
    ("vcoef075 (vc=0.75)", "vcoef075_s42"),
    ("mini512 (mb=512)", "mini512_s42"),
    ("hist32 (seq_len=32)", "hist32v3_s42"),
]


def load(run):
    mp = os.path.join(R, run, "metrics.jsonl")
    dp = os.path.join(R, run, "rollout_debug.jsonl")
    if not os.path.exists(mp):
        return None, None
    rows = [json.loads(l) for l in open(mp, encoding="utf-8")
            if l.strip() and "update" in l]
    dbg = {}
    if os.path.exists(dp):
        for l in open(dp, encoding="utf-8"):
            if l.strip():
                o = json.loads(l)
                dbg[o["update"]] = o
    return rows, dbg


print("=" * 78)
print("吞吐分解（只取 u5 之后的稳态段，避开首轮空 buffer）")
print("=" * 78)
print()
hdr = ("%-22s %6s %9s %9s %9s %11s %10s"
       % ("配置", "轮数", "rollout_s", "update_s", "合计_s", "步/轮", "ms/步"))
print(hdr)
print("-" * 78)

base = None
summary = []
for label, run in CASES:
    rows, dbg = load(run)
    if not rows:
        print("%-22s （无数据）" % label)
        continue
    st_rows = [x for x in rows if x["update"] >= 5] or rows
    ro = st.mean([x["rollout_s"] for x in st_rows])
    up = st.mean([x["update_s"] for x in st_rows])
    tot = st.mean([x["elapsed_s"] for x in st_rows])
    steps = [dbg[x["update"]]["steps"] for x in st_rows
             if x["update"] in dbg and dbg[x["update"]].get("steps")]
    sp = st.mean(steps) if steps else float("nan")
    ms = (ro / sp * 1000) if steps else float("nan")
    print("%-22s %6d %9.1f %9.1f %9.1f %11.0f %10.2f"
          % (label, len(rows), ro, up, tot, sp, ms))
    summary.append((label, ro, up, tot, sp, ms))
    if base is None:
        base = (ro, up, tot, ms)

print()
print("=== 相对基准的代价倍数 ===")
b_ro, b_up, b_tot, b_ms = base
print("%-22s %10s %10s %10s %10s" % ("配置", "rollout×", "update×", "合计×", "ms/步×"))
print("-" * 78)
for label, ro, up, tot, sp, ms in summary:
    print("%-22s %10.2f %10.2f %10.2f %10.2f"
          % (label, ro / b_ro, up / b_up, tot / b_tot, ms / b_ms))

print()
print("=== 读法 ===")
print("· rollout× ≈ 环境前向的代价（含 action_resolver 的 Gumbel 顺序采样与 kqp 更新）")
print("· update×  ≈ 反传/前向的代价（PPO 的 epochs × minibatches 次模型计算）")
print("· 两者分离 ⟹ 能判断一个改动是拖慢了**环境**还是拖慢了**模型**")
print()
print("=== 内存 vs 速度（并排看，判断旋钮是赚还是亏）===")
print("  实测 PSS：ent01(mb256)≈23.8G ｜ mini512≈28.6G ｜ hist32 ≈55G")
print("  ⟹ 并发上限 =(250−17)/PSS：mb256 → 9 条 ｜ mb512 → 8 条 ｜ hist32 → 4 条")
