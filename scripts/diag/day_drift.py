# -*- coding: utf-8 -*-
"""训练回合的「天」在漂吗？—— 若漂，所有训练 regime 的「平坦」读数都有嫌疑。

动机：rollout_debug.jsonl 里 mean_generated_keys 从 11.1M 掉到 6.5M（−18%），
而 success_rate 与 reward 都不动。**这是对我自己那条结论的威胁检验**：
我判定「训练侧从 u1 就饱和在 0.857」，但若每轮抽到的起始日在变，
那 0.857 的平坦就可能是两种变化的抵消，而不是饱和。

activation_window 限制的是回合**起始日**，回合可以跑出窗（CLAUDE.md 明写）。
所以只要起始日的抽样分布随 update 变化，构成就会漂。
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
name = "ent01_s42"
p = OUT / name / "rollout_debug.jsonl"
rows = []
for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        continue

print(f"{name}: {len(rows)} 条记录")
print()
print("=" * 92)
print("1. 顶层键（找 day / start / window / step 相关的）")
print("=" * 92)
keys = sorted(rows[0].keys())
print(f"  共 {len(keys)} 个顶层键：")
for k in keys:
    v = rows[0][k]
    t = type(v).__name__
    if isinstance(v, (int, float, str, bool)) or v is None:
        print(f"    {k:<40} {t:<8} 首值 {v!r}"[:112])
    elif isinstance(v, list):
        print(f"    {k:<40} {t:<8} len={len(v)} 首元素 {v[0]!r}"[:112])
    else:
        print(f"    {k:<40} {t:<8} keys={list(v)[:6]}"[:112])

print()
print("=" * 92)
print("2. 候选的 day/start/window 字段逐轮轨迹")
print("=" * 92)
CAND = [k for k in keys if any(t in k.lower() for t in
        ("day", "start", "window", "step", "index", "seed", "episode"))]
if not CAND:
    print("  没有 day/start/window 类字段 —— 无法从本文件判断天数构成")
for k in CAND:
    vals = []
    for r in rows:
        v = r.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(v)
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            vals.append(st.mean(v))
    if len(vals) < 4:
        continue
    h = len(vals) // 2
    print(f"  {k:<36} 首 {vals[0]:>12.2f}  末 {vals[-1]:>12.2f}"
          f"   半段增量 {st.mean(vals[h:]) - st.mean(vals[:h]):>+12.3f}")

print()
print("=" * 92)
print("3. 与 mean_generated_keys 并列看（确认它是不是单调漂）")
print("=" * 92)
gk = [r.get("mean_generated_keys") for r in rows]
print(f"  {'轮':>4}{'generated_keys':>20}{'failed_keys':>16}{'reward':>12}")
for i, r in enumerate(rows, start=1):
    if i <= 5 or i % 5 == 0 or i == len(rows):
        g = r.get("mean_generated_keys")
        f = r.get("mean_failed_keys")
        rw = r.get("mean_reward")
        print(f"  {i:>4}{g:>20,.0f}{f:>16,.0f}{rw:>12.5f}")
if all(isinstance(v, (int, float)) for v in gk if v is not None):
    print(f"\n  首 3 轮均值 {st.mean(gk[:3]):,.0f}   末 3 轮均值 {st.mean(gk[-3:]):,.0f}"
          f"   变化 {st.mean(gk[-3:]) / st.mean(gk[:3]) - 1:+.1%}")
    print("  ⚠ 若单调下降而 success/reward 都平 —— 是「奖励看不见的行为漂移」，")
    print("    不是「策略没有变化」。两者对「瓶颈在哪」的含义不同。")
