# -*- coding: utf-8 -*-
"""专家在**验证 regime** 上的生成/服务/等待量 —— 顺手就能拿到的对照。

意义：训练侧发现 RL 的生成量单调掉 41% 而 reward/success 都不动
（奖励的零空间）。要判断这是"效率提升"还是"漂坏了"，最需要的就是
**同一个 regime 上专家是多少**。若专家生成量也很低（甚至更低），
说明"少生成"是好的那一侧，RL 漂对了；若专家远高于 RL，说明 RL 漂坏了。

`outputs/eval/expert_seeds100_240.json` 里已经存了 generated_keys。
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

p = Path("/opt/qkd/graph_mappo/outputs/eval/expert_seeds100_240.json")
d = json.loads(p.read_text(encoding="utf-8"))
print("顶层键:", sorted(d.keys()))
print()
for k, v in d.items():
    if isinstance(v, list):
        print(f"  {k:<24} list len={len(v)}  首 3 项 {v[:3]}")
    else:
        print(f"  {k:<24} {type(v).__name__}  {v}")
print()
print("=" * 80)
print("逐种子的各物理量（验证 regime，240 步）")
print("=" * 80)
# 找出可用的数值序列
cand = {k: v for k, v in d.items()
        if isinstance(v, list) and v and isinstance(v[0], (int, float))}
if not cand:
    print("  没有数值序列")
else:
    keys = [k for k in ("success", "success_rate", "served_keys", "arrived_keys",
                        "generated_keys", "failed_keys", "waiting_keys",
                        "qkp_utilization") if k in cand]
    print(f"  {'seed':>5}" + "".join(f"{k[:14]:>16}" for k in keys))
    n = len(cand[keys[0]])
    for i in range(min(n, 16)):
        row = f"  {i:>5}"
        for k in keys:
            v = cand[k][i] if i < len(cand[k]) else None
            row += f"{v:>16,.4f}"[:16] if isinstance(v, float) else f"{v:>16,}"
        print(row)
    print()
    print("  均值 / SD：")
    for k in keys:
        v = [float(x) for x in cand[k]]
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f"    {k:<20} 均值 {st.mean(v):>16,.4f}   SD {sd:>14,.4f}")
    print()
    if "generated_keys" in cand and "served_keys" in cand:
        g = st.mean([float(x) for x in cand["generated_keys"]])
        s = st.mean([float(x) for x in cand["served_keys"]])
        a = st.mean([float(x) for x in cand["arrived_keys"]]) if "arrived_keys" in cand else float("nan")
        print(f"  ★ 专家的 生成/服务 = {g/s:.2f}（RL 训练侧末轮是 "
              f"{6436290/66172:.2f}）" if s else "")
        print(f"    专家 生成={g:,.0f}  服务={s:,.0f}  到达={a:,.0f}")
