# -*- coding: utf-8 -*-
"""生成量掉 41%，服务量掉了吗？—— 分辨「效率提升」与「一起塌掉」。

mean_generated_keys 单调降 41%（10.93M → 6.47M）而 success_rate / reward 都平。
两种可能，含义完全不同：

  (a) served **不降** → 用更少的密钥服务同样多的需求 ⟹ **效率真的提升了**，
      只是 success_rate = served/arrived 看不见它（arrived 也同比例降）。
      这是"指标没测到的好事"。
  (b) served **同比例降** → 需求也在少，等价于什么都没变（或更糟）。

判据：把 generated / arrived / served / failed 放在同一条轴上比降幅。
起始日是**每回合随机**抽的（env.py:96），所以若只是天数构成在漂，
四条线应当**同比例**漂；只有 (a) 才会出现 served 降幅远小于 generated。
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
KEYS = ["mean_generated_keys", "mean_arrived_keys", "mean_served_keys",
        "mean_failed_keys", "mean_waiting_keys", "mean_success_rate",
        "mean_reward", "mean_qkp_utilization", "mean_activated_edges"]

fams = {"ent01": ["ent01_s42", "ent01_s43", "ent01_s44"]}
for fam, names in fams.items():
    data = {}
    for n in names:
        p = OUT / n / "rollout_debug.jsonl"
        if not p.exists():
            continue
        rows = []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if rows:
            data[n] = rows
    if not data:
        print(f"{fam}: 无数据")
        continue
    L = min(len(r) for r in data.values())
    print("=" * 100)
    print(f"{fam}   （{len(data)} 个种子；每格 = 均值）")
    print("=" * 100)
    title = {"mean_generated_keys": "generated", "mean_arrived_keys": "arrived",
             "mean_served_keys": "served", "mean_failed_keys": "failed",
             "mean_waiting_keys": "waiting", "mean_success_rate": "succ",
             "mean_reward": "reward", "mean_qkp_utilization": "qkp_util",
             "mean_activated_edges": "act_edges"}
    print(f"  {'轮':>3}" + "".join(f"{title[k]:>12}" for k in KEYS))
    for i in range(L):
        if i < 4 or (i + 1) % 5 == 0 or i == L - 1:
            cells = []
            for k in KEYS:
                v = [r[i].get(k) for r in data.values() if r[i].get(k) is not None]
                if not v:
                    cells.append(f"{'—':>12}")
                    continue
                m = st.mean(v)
                cells.append(f"{m:>12,.0f}" if abs(m) > 100 else f"{m:>12.4f}")
            print(f"  {i+1:>3}" + "".join(cells))
    print()
    print("  首 3 轮 → 末 3 轮的变化率：")
    for k in KEYS:
        per = []
        for r in data.values():
            v = [x.get(k) for x in r if x.get(k) is not None]
            if len(v) >= 6:
                a, b = st.mean(v[:3]), st.mean(v[-3:])
                if a:
                    per.append(b / a - 1)
        if per:
            print(f"    {title[k]:<12} {st.mean(per):>+9.1%}   "
                  + "  ".join(f"{x:+.1%}" for x in per))
    print()
