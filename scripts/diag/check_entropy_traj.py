#!/usr/bin/env python
"""ent01 的 entropy / kl / ratio 逐轮形态 —— 判断 kl≈0.001 是
「PPO 正常收敛」还是「策略根本没动」。

### 这两个解释的区别

* **正常收敛**：entropy 从初始值稳步下降并收敛，kl 小是因为分布已经稳了。
* **策略没动**：entropy 几乎不动、ratio 一直贴 1、kl 一直很小
  ⟹ 45 步 × 30 轮也没把策略挪出多远，问题在有效步长而不在超参。

用法（服务器上）：/opt/qkd/venv/bin/python /tmp/check_entropy_traj.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
RUNS = ["ent01_s42", "ent01_s43", "base_s42", "vcoef1_s42"]
FIELDS = ["kl", "entropy", "ratio", "actor_grad", "V_std", "corr", "success_rate"]


def rows(run: str) -> list[dict]:
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(o.get("update"), int):
            out.append(o)
    return out


for run in RUNS:
    rs = rows(run)
    if not rs:
        print(f"  {run}: 无数据")
        continue
    print("=" * 78)
    print(f"{run}   （{len(rs)} 轮）")
    print("=" * 78)
    hdr = f"  {'u':>4}" + "".join(f"{f:>13}" for f in FIELDS)
    print(hdr)
    sel = rs[:4] + rs[len(rs) // 2: len(rs) // 2 + 1] + rs[-3:]
    seen = set()
    for o in sel:
        if o["update"] in seen:
            continue
        seen.add(o["update"])
        line = f"  {o['update']:>4}"
        for f in FIELDS:
            v = o.get(f)
            line += f"{v:>13.5f}" if isinstance(v, (int, float)) else f"{'—':>13}"
        print(line)
    print()
    for f in ("entropy", "kl", "ratio"):
        vals = [o[f] for o in rs if isinstance(o.get(f), (int, float))]
        if vals:
            print(f"    {f:<10} 首 {vals[0]:.5f}  末 {vals[-1]:.5f}  "
                  f"中位 {statistics.median(vals):.5f}  "
                  f"范围 [{min(vals):.5f}, {max(vals):.5f}]")
    print()

print("=" * 78)
print("判读提示")
print("=" * 78)
print("  · entropy **稳步下降并收敛** ⟹ kl 小是收敛，PPO 正常。")
print("  · entropy **几乎不动**（范围很窄、首末接近）⟹ 策略没被推动，")
print("    再调梯度侧超参（lr / epochs / minibatch）也难有量级改变。")
print("  · ratio 一直贴 1 ⟹ 与「clip 从未激活」一致。")
print("=" * 78)
