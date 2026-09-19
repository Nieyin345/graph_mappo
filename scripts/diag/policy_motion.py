# -*- coding: utf-8 -*-
"""策略到底动了多少？—— 挖 kl / entropy / 梯度范数的轨迹。

动机：ent01_s45 第 1 轮 kl=0.0021，小得可疑。PPO 的 clip 通常在 KL 到
0.01~0.03 才起作用；若整条曲线都停在 0.002，说明**策略几乎没动**——
那"训练改变不了什么"就不是超参问题，而是**有效步长被卡住了**。
这能解释为什么 RL 与 BC 起点、与专家的差距都不大。

entropy 同样关键：entropy_coef=0.01 这条线（本项目的核心结论）是否真的
把熵撑住了？若熵仍在快速塌缩，说明 0.01 还不够；若熵在涨，说明探索过量。

全部数据都在已有的 /tmp/*.log 里，**不需要新实验**。
"""
from __future__ import annotations

import re
import statistics as st
from pathlib import Path

KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(?[^=)]*\)?=(-?[\d.eE+]+)")


def parse(name):
    p = Path("/tmp") / f"{name}.log"
    if not p.exists():
        return None
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if "update=" not in line or "actor_loss" not in line:
            continue
        d = {}
        for k in ("kl", "entropy", "actor_loss", "critic_loss",
                  "actor_grad", "critic_grad", "reward", "success_rate"):
            m = re.search(rf"(?:^|\s){k}=(-?[\d.eE+]+)", line)
            if m:
                d[k] = float(m.group(1))
        if d:
            rows.append(d)
    return rows or None


FAMS = {
    "ent01": ["ent01_s42", "ent01_s43", "ent01_s44"],
    "base(r6)": ["r6_base", "r6_base_lr1e4", "r6_g999"],
    "ent0.001(vcoef1)": ["vcoef1_s42", "vcoef1_s43", "vcoef1_s44"],
}
KS = ("kl", "entropy", "success_rate", "actor_grad", "critic_grad")

for fam, names in FAMS.items():
    print("=" * 100)
    print(f"{fam}   （每格 = 多个种子的均值，括号内为逐种子 SD）")
    print("=" * 100)
    series = {k: [] for k in KS}
    n = 0
    for name in names:
        r = parse(name)
        if not r:
            continue
        n += 1
        for k in KS:
            series[k].append([x.get(k) for x in r])
    if n == 0:
        print("  （无数据）\n")
        continue
    L = min(len(s) for s in series["kl"])
    print(f"  种子数 {n}，共同长度 {L} 轮")
    hdr = "  " + f"{'轮':>4}" + "".join(f"{k:>16}" for k in KS)
    print(hdr)
    for i in list(range(0, min(L, 6))) + [9, 14, 19, 24, 29]:
        if i >= L:
            continue
        cells = []
        for k in KS:
            vals = [s[i] for s in series[k] if s[i] is not None]
            if not vals:
                cells.append(f"{'—':>16}")
                continue
            m = st.mean(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{m:>10.4f}({sd:.3f})")
        print(f"  {i+1:>4}" + "".join(cells))
    print()

print("=" * 100)
print("判读要点")
print("=" * 100)
print("  · PPO 的 clip 一般设 0.2，对应 KL 在 0.01~0.03 量级才'吃满'。")
print("    若 KL 长期 ≪0.01 → **有效步长被卡住**，加 lr / 加 epochs 才有意义。")
print("  · entropy 若单调下坠且不回头 → 探索在枯竭，entropy_coef 需要更大。")
print("  · 注意：vcoef1 是 ent=0.001 的旧基线，与 ent01 比熵的**水平**才有意义。")
