# -*- coding: utf-8 -*-
"""critic 到底行不行？—— 用**已经记下来的** corr(V,R) 直接看。

为什么这是关键：GAE 的优势估计 = 回报 − 价值预测。若 critic 的预测与真实回报
几乎不相关，优势里就基本全是噪声，PPO 的更新方向就不可信 —— 那么无论怎么调
actor 的超参都收效有限。而 corr(V,R) 每轮都在日志里，**不需要新实验**。

这条线索来自一个已确认的事实：从 BC 热启动时，**44/88 个 run 都触发了
critic value head 重置**（reward 段与 checkpoint 不同）。重置之后值头是随机的，
但 Adam 的一二阶矩**没有一起重置**（它们带着旧奖励尺度下的梯度量级，
β2=0.999 意味着衰减极慢）。这是否让 critic 迟迟学不起来，可以直接看曲线。
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
OUT = ROOT / "outputs"

KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(-?[\d.eE+]+)")


def rounds_from_log(name):
    """优先从 logs 里解析 UpdateStats（metrics.jsonl 字段名可能不同）。"""
    p = Path("/tmp") / f"{name}.log"
    if not p.exists():
        return None
    series = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if "UpdateStats" not in line and "update=" not in line:
            continue
        d = {k: float(v) for k, v in KV.findall(line)}
        if "corr(V,R)" in line:
            m = re.search(r"corr\(V,R\)=(-?[\d.eE+]+)", line)
            if m:
                d["corr"] = float(m.group(1))
        if "corr" in d or "critic_loss" in d:
            series.append(d)
    return series or None


fams = {
    "ent01": ["ent01_s42", "ent01_s43", "ent01_s44", "ent01_s45", "ent01_s46"],
    "base(r6)": ["r6_base", "r6_base_lr1e4", "r6_g999"],
}
resets = {}
for name in [n for f in fams.values() for n in f]:
    lp = Path("/tmp") / f"{name}.log"
    resets[name] = bool(lp.exists() and "re-initialized critic value head" in
                        lp.read_text(encoding="utf-8", errors="replace"))

print("=" * 96)
print("1. 各 run 的 corr(V,R) 轨迹（critic 预测与回报的相关系数）")
print("=" * 96)
print("  值头重置?  run             u1     u2     u3     u5    u10    u15    u20    u30")
for fam, names in fams.items():
    print(f"  --- {fam} ---")
    for name in names:
        s = rounds_from_log(name)
        if not s:
            continue
        vals = {}
        for i, d in enumerate(s, start=1):
            if "corr" in d:
                vals[i] = d["corr"]
        if not vals:
            continue
        ks = [1, 2, 3, 5, 10, 15, 20, 30]
        row = "".join(
            f"{vals[k]:>7.3f}" if k in vals else f"{'—':>7}"
            for k in ks
        )
        r = "重置" if resets[name] else "保留"
        print(f"  {r:<9} {name:<15}{row}")

print()
print("=" * 96)
print("2. critic_loss 与梯度范数（判据：值头重置后是否异常大）")
print("=" * 96)
print(f"  {'run':<16}{'cl_u1':>9}{'cl_u2':>9}{'cl_u5':>9}{'ag_u1':>9}{'cg_u1':>9}{'cg_u5':>9}")
for fam, names in fams.items():
    for name in names:
        s = rounds_from_log(name)
        if not s:
            continue
        def g(i, k):
            return s[i - 1].get(k) if len(s) >= i else None
        def f(x):
            return f"{x:>9.4f}" if isinstance(x, float) else f"{'—':>9}"
        print(f"  {name:<16}{f(g(1,'critic_loss'))}{f(g(2,'critic_loss'))}"
              f"{f(g(5,'critic_loss'))}{f(g(1,'actor_grad'))}{f(g(1,'critic_grad'))}"
              f"{f(g(5,'critic_grad'))}")

print()
print("=" * 96)
print("3. 值头重置分组：corr(V,R) 的平台均值")
print("=" * 96)
for grp in (True, False):
    allc, late = [], []
    for name, r in resets.items():
        if r != grp:
            continue
        s = rounds_from_log(name)
        if not s:
            continue
        c = [d["corr"] for d in s if "corr" in d]
        if c:
            allc += c
            late += c[len(c) // 2:]
    if allc:
        lab = "重置了值头" if grp else "保留值头"
        print(f"  {lab:<12} n={len(allc):>4} 点  corr 均值 {st.mean(allc):+.4f}"
              f"  后半程均值 {st.mean(late):+.4f}"
              f"  (SD {st.stdev(allc):.4f})")
