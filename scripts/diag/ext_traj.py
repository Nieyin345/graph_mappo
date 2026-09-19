# -*- coding: utf-8 -*-
"""续训 u30→u50 的完整轨迹：是"平台"还是"熵涨坏了"？

A 节说曲线还在爬（+0.0174），B 节说真续跑 20 轮只有 +0.0013 且 u50 = −0.0331。
**外推失败**。现在要分辨 u50 那个 −0.0331 是：
  (a) 单点噪声（那就照端点陷阱处理，只看半段均值），还是
  (b) **熵持续上涨把策略推散**（那就与 entropy_coef=0.01 有关，是真机制）。

判据：把 entropy 与成功率放在同一条时间轴上。若 u50 处 entropy 明显高于 u40，
且训练侧成功率同步下滑 → (b)；若 entropy 平稳 → (a)。
"""
from __future__ import annotations

import re
import statistics as st
from pathlib import Path

KV = ("entropy", "success_rate", "kl", "actor_grad", "critic_grad", "reward")


def parse(name):
    p = Path("/tmp") / f"{name}.log"
    if not p.exists():
        return None
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if "update=" not in line or "actor_loss" not in line:
            continue
        d = {}
        for k in KV:
            m = re.search(rf"(?:^|\s){k}=(-?[\d.eE+]+)", line)
            if m:
                d[k] = float(m.group(1))
        if d:
            rows.append(d)
    return rows or None


for name in ("ent01_s44_u30to50", "ent01_s44"):
    r = parse(name)
    print("=" * 92)
    print(f"{name}")
    print("=" * 92)
    if not r:
        print("  无日志\n")
        continue
    print(f"  共 {len(r)} 轮")
    print(f"  {'轮':>4}{'entropy':>11}{'train_succ':>12}{'kl':>10}{'a_grad':>10}")
    for i, d in enumerate(r, start=1):
        if i <= 6 or i % 5 == 0 or i == len(r):
            print(f"  {i:>4}{d.get('entropy', float('nan')):>11.4f}"
                  f"{d.get('success_rate', float('nan')):>12.4f}"
                  f"{d.get('kl', float('nan')):>10.4f}"
                  f"{d.get('actor_grad', float('nan')):>10.4f}")
    e = [d["entropy"] for d in r if "entropy" in d]
    s = [d["success_rate"] for d in r if "success_rate" in d]
    if len(e) >= 4:
        print(f"\n  entropy:  首 {e[0]:.4f}  中 {e[len(e)//2]:.4f}  末 {e[-1]:.4f}"
              f"   (后段增量 {e[-1]-e[len(e)//2]:+.4f})")
        # 半段均值，避免端点
        print(f"  entropy 前半均值 {st.mean(e[:len(e)//2]):.4f}  "
              f"后半均值 {st.mean(e[len(e)//2:]):.4f}")
    if len(s) >= 4:
        print(f"  train_succ 前半均值 {st.mean(s[:len(s)//2]):.4f}  "
              f"后半均值 {st.mean(s[len(s)//2:]):.4f}"
              f"   后段增量 {st.mean(s[len(s)//2:]) - st.mean(s[:len(s)//2]):+.4f}")
    print()
