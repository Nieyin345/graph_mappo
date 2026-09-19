#!/usr/bin/env python
"""上一条结论的稳健性检验：`ent01` 超过专家，会不会是**一次挑轮次的产物**？

初算结果（跨训练种子，n=3；临界值见下）：

| 轮 | Δ | SE | t | p(df=2) | 按 4.303 判 |
|---|---|---|---|---|---|
| u15 | +0.0047 | 0.0050 | +0.94 | 0.448 | 打平 |
| u20 | +0.0031 | 0.0180 | +0.17 | 0.879 | 打平 |
| u25 | +0.0192 | 0.0047 | +4.09 | 0.055 | **未达** |
| u30 | +0.0205 | 0.0068 | +3.03 | 0.094 | **未达** |

u20 的 SD(Δ_s)=0.0312 明显大于邻居（0.0087 / 0.0082 / 0.0118）——**一个种子里
有一次大幅摆动**（s42 在 u20 是 −0.0328，而它在 u25 是 +0.0098）。

所以必须查：**这是"训练到 u25 才真的超过"，还是"u20 那次是离群点、而结论
对轮次选择敏感"？** 若后者，那么"超过专家"就是挑点挑出来的。

四件事：
  1. **逐种子看尾部形状** —— 三个种子是不是都在 u25/u30 抬高？
  2. **去掉离群的 s42** 会怎样（会不会结论全靠两个种子）？
  3. **平台区整体**（u25+u30 合并）而不是单点；
  4. **反事实**：专家整体抬高多少，结论就翻 —— 量余量。**不是**量"上界"。

用法（服务器上）：python scripts/diag/ent01_robustness.py

⚠ 2026-09-19 订正：本脚本的 t 值**不能**对着 2 读。n=3 → df=2 → 临界值 **4.303**。
  最权威的阈值与功效计算在 `scripts/diag/ent01_smallsample.py` 与
  `scripts/diag/rho_measure.py`；本脚本只用来看**方向与形状**。
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = OUT / "eval" / "expert_seeds100_240.json"
SEEDS = [42, 43, 44]


def expert() -> tuple[list[int], list[float]]:
    d = json.loads(EXPERT.read_text(encoding="utf-8"))
    return list(d["seeds"]), [float(x) for x in d["success"]]


def points(run: str) -> dict[int, list[float]]:
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        return {}
    out: dict[int, list[float]] = {}
    last = None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "update" in o:
            last = o["update"]
        ev = o.get("eval_validation")
        if isinstance(ev, dict) and ev.get("per_seed_success") and last is not None:
            out[last] = [float(x) for x in ev["per_seed_success"]]
    return out


seeds, base = expert()
pts = {s: points(f"ent01_s{s}") for s in SEEDS}

print("=== 1. 逐种子尾部形状（每行是该种子对专家的配对差）===")
print(f"  {'run':<12}" + "".join(f"{f'u{u}':>9}" for u in (15, 20, 25, 30)))
for s in SEEDS:
    row = f"  ent01_s{s:<6}"
    for u in (15, 20, 25, 30):
        ps = pts[s].get(u)
        if ps is None or len(ps) != len(seeds):
            row += f"{'—':>9}"
            continue
        d = statistics.mean([ps[i] - base[i] for i in range(len(seeds))])
        row += f"{d:>+9.4f}"
    print(row)
print()
print("  注意 s42 在 u20 = −0.033 而 u25 = +0.010：**一个种子里的一次大幅摆动**。")
print()

# ---------- 2. 去掉离群种子 ----------
print("=== 2. 去掉 s42（u20 的离群者）后 ===")
for u in (15, 20, 25, 30):
    ds = []
    for s in (43, 44):
        ps = pts[s].get(u)
        if ps is None or len(ps) != len(seeds):
            continue
        ds.append(statistics.mean([ps[i] - base[i] for i in range(len(seeds))]))
    if len(ds) < 2:
        continue
    m = statistics.mean(ds)
    sd = statistics.stdev(ds)
    se = sd / math.sqrt(len(ds))
    print(f"  u{u}: Δ={m:+.4f}  SD={sd:.4f}  SE={se:.4f}  t={m/se:+.2f}  (n=2)")
print("  → n=2 的 SE 本身很不可靠，这里只看**方向是否一致**，不看 t。")
print()

# ---------- 3. 平台区合并（u25+u30）----------
print("=== 3. 平台区合并（u25 与 u30 各自的配对差先平均，再跨种子）===")
per_seed = []
for s in SEEDS:
    us = []
    for u in (25, 30):
        ps = pts[s].get(u)
        if ps is None or len(ps) != len(seeds):
            continue
        us.append(statistics.mean([ps[i] - base[i] for i in range(len(seeds))]))
    if us:
        per_seed.append(statistics.mean(us))
        print(f"  ent01_s{s}: u25/u30 均值差 = {statistics.mean(us):+.4f}")
m = statistics.mean(per_seed)
sd = statistics.stdev(per_seed)
se = sd / math.sqrt(len(per_seed))
print(f"  Δ = {m:+.4f}  SD = {sd:.4f}  SE = {se:.4f}  **t = {m/se:+.2f}** "
      f"(n={len(per_seed)})")
print(f"  ⚠ n={len(per_seed)} → df={len(per_seed)-1}，临界值 **4.303**，不是 2。"
      f"t={m/se:+.2f} {'达到' if abs(m/se) > 4.303 else '**未达到**'}。")
print()

# ---------- 4. 反事实：专家上界 ----------
print("=== 4. 结论离边界多远：把专家整体抬高看还显不显著 ===")
sd_base = statistics.stdev(base)
print(f"  专家均值 {statistics.mean(base):.4f}  SD {sd_base:.4f}  (n={len(base)})")
print("  （抬高专家 = 加一个常数到所有逐种子差上，SE 不变、Δ 平移）")
for shift in (0.0, 0.005, 0.010, 0.015, 0.020):
    ds = []
    for s in SEEDS:
        us = []
        for u in (25, 30):
            ps = pts[s].get(u)
            if ps is None:
                continue
            us.append(statistics.mean([ps[i] - base[i] for i in range(len(seeds))]))
        if us:
            ds.append(statistics.mean(us))
    mm = statistics.mean(ds) - shift
    ssd = statistics.stdev(ds)
    sse = ssd / math.sqrt(len(ds))
    t = mm / sse if sse else float("nan")
    print(f"  专家 +{shift:.3f} → Δ={mm:+.4f}  t={t:+.2f}  "
          f"{'仍显著' if abs(t) > 4.303 else '**不再显著**'}")
print()

print("=== 结论 ===")
print("  判读要点（**用 df=2 的临界值 4.303，不要用 2**）：")
print("   - 若 u25 与 u30 **两个点都**过 4.303、且平台区合并后也过，则结论稳健；")
print("   - 若只有单点接近或无一点过，则只能说'方向一致、量级 +0.02，未达显著'。")
print("   - 第 4 节的反事实给的是**余量**：专家要再好 0.010 结论就翻。")
print("     这本身就是「效应很小」的证据——**留着当保留意见，别删**。")
