#!/usr/bin/env python
"""判读线程数实验：8 线程训练出的策略 vs 4 线程的，差多少。

### 为什么这个判据决定"能不能改"

`OMP_NUM_THREADS` **确定性地改变训练结果**（已实测：2 vs 4 线程差
**+0.0176，t=+7.68**，两次重复给出同一个数；而**评测不受影响**，
四种组合逐位相同 0.767955）。

所以"4→8 线程省 1.34x"这句话**不足以**支持改配置 —— 必须先回答
**改了以后结果会不会变**：

  · 差 ≈ 0  ⟹ 白捡 1.34x，全项目改 8（新臂都跑 8）
  · 差 ≈ 0.018 ⟹ 现有全部 arm 要重建基线，**不值得为 1.34x 重跑一切**

### 配对方式

`ent01_t8_s42/43/44`（**8 线程**）对照 `ent01_s42/43/44`（**4 线程**）。
配置链、BC 起点、种子、轮数全部相同，**唯一差别是线程数**。

判据：逐种子配对差 Δ_s = sr(8线程) − sr(4线程)，
**单样本 t，df=2，临界值 4.303**（不是 2 —— 本项目在这里栽过，
见 [[thresholds-and-transcribed-numbers]]）。

⚠ **记账**：对照臂是几小时前跑的，机器负载与现在不同。这一点无法完全排除，
但 `probe_thread_repro.sh` 当时已确认"背景负载改变结果"**不成立**
（4 线程那两次在 24 进程背景下仍与 2 线程逐位复现各自的值）——
真正的变量只有 `OMP_NUM_THREADS`。若结论落在临界值附近，要重跑一次对照。

用法（服务器上）：/opt/qkd/venv/bin/python scripts/diag/thread_sweep_verdict.py
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
SEEDS = (42, 43, 44)
PLATEAU = (25, 30)          # 与预注册一致的平台窗口
EXPERT = OUT / "eval" / "expert_seeds100_240.json"
T_CRIT_DF2 = 4.303          # ★ df=2 的双侧 95% 临界值


def val_points(run: str) -> dict[int, list[float]]:
    """{update: per_seed_success}"""
    p = OUT / run / "metrics.jsonl"
    out: dict[int, list[float]] = {}
    if not p.exists():
        return out
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


def plateau_mean(run: str):
    """平台窗口的逐种子均值。返回 list 或 None。"""
    pts = val_points(run)
    us = [u for u in PLATEAU if u in pts]
    if not us:
        return None
    n = len(pts[us[0]])
    if any(len(pts[u]) != n for u in us):
        return None
    return [statistics.mean(pts[u][i] for u in us) for i in range(n)]


print("=" * 78)
print("线程数实验判读：8 线程 vs 4 线程（同配置/同 BC/同种子/同轮数）")
print("=" * 78)

a4, a8 = {}, {}
for s in SEEDS:
    a4[s] = plateau_mean(f"ent01_s{s}")          # 4 线程（已存在）
    a8[s] = plateau_mean(f"ent01_t8_s{s}")       # 8 线程（新跑）

missing = [s for s in SEEDS if a4[s] is None or a8[s] is None]
if missing:
    print(f"  数据不全：缺 {missing}")
    print(f"    ent01_s{{42,43,44}}   : "
          f"{[s for s in SEEDS if a4[s] is None] or '齐'}")
    print(f"    ent01_t8_s{{42,43,44}}: "
          f"{[s for s in SEEDS if a8[s] is None] or '齐'}")
    raise SystemExit(1)

diffs = []
print()
print(f"  {'种子':>6}{'4线程平台均值':>16}{'8线程平台均值':>16}{'Δ(8−4)':>12}")
for s in SEEDS:
    m4, m8 = statistics.mean(a4[s]), statistics.mean(a8[s])
    d = m8 - m4
    diffs.append(d)
    print(f"  {s:>6}{m4:>16.4f}{m8:>16.4f}{d:>+12.4f}")

m = statistics.mean(diffs)
sd = statistics.stdev(diffs)
se = sd / math.sqrt(len(diffs))
t = m / se if se > 0 else float("inf")
df = len(diffs) - 1

print()
print(f"  Δ = {m:+.4f}   SD(Δ_s) = {sd:.4f}   SE = {se:.4f}")
print(f"  t = {t:+.3f}  (df={df}, 双侧临界值 **{T_CRIT_DF2}**)")

# p 值现算（不手抄）——本项目已有两次"手抄 p 值悄悄过期"的教训
try:
    from scipy import stats
    p = 2 * float(stats.t.sf(abs(t), df))
    src = "scipy"
except ImportError:
    p = float("nan")
    src = "无 scipy，p 未算（临界值判据仍有效）"
print(f"  p = {p:.4f}  ({src})")

print()
print("=" * 78)
print("判读（跑之前写死的）")
print("=" * 78)
if abs(t) >= T_CRIT_DF2:
    print(f"  ✗ |t| = {abs(t):.3f} ≥ {T_CRIT_DF2} ⟹ **线程数确实改变训练结果**")
    print(f"    现有全部 arm 都是 4 线程，改 8 需要重建基线。")
    print(f"    **不建议**为 1.34x 的提速重跑所有实验。")
    print(f"    可在**新开的**实验线上用 8 线程，但不可与历史 arm 直接比。")
else:
    print(f"  ✓ |t| = {abs(t):.3f} < {T_CRIT_DF2} ⟹ **以 n=3 的分辨率测不出差异**")
    print(f"    ⟹ 建议全项目线程数 4 → 8，白捡整轮 **1.34x**")
    print()
    print(f"  ⚠ 但要说准确：是「以 n=3（df=2，临界 {T_CRIT_DF2}）测不出」，")
    print(f"    **不是**「两者相同」。本项目 n=3 的分辨率约 0.035，")
    print(f"    而 2 vs 4 线程的已知差是 0.0176 —— **低于这个分辨率**。")
    print(f"    所以「测不出」与「确实有 ~0.018 的差」并不矛盾，")
    print(f"    只是这次的 n 不足以区分。若要把这个风险压小，")
    print(f"    需加种子（每个种子约 30 分钟）。")

print()
print("  参考：专家 = ", end="")
if EXPERT.exists():
    ex = json.loads(EXPERT.read_text(encoding="utf-8"))
    print(f"{statistics.mean(float(x) for x in ex['success']):.4f}")
else:
    print("（缺 expert_seeds100_240.json）")
print("=" * 78)
