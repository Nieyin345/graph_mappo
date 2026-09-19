#!/usr/bin/env python
"""判读 entropy_coef 0.03 vs 0.01：本项目唯一确认有效的方向，再推一档是
**单调**（还能更高）还是**已过峰**（0.01 附近就是最优）。

### 为什么这一档值得跑

`ent=0.01` 是全项目唯一超过专家的单参数改动（合并 n=5，p=0.0074）。
即：**本项目唯一确认有效的方向就是"把熵推大"**。而 0.01 是一次跳 10 倍
跳出来的（0.001 → 0.01），**中间和上面都没测过** —— 不知单调还是过峰。

### 配对方式

`ent03_s42/43/44`（entropy_coef=0.03）对照**已有的** `ent01_s42/43/44`
（=0.01）。配置链、BC 起点、种子、轮数、**线程数（都是 4）**全部相同，
唯一差别是 entropy_coef。

判据：逐种子配对差 Δ_s = sr(0.03) − sr(0.01)，单样本 t，df=2，
临界值 **4.303**（不是 2 —— 本项目在这里栽过）。

  |t| ≥ 4.303 且 Δ>0 ⟹ **单调**，还能再往上推
  |t| ≥ 4.303 且 Δ<0 ⟹ **已过峰**，0.01 附近是最优（负结论也值钱）
  |t| < 4.303        ⟹ 测不出，只能说"这一档没看出差别"

用法（服务器上）：/opt/qkd/venv/bin/python scripts/diag/ent03_verdict.py
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
    pts = val_points(run)
    us = [u for u in PLATEAU if u in pts]
    if not us:
        return None
    n = len(pts[us[0]])
    if any(len(pts[u]) != n for u in us):
        return None
    return [statistics.mean(pts[u][i] for u in us) for i in range(n)]


print("=" * 78)
print("entropy_coef 0.03 vs 0.01（同链/同 BC/同种子/同轮数/同 4 线程）")
print("=" * 78)

a1, a3 = {}, {}
for s in SEEDS:
    a1[s] = plateau_mean(f"ent01_s{s}")   # 0.01（已有）
    a3[s] = plateau_mean(f"ent03_s{s}")   # 0.03（新跑）

missing = [s for s in SEEDS if a1[s] is None or a3[s] is None]
if missing:
    print(f"  数据不全：缺 {missing}")
    print(f"    ent01_s{{42,43,44}} (0.01): "
          f"{[s for s in SEEDS if a1[s] is None] or '齐'}")
    print(f"    ent03_s{{42,43,44}} (0.03): "
          f"{[s for s in SEEDS if a3[s] is None] or '齐'}")
    raise SystemExit(1)

diffs = []
print()
print(f"  {'种子':>6}{'0.01 平台均值':>16}{'0.03 平台均值':>16}{'Δ(0.03−0.01)':>16}")
for s in SEEDS:
    m1, m3 = statistics.mean(a1[s]), statistics.mean(a3[s])
    d = m3 - m1
    diffs.append(d)
    print(f"  {s:>6}{m1:>16.4f}{m3:>16.4f}{d:>+16.4f}")

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
    if m > 0:
        print(f"  ✓ |t| = {abs(t):.3f} ≥ {T_CRIT_DF2} 且 Δ>0 ⟹ **单调，还能再往上推**")
        print(f"    建议下一档测 entropy_coef 0.1（再跳一档），并把 0.03/0.1 一起")
        print(f"    放到验证 regime 上复测。")
    else:
        print(f"  ✗ |t| = {abs(t):.3f} ≥ {T_CRIT_DF2} 且 Δ<0 ⟹ **已过峰**")
        print(f"    0.01 附近就是最优 —— **这条线可以收了**（负结论也值钱）。")
else:
    print(f"  ○ |t| = {abs(t):.3f} < {T_CRIT_DF2} ⟹ **以 n=3 的分辨率测不出**")
    print(f"    分辨不出「单调」还是「过峰」。")
    print()
    print(f"  ⚠ 要说准确：是「以 n=3（df=2，临界 {T_CRIT_DF2}）测不出」，")
    print(f"    **不是**「两者相同」。本项目 n=3 的分辨率约 0.035。")
    print(f"    若要定论，需加种子（每个种子约 30 分钟，4 线程）。")

print()
print("  参考：专家 = ", end="")
if EXPERT.exists():
    ex = json.loads(EXPERT.read_text(encoding="utf-8"))
    print(f"{statistics.mean(float(x) for x in ex['success']):.4f}")
else:
    print("（缺 expert_seeds100_240.json）")
print("=" * 78)
