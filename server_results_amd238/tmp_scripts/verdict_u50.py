#!/usr/bin/env python
"""u50 判决：按预注册判据（三种子分辨率 0.020），并把"是不是一个种子带偏的"查清。

判据（启动前写死在 docs/训练诊断记录.md）：
  u50-u30 >= +0.020           → 还没到平台
  |u50-u30| < 0.020           → 平台就在 u25 之后
  u50-u30 <= -0.020           → 后期退化

同时给出**逐种子**的 u50 vs u30 配对（同 15 个验证种子），
用以区分"三个种子都退"与"一个种子带偏均值"。
"""
import json
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
SEEDS = list(range(100, 115))


def series(run):
    """返回 {轮: (均值, 逐种子 list)}"""
    p = OUT / run / "metrics.jsonl"
    out = {}
    if not p.exists():
        return out
    last_u = None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "update" in r:
            last_u = r["update"]
        ev = r.get("eval_validation")
        if isinstance(ev, dict) and ev.get("per_seed_success"):
            out[last_u] = (float(ev["mean_success_rate"]),
                           [float(x) for x in ev["per_seed_success"]],
                           list(ev.get("seeds") or []))
    return out


base = {s: series(f"ent01_s{s}") for s in (42, 43, 44)}
ext = {s: series(f"ent01_s{s}_u30to50") for s in (42, 43, 44)}

print("=== 每股的完整验证序列 ===")
for s in (42, 43, 44):
    b = base[s]
    e = ext[s]
    line = " ".join(f"u{u}={b[u][0]:.4f}" for u in sorted(b))
    line2 = " ".join(f"u{u}={e[u][0]:.4f}" for u in sorted(e))
    print(f"  s{s}  {line}")
    print(f"       {line2}")

print()
print("=== 三种子均值（每个轮次）===")
allu = {}
for u in [25, 30, 35, 40, 45, 50]:
    vals = []
    for s in (42, 43, 44):
        src = base[s] if u <= 30 else ext[s]
        if u in src:
            vals.append(src[u][0])
    if len(vals) == 3:
        allu[u] = sum(vals) / 3
        print(f"  u{u}: {vals[0]:.4f} {vals[1]:.4f} {vals[2]:.4f}  →  均值 {allu[u]:.4f}")

m30, m50 = allu.get(30), allu.get(50)
print()
print("=== 预注册判据 ===")
print(f"  u30 均值 = {m30:.4f}")
print(f"  u50 均值 = {m50:.4f}")
d = m50 - m30
print(f"  Δ = u50-u30 = {d:+.4f}   分辨率阈值 0.020")
if d >= 0.020:
    verdict = "还没到平台 → 再延到 u70"
elif abs(d) < 0.020:
    verdict = "平台就在 u25 之后 → 转向'同预算更快达到'"
else:
    verdict = "后期退化 → 转向早停/正则"
print(f"  → {verdict}")

print()
print("=== 逐种子 u50 vs u30（同 15 个验证种子配对）===")
print(f"  {'种子':>4}{'u30':>9}{'u50':>9}{'Δ':>9}{'配对SE':>9}{'t':>8}{'胜/负':>8}")
for s in (42, 43, 44):
    a, b = base[s][30], ext[s][50]
    if a[2] != b[2]:
        print(f"  s{s}: 种子集不同，跳过")
        continue
    pa, pb = a[1], b[1]
    diffs = [pb[i] - pa[i] for i in range(len(pa))]
    md = sum(diffs) / len(diffs)
    se = statistics.stdev(diffs) / len(diffs) ** 0.5
    w = sum(1 for x in diffs if x > 0)
    print(f"  s{s:>3}{a[0]:>9.4f}{b[0]:>9.4f}{md:>+9.4f}{se:>9.4f}{md/se:>+8.2f}{w:>5}/{len(diffs)-w:<3}")

print()
print("=== 趋势拟合（u25..u50，6 点，每点 = 三种子均值）===")
xs = [25, 30, 35, 40, 45, 50]
ys = [allu[u] for u in xs]
n = len(xs)
mx, my = sum(xs) / n, sum(ys) / n
sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
sxx = sum((x - mx) ** 2 for x in xs)
slope = sxy / sxx
print(f"  斜率 = {slope:+.6f} /轮  → 每 5 轮 {slope*5:+.4f}，u25→u50 共 {slope*25:+.4f}")
print(f"  截距处（u25）预测 {my + slope*(25-mx):.4f}，u50 预测 {my + slope*(50-mx):.4f}")
print()
print("=== 观察：s44 的 u50 是不是离群 ===")
for s in (42, 43, 44):
    e = ext[s]
    seq = [e[u][0] for u in sorted(e)]
    print(f"  s{s}: " + " ".join(f"{v:.4f}" for v in seq)
          + f"   u45→u50 变化 {seq[-1]-seq[-2]:+.4f}")
