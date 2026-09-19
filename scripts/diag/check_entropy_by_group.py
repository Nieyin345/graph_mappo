#!/usr/bin/env python
"""核一个说法：`entropy_coef=0.001` 时「探索迅速坍缩」—— 这是真的吗？

### 为什么要核

`configs/train_ent01.yaml` 是**全项目唯一超过专家的单参数改动**，
它给自己的机理写的是（第 18-21 行，逐字）：

> 机理：本项目训练侧优势早已饱和（mean_abs_advantage 1.08 → 0.32），
> 策略在训练分布上已近确定性。**熵系数过小（0.001）时探索迅速坍缩**，
> rollout 分布变窄，到验证 regime 就吃不到新的状态覆盖；0.01 让策略在
> 验证分布上仍有熵，这与 r8 的发现（涨的是验证侧、不是训练侧）方向一致。

「迅速坍缩」是一个**可核**的断言：看 entropy 轨迹就行。
如果 0.001 的臂熵是**下降**的，这句成立；如果是**上升**的，这句就是错的
（那机理要改写，虽然结论可能仍然成立）。

用法（服务器上）：/opt/qkd/venv/bin/python /tmp/check_entropy_by_group.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def traj(run: str) -> list[tuple[int, float]]:
    p = OUT / run / "metrics.jsonl"
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        u, e = o.get("update"), o.get("entropy")
        if isinstance(u, int) and isinstance(e, (int, float)):
            out.append((u, float(e)))
    return out


def group_of(name: str) -> str | None:
    """按 run 名判断它属于哪一组 entropy_coef。

    ⚠ 这是**按名字猜**，不是从 resolved_config 读。所以脚本会把
    猜出来的组成员连同它的首末熵一起打印，让人肉眼复核。
    带 ent01 / ent 的 -> 0.01；vcoef1 已知是 0.001；
    r6_base / r7_base / r8_base / base -> 0.001。
    """
    n = name
    if "ent01_off" in n:
        return "0.001(off)"
    if "ent01" in n or n.startswith("ent"):
        return "0.01"
    if n.startswith(("vcoef1", "base", "r6_base", "r7_base", "r8_base")):
        return "0.001"
    return None


runs = sorted(d.name for d in OUT.iterdir() if d.is_dir())
groups: dict[str, list[tuple[str, list]]] = {}
for r in runs:
    g = group_of(r)
    t = traj(r)
    if g and len(t) >= 10:
        groups.setdefault(g, []).append((r, t))

print("=" * 78)
print("按 entropy_coef 分组看熵轨迹（首 → 末，Δ）")
print("=" * 78)
summary: dict[str, list[float]] = {}
for g in sorted(groups):
    print(f"\n  ── entropy_coef = {g} ──  （{len(groups[g])} 个 run，≥10 轮）")
    deltas, firsts, lasts = [], [], []
    for r, t in groups[g]:
        e0, e1 = t[0][1], t[-1][1]
        d = e1 - e0
        deltas.append(d)
        firsts.append(e0)
        lasts.append(e1)
        mark = "↑" if d > 0.05 else ("↓" if d < -0.05 else "→")
        print(f"    {r:<26} u{t[0][0]:>2} {e0:.3f}  →  u{t[-1][0]:>2} {e1:.3f}"
              f"   Δ {d:+.3f} {mark}")
    summary[g] = deltas
    print(f"    {'':<26} 首均值 {statistics.mean(firsts):.3f}   "
          f"末均值 {statistics.mean(lasts):.3f}   "
          f"Δ均值 **{statistics.mean(deltas):+.3f}**")

print()
print("=" * 78)
print("判读")
print("=" * 78)
if len(summary) >= 2:
    for g in sorted(summary):
        ds = summary[g]
        up = sum(1 for d in ds if d > 0.05)
        dn = sum(1 for d in ds if d < -0.05)
        print(f"  entropy_coef={g:<12} n={len(ds):<3} "
              f"上升 {up}  下降 {dn}  Δ均值 {statistics.mean(ds):+.3f}")
    print()
    if all(statistics.mean(d) > 0 for d in summary.values()):
        print("  ★ **两组熵都在上升** ⟹ 「0.001 时探索迅速坍缩」**不成立**。")
        print("    真实的区别是**上升幅度/平台高度**，不是「坍缩 vs 不坍缩」。")
        print("    ⚠ `configs/train_ent01.yaml` 第 18-21 行的机理叙述需要改写")
        print("      （结论——0.01 更好——可能仍然成立，但理由不是「坍缩」）。")
    else:
        print("  有组是下降的，逐行看上面的表。")
else:
    print("  分组数据不足。")
print("=" * 78)
