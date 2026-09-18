#!/usr/bin/env python
"""快路径（候选查找改预计算表）在**真实训练**里到底省了多少？

背景：`8c86f9c` 把 `_matching_log_prob_entropy_arrays` 的 79,451 次
`np.flatnonzero` 换成预计算查表，探针测得 **1.183x**（同负载交替、组内极差 <0.1%）。
但那是在 **1440 步剖面**上测的，**从未在真实训练里量过**。

为什么现在能量：ent01 三条臂与它们的 u30→u50 延长臂是**同一份配置、同一份起点**
（延长臂从 ent01 的 u30 checkpoint 恢复），唯一差别是**跑的时刻**——
而快路径正是在两者之间同步上线的。于是 `update_s` 的差就是它的真实收益。

**但这条比较有一半是跨时间的**，所以必须把两件事分开：
  - `rollout_s` 应当**完全不受**快路径影响（它走的是 rollout 路径，不是更新路径的
    数组函数）→ 拿它当**负载/时间的对照**。若 rollout_s 没变而 update_s 变了，
    那变化就更可能来自代码而不是机器。
  - `update_s` 是待测项。
若 rollout_s 也同幅度变化，说明是机器整体快了/慢了，代码结论不成立。

用法（服务器上）：python /tmp/probe_fastpath_real.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def rows(run):
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return [r for r in out if "update" in r]


def stat(run, lo=None, hi=None):
    rs = rows(run)
    if lo is not None:
        rs = [r for r in rs if r["update"] >= lo]
    if hi is not None:
        rs = [r for r in rs if r["update"] <= hi]
    if not rs:
        return None
    def med(key):
        v = sorted(r[key] for r in rs if isinstance(r.get(key), (int, float)))
        return v[len(v) // 2] if v else float("nan")
    return {
        "n": len(rs),
        "rollout": med("rollout_s"),
        "update": med("update_s"),
        "elapsed": med("elapsed_s"),
    }


# 延长臂从 ent01 的 u30 恢复 → u31..u50 是快路径之后的；ent01 的 u1..u30 是之前的。
# 两侧都取 **u31-u50 vs u21-u30** 之外的窗口不可行（ent01 只到 30），
# 所以改用"延长臂的 u31-u50"对"ent01 的 u21-u30"——
# 后者是 ent01 的末期，机器状态与启动初期不同，所以 rollout_s 的对照是必需的。
PAIRS = [
    ("ent01_s42", "ent01_s42_u30to50"),
    ("ent01_s43", "ent01_s43_u30to50"),
    ("ent01_s44", "ent01_s44_u30to50"),
]

print("=== 延长臂 vs 原臂（原臂 u21-30，延长臂 u41-50）===")
print(f"  {'run':<24}{'rollout_s':>11}{'update_s':>11}{'elapsed_s':>11}   (中位数)")
print()
agg = {"rb": [], "ub": [], "ra": [], "ua": []}
for base, ext in PAIRS:
    b = stat(base, 21, 30)
    a = stat(ext, 41, 50)
    if not b or not a:
        print(f"  {base}: 数据不足")
        continue
    print(f"  [原臂] {base:<17}{b['rollout']:>11.1f}{b['update']:>11.1f}{b['elapsed']:>11.1f}")
    print(f"  [延臂] {ext:<17}{a['rollout']:>11.1f}{a['update']:>11.1f}{a['elapsed']:>11.1f}")
    dr = a["rollout"] / b["rollout"]
    du = a["update"] / b["update"]
    print(f"         比值 延/原       rollout {dr:.3f}x   update {du:.3f}x")
    print()
    agg["rb"].append(b["rollout"]); agg["ub"].append(b["update"])
    agg["ra"].append(a["rollout"]); agg["ua"].append(a["update"])

if agg["ub"]:
    n = len(agg["ub"])
    mr = sum(agg["ra"]) / n, sum(agg["rb"]) / n
    mu = sum(agg["ua"]) / n, sum(agg["ub"]) / n
    print("=== 汇总（三种子平均）===")
    print(f"  rollout_s  {mr[1]:.1f} → {mr[0]:.1f}   比值 {mr[0]/mr[1]:.3f}x")
    print(f"  update_s   {mu[1]:.1f} → {mu[0]:.1f}   比值 {mu[0]/mu[1]:.3f}x")
    print()
    # 判据：rollout 若也有同向同幅的变化，则无法把 update 的变化归给代码
    r_ratio = mr[0] / mr[1]
    u_ratio = mu[0] / mu[1]
    print("=== 判读 ===")
    print(f"  对照项 rollout 比值 {r_ratio:.3f}x（应 ≈1.00，快路径不碰 rollout 路径）")
    if abs(r_ratio - 1.0) > 0.05:
        print(f"  ⚠ rollout 也变了 {(r_ratio-1)*100:+.1f}% → 机器整体快了/慢了，")
        print("     update 的变化要先扣掉这一项，不能直接读成代码收益。")
    # 扣掉负载漂移后的净效应
    net = u_ratio / r_ratio
    print(f"  扣掉负载漂移后的 update 净比值 = {u_ratio:.3f} / {r_ratio:.3f} = {net:.3f}x")
    # **预期的正确算法**（初稿这里算错了，见下）：
    # 快路径是 1.183x，但它只作用于 `_matching_log_prob_entropy_arrays`，
    # 而那只是**一次 update 的 10%**（2.33s / 23.4s，剖面实测）。
    # 所以对 update 整体的预期收益 = 1.183x 作用在 10% 上：
    #     新 update = 90% + 10%/1.183 = 0.90 + 0.0845 = 0.9845
    #     → update 比值预期 0.985，即**省 1.5%**
    # 初稿写的是"探针预期 1.183x，即省 15.5%" —— 把**函数级**的倍数
    # 直接当成了**整体**的倍数。这是一个量纲错误：18% 的加速作用在 10% 的
    # 组成上是 1.5%，不是 18%。**函数有多快不等于训练有多快。**
    implied = 0.90 + 0.10 / 1.183
    print(f"  预期（剖面：该函数占 update 的 10%，函数本身快 1.183x）:")
    print(f"     0.90 + 0.10/1.183 = {implied:.4f} → 预期省 {(1-implied)*100:.1f}%")
    print(f"  实测净比值 {net:.3f} → 省 {(1-net)*100:.1f}%")
    print()
    # 分辨率：这个效应是否**大到能被这次比较测出来**？
    # 组内（同 run 内逐轮）update_s 的波动就是噪声尺度。
    import statistics
    sp = [stat(x, 21, 30)["update"] for x in ("ent01_s42", "ent01_s43", "ent01_s44")]
    # 用逐轮极差估计噪声（比 stdev 保守，因为是同一 run 内的不同轮）
    spreads = []
    for run in ("ent01_s42", "ent01_s43", "ent01_s44",
                "ent01_s42_u30to50", "ent01_s43_u30to50", "ent01_s44_u30to50"):
        v = sorted(r["update_s"] for r in rows(run) if isinstance(r.get("update_s"), (int, float)))
        if len(v) >= 6:
            edges = v[:3] + v[-3:]
            spreads.append((max(edges) - min(edges)) / (sum(edges) / len(edges)))
    if spreads:
        noise_pct = 100 * sum(spreads) / len(spreads)
        print(f"  逐轮 update_s 的组内相对波动 ≈ {noise_pct:.1f}%（6 个 run 的端部极差均值）")
        print(f"  待测效应 ≈ {(1-implied)*100:.1f}%")
        if noise_pct > (1 - implied) * 100:
            print(f"  → 组内噪声 {noise_pct:.1f}% **大于**待测效应 {(1-implied)*100:.1f}%：")
            print("     这次比较**测不出**它。结论只能是'与 1.5% 的预测不矛盾'，")
            print("     不能是'快路径有效'或'快路径无效'。")
        else:
            print("  → 噪声小于效应，可以判读")

print()
print("=== 反查：实际生效代码的时间线 ===")
pol = Path("/opt/qkd/graph_mappo/qkd_rl/rl/algos/policy.py")
import datetime
mt = datetime.datetime.fromtimestamp(pol.stat().st_mtime)
print(f"  policy.py mtime = {mt:%Y-%m-%d %H:%M:%S}")
print(f"  含 arc_pos_lookup: {pol.read_text(encoding='utf-8',errors='replace').count('arc_pos_lookup')} 处")
