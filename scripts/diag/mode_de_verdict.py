#!/usr/bin/env python
"""判读 `model.mode: mixed → demand_edge` 这一刀。

### 这一刀在问什么

见 `docs/当前状态与优化方向.md` §2 P1。`graph_mappo.py:254` 那个
`if self.fuse_physical_to_node and n_phys > 0:` 是**布尔开关**，只决定
「物理边要不要往节点上聚合」：

  · `mixed`（现用）：节点聚合同时吃**物理边**与**需求边**两种消息
  · `demand_edge`：节点**只听需求边**，物理链路信息不再直接进节点，
    只能经由边嵌入间接传递

**它不改任何权重形状** ⟹ BC 暖启动照常加载 ⟹ 可与 ent01 直接配对。
这是全项目**唯一能便宜检验结构假设的旋钮**（§1.3 的「信息缺失 vs 容量不足」）。

### 配对

  `mode_de_s42/43/44`（mode=demand_edge）vs `ent01_t8_s42/43/44`（mode=mixed）

两者配置链相同（`rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml`）、
BC 起点相同、种子相同、轮数 30、**线程数都是 8** —— 唯一差别是 `model.mode`。

⚠ **硬件记账**：对照臂 `ent01_t8_*` 跑在**旧节点**（32 核/125 GB，EPYC 7402P），
实验臂跑在**新节点**（128 vCPU/250 GB，EPYC 7543）。按本项目已测结论，
决定结果的是 `OMP_NUM_THREADS`（两臂都是 8）而非机器速度，所以这个配对成立；
但这一条是**假定**，不是实测，故记账在此。

### 判据（跑之前写死）

  平台窗口 u25/u30，Δ_s = sr(demand_edge_s) − sr(mixed_s)
  单样本 t，df=2，双侧临界值 **4.303**

记忆点：`mode=demand_edge` 使节点**丧失一路输入**，若"信息缺失"假设成立，
**预期为负**（变差）。但**无论符号**，|t| ≥ 4.303 就是结论 —— 它说明
物理边消息**确实被用到了**，这本身就是关于结构的信息。

|t| < 4.303 ⟹ 以 n=3 分辨率（~0.035）**测不出** ⟹ 必须报「测不出」，
**不能**报「没影响」。

用法（服务器上）：/opt/qkd/venv/bin/python scripts/diag/mode_de_verdict.py
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
SEEDS = (42, 43, 44)
PLATEAU = (25, 30)          # 与 ent01 / t8 判读一致
EXPERT = OUT / "eval" / "expert_seeds100_240.json"
T_CRIT_DF2 = 4.303          # ★ df=2 的双侧 95% 临界值（不是 2，本项目栽过）
CTRL_FMT = "ent01_t8_s{}"   # mixed，8 线程（同制式对照）
EXP_FMT = "mode_de_s{}"     # demand_edge，8 线程


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
    pts = val_points(run)
    us = [u for u in PLATEAU if u in pts]
    if not us:
        return None
    n = len(pts[us[0]])
    if any(len(pts[u]) != n for u in us):
        return None
    return [statistics.mean(pts[u][i] for u in us) for i in range(n)]


print("=" * 78)
print("mode 判读：demand_edge vs mixed（同配置链/BC/种子/轮数/8 线程）")
print("=" * 78)

ctrl, exp = {}, {}
for s in SEEDS:
    ctrl[s] = plateau_mean(CTRL_FMT.format(s))
    exp[s] = plateau_mean(EXP_FMT.format(s))

missing = [s for s in SEEDS if ctrl[s] is None or exp[s] is None]
if missing:
    print(f"  数据不全：缺 {missing}")
    print(f"    {CTRL_FMT.format('{42,43,44}')} : "
          f"{[s for s in SEEDS if ctrl[s] is None] or '齐'}")
    print(f"    {EXP_FMT.format('{42,43,44}')}  : "
          f"{[s for s in SEEDS if exp[s] is None] or '齐'}")
    raise SystemExit(1)

diffs = []
print()
print(f"  {'种子':>6}{'mixed(ent01_t8)':>18}{'demand_edge':>16}{'Δ(de−mixed)':>14}")
for s in SEEDS:
    m_c, m_e = statistics.mean(ctrl[s]), statistics.mean(exp[s])
    d = m_e - m_c
    diffs.append(d)
    print(f"  {s:>6}{m_c:>18.4f}{m_e:>16.4f}{d:>+14.4f}")

m = statistics.mean(diffs)
sd = statistics.stdev(diffs)
se = sd / math.sqrt(len(diffs))
t = m / se if se > 0 else float("inf")
df = len(diffs) - 1

print()
print(f"  Δ = {m:+.4f}   SD(Δ_s) = {sd:.4f}   SE = {se:.4f}")
print(f"  t = {t:+.3f}  (df={df}, 双侧临界值 **{T_CRIT_DF2}**)")

# p 值现算（不手抄）—— 本项目有两次"手抄数字悄悄过期"的教训
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
    direction = "变差" if m < 0 else "变好"
    print(f"  ✗ |t| = {abs(t):.3f} ≥ {T_CRIT_DF2} ⟹ **物理边消息确实被用到了**")
    print(f"    Δ = {m:+.4f}（demand_edge 相对 mixed {direction}）")
    if m < 0:
        print("    ⟹ 支持「节点需要物理边输入」—— 即 §1.3 的**信息**一侧，")
        print("      而不是「容量不够」。这是可据以改结构的第一个硬证据。")
    else:
        print(f"    ⟹ 反直觉：拿掉一路输入反而更好。要先查是不是")
        print(f"      `fuse_physical_to_node` 引入的是**噪声**而非信息。")
else:
    print(f"  ✓ |t| = {abs(t):.3f} < {T_CRIT_DF2} ⟹ **以 n=3 的分辨率测不出差异**")
    print(f"    必须报「测不出」，**不能**报「没影响」：")
    print(f"      n=3（df=2）的分辨率约 **0.035**，比这小的效应测不到。")
    print(f"    要把它变成结论，需加训练种子（每种子约 30 分钟）。")

print()
print("  参考：专家 = ", end="")
if EXPERT.exists():
    ex = json.loads(EXPERT.read_text(encoding="utf-8"))
    print(f"{statistics.mean(float(x) for x in ex['success']):.4f}")
else:
    print("（缺 expert_seeds100_240.json）")
print("=" * 78)
