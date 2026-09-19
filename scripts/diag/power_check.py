#!/usr/bin/env python
"""在跑的复筛链，**3 个种子够不够分辨 +0.035**？

为什么现在算而不是等结果：链条要跑 7.8h。如果 3 个种子的分辨率不够，
那我现在就该加种子（还有时间），而不是 7.8h 后拿到一堆读不出的数。

### 口径

预注册判据是「Δ = 三种子在 u30 对 ent01 的配对差，≥+0.035 判有效」。
配对差的统计量是

    Δ = mean_s ( arm_s − ent01_s )        s ∈ {42, 43, 44}

它的标准误 = SD(逐种子差) / sqrt(3)。所以关键量是 **SD(逐种子差)**。

### 怎么估这个 SD（不猜，用实测数据）

SD(逐种子差) 取决于 arm 与 ent01 的**逐种子相关**：

    SD(diff)² = SD(arm)² + SD(ent01)² − 2·ρ·SD(arm)·SD(ent01)

- ρ = 1（完全相关）→ SD(diff) = 0，配对有无限分辨力
- ρ = 0（独立）    → SD(diff) = sqrt(2)·SD ≈ 1.41·SD

**ρ = 0 是保守上界**，这里就用它，再配上实测的 SD(ent01)。

另外做一个**经验校准**：找 outputs/ 里**同配置、不同种子**的成对 run，
直接量"同一配置两次跑出来的逐种子差"有多大 —— 那是零分布的真实尺度。

用法（服务器上）：python /tmp/power_check.py
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
SEEDS = [42, 43, 44]
DELTA = 0.035


def val_at_u(run: str, want_u: int) -> float | None:
    """取 run 在轮次 want_u 的验证成功率（验证点落在 u5/u10/…/u30）。"""
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        return None
    last = None
    for line in p.read_text(encoding="utf-8").splitlines():
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
        if isinstance(ev, dict) and ev.get("per_seed_success") and last == want_u:
            return float(ev["mean_success_rate"])
    return None


print("=== 1. 基线 ent01 三种子在 u30 的实测散布 ===")
base = {}
for s in SEEDS:
    v = val_at_u(f"ent01_s{s}", 30)
    if v is not None:
        base[s] = v
        print(f"  ent01_s{s}  u30 = {v:.4f}")
if len(base) < 2:
    print("  !! 基线数据不足，无法继续")
    raise SystemExit(1)

vals = list(base.values())
sd_base = statistics.stdev(vals) if len(vals) > 1 else 0.0
print(f"  均值 {statistics.mean(vals):.4f}   SD {sd_base:.4f}  (n={len(vals)})")
print()
print("  注：这就是本项目记录的『逐种子成功率跨 0.29–0.88』在**同一配置内**的")
print("  对应量 —— 同一配置换种子也会动，SD 就是这个数。")
print()

# ---------- 2. 保守口径（ρ=0） ----------
print("=== 2. 分辨率（保守：假设 arm 与 ent01 逐种子独立，ρ=0）===")
sd_diff = math.sqrt(2) * sd_base
se = sd_diff / math.sqrt(len(SEEDS))
print(f"  SD(逐种子差) = sqrt(2)×{sd_base:.4f} = {sd_diff:.4f}   （ρ=0 是上界）")
print(f"  SE(Δ) = {sd_diff:.4f}/sqrt({len(SEEDS)}) = {se:.4f}")
print()
t_at_delta = DELTA / se if se > 0 else float("inf")
print(f"  Δ={DELTA} → t = {t_at_delta:.2f}")
print(f"  → {'可分辨（t>2）' if t_at_delta > 2 else '**分辨不了**（t<2）'}")
print()

# 反推：SE 要做到多少才能分辨 0.035（t=2）
se_needed = DELTA / 2.0
n_needed = (sd_diff / se_needed) ** 2
print(f"  若要 t>2 分辨 0.035，需要 SE ≤ {se_needed:.4f} → n ≥ {n_needed:.1f} 个种子")
print(f"  （保守 ρ=0 口径下）")
print()

# ---------- 3. 经验校准：同配置两次跑的零分布 ----------
print("=== 3. 经验校准：找同配置、不同种子的成对 run ===")
print("  （零分布 = 同一配置两次跑出来的差有多大；它比公式更可信）")
print()

# 用 ent01 家族：ent01_s42 与 ent01_s42_u30to50 是同配置续跑；
# 更直接的零分布来自"同配置两套种子"。这里扫 outputs 找候选。
groups = defaultdict(list)
for d in sorted(OUT.iterdir()):
    if not d.is_dir():
        continue
    rc = d / "resolved_config.yaml"
    if not rc.exists():
        continue
    # 用 resolved_config 的哈希分组（排除 run-name / seed 等无关字段）
    txt = rc.read_text(encoding="utf-8", errors="replace")
    keys = [
        ln.strip() for ln in txt.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
        and not ln.strip().startswith(("run_name", "seed", "output_dir", "resume"))
    ]
    groups[hash("\n".join(keys))].append(d.name)

dups = {k: v for k, v in groups.items() if len(v) >= 2}
print(f"  找到 {len(dups)} 组『同 resolved 配置』的 run（≥2 个）")
shown = 0
for k, v in sorted(dups.items(), key=lambda kv: -len(kv[1])):
    if shown >= 8:
        break
    print(f"    {len(v)} 个: {', '.join(v[:6])}")
    shown += 1
if not dups:
    print("    （无）—— 说明历史 run 的 resolved_config 都不同，")
    print("     那么零分布只能等 runt 锚点跑完才有（它本身就是同配置重跑）")
print()

# ---------- 4. 结论 ----------
print("=== 4. 结论 ===")
print(f"  保守口径（ρ=0）下，3 个种子对 Δ={DELTA} 的 t = {t_at_delta:.2f}。")
if t_at_delta > 3:
    print("  → 分辨率充足。而且**保守口径给的是下界**：")
    print("    arm 与 ent01 从同一个 BC 起点出发、用同一批种子，逐种子正相关是")
    print("    理应存在的（seed 43 好的时候两边一起好），ρ>0 会让 SE 更小。")
    print("  → 因此 3 个种子够用，不需要现在加种子。")
elif t_at_delta > 2:
    print("  → 勉强够（2<t<3），结论要谨慎措辞。")
else:
    print("  → **不够**。应当趁链条还能改，考虑加种子。")
print()
print("  另外：`runt` 臂本身就是**同配置重跑**，它跑完后可以直接量出")
print("  真实的零分布（含机器漂移），比这里的公式估计更可信 ——")
print("  这也是当初把它放进链里的原因（不只是量时间漂移）。")
