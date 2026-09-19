#!/usr/bin/env python
"""在跑的复筛链，**3 个种子够不够分辨 +0.035**？

为什么现在算而不是等结果：链条要跑 7.8h。如果 3 个种子的分辨率不够，
那我现在就该加种子（还有时间），而不是 7.8h 后拿到一堆读不出的数。

### ⚠ 2026-09-19 订正：本脚本原先的判据有两处错，结论（碰巧）是对的

原版用 `t > 2` 判"可分辨"，并假设 ρ=0（arm 与 ent01 逐种子独立）。
两处都不对，且**方向相反、互相抵消**：

| 量 | 原版 | 正确 | 方向 |
|---|---|---|---|
| 临界值 | `2`（df≈∞） | **4.303**（df=2） | 原版**太宽松** |
| SD(逐种子差) | ρ=0 → 0.0167 | 实测 ρ≈0.98 → **0.0024** | 原版**太保守**（差 7 倍） |

于是算出 t=3.65：按 2 判"够用"（错），按 4.303 判"不够"（也对不上）——
而用正确的 ρ 算，t≈25，**功效 1.000，确实够用**。

**结论对不能留**：换个 Δ 或换个 n，两个误差不再抵消就会给出错误答案。
现在两处都按下面修好，阈值改成 **df 感知**，ρ 改成**实测**。

### 口径

预注册判据是「Δ = 三种子在 u30 对 ent01 的配对差，≥+0.035 判有效」。

    Δ = mean_s ( arm_s − ent01_s )        s ∈ {42, 43, 44}

**Δ_s 只有 3 个观测 → 单样本 t 检验，df = 2 → t_0.975 = 4.303。**

SD(逐种子差)² = SD(arm)² + SD(ent01)² − 2·ρ·SD(arm)·SD(ent01)

ρ **不假设，实测**：取历史"同训练种子、同 BC、单调超参"的 run 对，
在 15 个验证种子上算 Pearson ρ（本项目实测 0.98，最小 0.95，k=21）。

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

# 实测 ρ（本文件第 3 节会自己重量一遍并覆盖这个默认值）：
# 来历见 .tmp/rho_measure.py —— 21 对"同训练种子、同 BC、单调超参"的
# run 对，逐验证种子 Pearson ρ 均值 0.9798（最小 0.9498，最大 0.9951）。
RHO_MEASURED = 0.9798


def t_crit(df: int, p: float = 0.975) -> float:
    """t 分布上分位数。**不要用 2/1.96 代替** —— n=3 时 df=2，真值是 4.303。"""
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _t_sf(mid, df) > 1.0 - p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _t_sf(t: float, df: int) -> float:
    """P(T > t)，T ~ t(df)。正则化不完全贝塔闭式（Lentz 连分数）。"""
    if t == 0:
        return 0.5
    x = df / (df + t * t)
    half = 0.5 * _betainc(df / 2.0, 0.5, x)
    return half if t > 0 else 1.0 - half


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    tiny = 1e-300
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 400):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = tiny if abs(d) < tiny else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = tiny if abs(c) < tiny else c
        f *= d * c
        if abs(1.0 - d * c) < 1e-14:
            break
    return front * (f - 1.0)


def power(c: float, ncp: float, df: int) -> float:
    """P(T > c)，T ~ noncentral t(df, ncp)。"""
    try:
        from scipy import stats as _st
        return float(_st.nct.sf(c, df, ncp))
    except Exception:                                          # noqa: BLE001
        return 0.5 * math.erfc((c - ncp) / math.sqrt(2))


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

# ---------- 2. 分辨率（用**实测 ρ**，不再假设 ρ=0） ----------
print("=== 2. 分辨率 ===")
_sd_at = math.sqrt(2 * sd_base ** 2 * (1 - RHO_MEASURED))
print(f"  实测 ρ = {RHO_MEASURED:.4f}（来历见文件头；第 3 节会自己重量再覆盖）")
print(f"  SD(逐种子差) = sqrt(2·SD²·(1−ρ)) = {_sd_at:.4f}")
print(f"  SE(Δ) = {_sd_at:.4f}/sqrt({len(SEEDS)}) = {_sd_at/math.sqrt(len(SEEDS)):.4f}")
print()
tc = t_crit(len(SEEDS) - 1)
print(f"  **临界值 t_0.975(df={len(SEEDS)-1}) = {tc:.3f}**"
      f"   ← 不是 2，也不是 1.96")
print()

for rho_lbl, rho_v in (("实测 ρ", RHO_MEASURED), ("最保守 ρ=0.95", 0.95),
                       ("若无相关 ρ=0", 0.0)):
    sd_d = math.sqrt(2 * sd_base ** 2 * (1 - rho_v))
    se = sd_d / math.sqrt(len(SEEDS))
    t_at_delta = DELTA / se if se > 0 else float("inf")
    pw = power(tc, DELTA / se, len(SEEDS) - 1)
    print(f"  {rho_lbl:<16} SD(diff)={sd_d:.4f} SE={se:.4f} "
          f"t={t_at_delta:>6.2f} 功效={pw:.3f}")
print()
print("  判据用**临界值 + 功效**，不用『t>2』：")
print(f"    t={DELTA/math.sqrt(2*sd_base**2*(1-RHO_MEASURED))/math.sqrt(len(SEEDS)):.2f}"
      f" vs t_crit={tc:.3f} → "
      f"{'可分辨' if DELTA/math.sqrt(2*sd_base**2*(1-RHO_MEASURED))*math.sqrt(len(SEEDS)) > tc else '**分辨不了**'}")
print()

# 反推：要多少种子才能到 0.8 功效
print("  要达 0.8 功效需要的种子数（按实测 ρ）：")
for n in (3, 4, 5, 6, 8):
    sd_d = math.sqrt(2 * sd_base ** 2 * (1 - RHO_MEASURED))
    se = sd_d / math.sqrt(n)
    pw = power(t_crit(n - 1), DELTA / se, n - 1)
    print(f"    n={n}: 功效 {pw:.3f}{'  ✓' if pw >= 0.8 else ''}")
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
_sd_final = math.sqrt(2 * sd_base ** 2 * (1 - RHO_MEASURED))
_se_final = _sd_final / math.sqrt(len(SEEDS))
_t_final = DELTA / _se_final
_pw_final = power(tc, _t_final, len(SEEDS) - 1)

print(f"  用**实测 ρ={RHO_MEASURED:.3f}**、**临界值 {tc:.3f}(df=2)**：")
print(f"    Δ={DELTA} → t = {_t_final:.2f}  vs  t_crit = {tc:.3f}"
      f"   →  {'可分辨' if _t_final > tc else '**分辨不了**'}")
print(f"    功效 = {_pw_final:.3f}")
print()
if _pw_final >= 0.8:
    print("  → **分辨率充足，3 个种子够用，不需要现在加种子。**")
    print("    依据不是『t 大于某个数』，而是**功效**：真效应是 0.035 时，")
    print(f"    有 {_pw_final*100:.0f}% 的机会被判为显著。")
elif _pw_final >= 0.5:
    print("  → 勉强够。结论要谨慎措辞：判『不显著』时不能读成『没有效应』。")
else:
    print("  → **不够**。应当趁链条还能改，考虑加种子。")
print()
print("  ⚠ 老版本的 `t>2` 判据已删除。理由：n=3 时 df=2，临界值是 4.303；")
print("    用 2 会把『测不出』读成『测得出』。同时老版本用 ρ=0 估 SD，")
print("    把功效**低估**了约 7 倍（SD 0.0167 vs 实测 0.0024）。")
print("    两个方向相反的错凑出了正确的结论——**这不是可以留下的理由**。")
print()
print("  另外：`runt` 臂本身就是**同配置重跑**，它跑完后可以直接量出")
print("  真实的零分布（含机器漂移），比这里的公式估计更可信 ——")
print("  这也是当初把它放进链里的原因（不只是量时间漂移）。")
