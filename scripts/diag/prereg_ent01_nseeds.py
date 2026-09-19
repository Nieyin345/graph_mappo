#!/usr/bin/env python
"""**预注册**：要几个训练种子才能判定 `ent01` 是否真的超过专家？

### 起因

订正后的结论是：ent01 相对专家 Δ≈+0.0199（u25/u30 平台），
但在 n=3（df=2，临界值 4.303）下 t=3.46、**p=0.106，未达显著**。

这不是"没有效应"，是**功效不够**。所以问题是：**再加几个种子才算够？**
回答它必须用**已有的 n=3 数据**去算，而不是跑完再看（跑完再看就是 p-hacking）。

### 这个比较的 SD 与 power_check.py 那个不是同一个

- `power_check.py` 算的是 **arm_s − ent01_s**（两者都随种子变）
  → SD(diff) 里 ρ 很重要。
- 这里算的是 **ent01_s − 专家**，**专家是一个固定的数**
  → SD(Δ_s) = SD(ent01_s)，与 ρ 无关。

所以直接用 ent01 三种子在平台区的散布：
  u25/u30 的逐种子平台均值 = +0.0084 / +0.0256 / +0.0256 → SD = 0.0099

### 预注册内容（**跑之前**写死；下面的 n 由本脚本第 2 节算出并打印）

- 加种子 **45、46**（与 42/43/44 不重复），同配置、同 BC、同 30 轮。
- 判据：**合并 n=5**，取 u25/u30 平台均值，单样本 t 检验，
  df=4，**临界值 2.776**，双侧 **p<0.05** 判"显著超过专家"。
- **无论方向如何都照实报**：若合并后不显著，结论就是"未测出"，
  不许再补种子直到显著（那正是 p-hacking）。

用法（服务器上）：python scripts/diag/prereg_ent01_nseeds.py
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = OUT / "eval" / "expert_seeds100_240.json"

# 已有三种子（用于估 SD，**不用于定结论**）
OLD = [42, 43, 44]
PLATEAU = (25, 30)


def t_crit(df: int, p: float = 0.975) -> float:
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _t_sf(mid, df) > 1.0 - p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _t_sf(t: float, df: int) -> float:
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
    try:
        from scipy import stats as _st
        return float(_st.nct.sf(c, df, ncp))
    except Exception:                                          # noqa: BLE001
        return 0.5 * math.erfc((c - ncp) / math.sqrt(2))


d = json.loads(EXPERT.read_text(encoding="utf-8"))
ex = dict(zip(d["seeds"], (float(x) for x in d["success"])))
seeds = sorted(ex)
base = [ex[s] for s in seeds]


def points(run: str) -> dict[int, list[float]]:
    p = OUT / run / "metrics.jsonl"
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


per_seed_old = []
for s in OLD:
    pts = points(f"ent01_s{s}")
    us = [statistics.mean([pts[u][i] - base[i] for i in range(len(seeds))])
          for u in PLATEAU if u in pts and len(pts[u]) == len(seeds)]
    if us:
        per_seed_old.append(statistics.mean(us))

m_old = statistics.mean(per_seed_old)
sd = statistics.stdev(per_seed_old)
print("=" * 74)
print("1. 用已有 n=3 数据估这个比较的尺度")
print("=" * 74)
for s, v in zip(OLD, per_seed_old):
    print(f"  ent01_s{s}: u25/u30 平台配对差 = {v:+.4f}")
print(f"  Δ = {m_old:+.4f}   SD(Δ_s) = {sd:.4f}   （专家是固定值 ⟹ 与 ρ 无关）")
print()

print("=" * 74)
print("2. 功效 → 需要几个种子")
print("=" * 74)
print(f"  {'n':>3} {'SE':>8} {'ncp':>7} {'t_crit':>8} {'功效':>8}")
need = None
for n in range(3, 11):
    se = sd / math.sqrt(n)
    df, tc = n - 1, t_crit(n - 1)
    pw = power(tc, m_old / se, df)
    print(f"  {n:>3} {se:>8.4f} {m_old/se:>7.2f} {tc:>8.3f} {pw:>8.3f}"
          f"{'  ← 首次 ≥0.8' if pw >= 0.8 and need is None else ''}")
    if pw >= 0.8 and need is None:
        need = n
print()
print(f"  ⟹ 达到 0.8 功效需要 **n = {need}** 个训练种子 → 再加 **{need-3}** 个。")
print()
print("  ⚠ 这是用 n=3 估的 SD（约 50% 不确定度）。若真实 SD 更大，需要更多种子；")
print("    所以**预先写明**：若合并后仍未达显著，结论就是『未测出』，")
print("    不再追加种子——追加到显著为止是 p-hacking。")
print()

print("=" * 74)
print("3. 预注册判据（照此执行，不事后改）")
print("=" * 74)
df_f = need - 1
_new = [45, 46, 47, 48][:need - 3]
print(f"  新种子：**{', '.join(map(str, _new))}**（不与 {OLD} 重复）")
print(f"  配置：configs/train_ent01.yaml，同 BC checkpoint，30 轮")
print(f"  判据：合并 **n={need}**，u25/u30 平台均值，单样本 t 检验")
print(f"        df={df_f}，临界值 **{t_crit(df_f):.3f}**，双侧 **p < 0.05**")
print(f"  报法：无论方向，照实报。显著 → 说「超过」；不显著 → 说「未测出」。")
print(f"  预算：{need-3} 个种子 = 1 波，约 2h")
