#!/usr/bin/env python
"""小样本判决：n=3 时 t 的临界值是 **4.303**，不是 2。

### 为什么必须写这个脚本

前面我用 `t > 2` 作为"显著"的判据，得到 u25 t=+4.09、u30 t=+3.03、
平台合并 t=+3.46，于是说"显著超过专家"。**这个阈值是错的。**

这里的结构是：每个**训练种子**给出一个配对差 Δ_s（在 15 个验证种子上配对），
Δ_s 是 3 个 iid 抽样。检验 H0: E[Δ]=0 是**单样本 t 检验，df = n−1 = 2**。
t 分布 df=2 的 0.975 分位数是 **4.303**，不是正态的 1.96。

用 2 当阈值 = 把 n=3 的噪声当成 n=∞。**这正好是本项目记录里反复出现的
同一类错误：拿一个不适用的数量级当判据**（哨兵阈值、内存判据都栽过）。

另外 SD(Δ_s) 本身在轮次间从 0.0082 跳到 0.0312，n=3 估出的 SD
置信区间很宽，所以**必须报"结论对 SD 估计的敏感性"**，不能只报一个 t。

用法（服务器上）：python /tmp/ent01_smallsample.py
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = OUT / "eval" / "expert_seeds100_240.json"
SEEDS = [42, 43, 44]
ROUNDS = (15, 20, 25, 30)


def _t_sf(t: float, df: int) -> float:
    """P(T > t)，df 自由度。闭式解，够用（df 小）。"""
    x = df / (df + t * t)
    # 正则化不完全贝塔 I_x(df/2, 1/2) 的连分数（Lentz）
    a, b, aa, c = 1.0, 1.0 - x, 1.0, 1.0
    # 用连分数算 I_x(df/2, 1/2)
    def betainc(a_, b_, x_):
        if x_ <= 0:
            return 0.0
        if x_ >= 1:
            return 1.0
        lbeta = math.lgamma(a_) + math.lgamma(b_) - math.lgamma(a_ + b_)
        front = math.exp(math.log(x_) * a_ + math.log(1 - x_) * b_ - lbeta) / a_
        f, cc, d = 1.0, 1.0, 0.0
        for i in range(0, 300):
            m = i // 2
            if i == 0:
                num = 1.0
            elif i % 2 == 0:
                num = (m * (b_ - m) * x_) / ((a_ + 2 * m - 1) * (a_ + 2 * m))
            else:
                num = -((a_ + m) * (a_ + b_ + m) * x_) / (
                    (a_ + 2 * m) * (a_ + 2 * m + 1))
            d = 1.0 + num * d
            d = 1e-30 if abs(d) < 1e-30 else d
            d = 1.0 / d
            cc = 1.0 + num / cc
            cc = 1e-30 if abs(cc) < 1e-30 else cc
            f *= d * cc
            if abs(1.0 - d * cc) < 1e-12:
                break
        return front * (f - 1.0)

    ib = betainc(df / 2.0, 0.5, x)
    return 0.5 * ib if t > 0 else 1.0 - 0.5 * betainc(df / 2.0, 0.5, x)


# t 分布分位数（数值反解）
def _t_crit(df: int, p: float = 0.975) -> float:
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1.0 - _t_sf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


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


pts = {s: points(f"ent01_s{s}") for s in SEEDS}

print("=" * 72)
print("n=3（df=2）的正确临界值")
print("=" * 72)
for df, name in ((2, "df=2 (n=3)"), (1, "df=1 (n=2)")):
    print(f"  {name}: t_0.975 = {_t_crit(df):.3f}   "
          f"t_0.95 = {_t_crit(df, 0.95):.3f}")
print("  （对照：正态/大样本的 1.96 —— **n=3 时用它就是错的**）")
print()

print("=" * 72)
print("逐轮：Δ_s → t(df=2) → 精确 p（双侧）")
print("=" * 72)
rows = []
for u in ROUNDS:
    ds = []
    for s in SEEDS:
        ps = pts[s].get(u)
        if ps is None or len(ps) != len(seeds):
            continue
        ds.append(statistics.mean([ps[i] - base[i] for i in range(len(seeds))]))
    m = statistics.mean(ds)
    sd = statistics.stdev(ds)
    se = sd / math.sqrt(len(ds))
    t = m / se if se else float("nan")
    p = 2 * _t_sf(abs(t), len(ds) - 1)
    rows.append((u, ds, m, sd, se, t, p))
    sig = "**显著**" if p < 0.05 else "不显著"
    print(f"  u{u}: Δ={m:+.4f} SD={sd:.4f} SE={se:.4f} t={t:+.3f} "
          f"p={p:.4f} (df={len(ds)-1})  {sig}")
print()

print("=" * 72)
print("敏感性：换个 SD 估计，结论会不会翻")
print("=" * 72)
var_pool = statistics.mean([r[3] ** 2 for r in rows])
sd_pool = math.sqrt(var_pool)
print(f"  逐轮 SD = {[round(r[3], 4) for r in rows]}")
print(f"    → u20 的 SD 是邻居的 ~3.8 倍，**n=3 估出的 SD 本身极不稳**")
print(f"  合并 SD（跨 4 轮）= {sd_pool:.4f}")

print()
print("  用不同 SD 估计，u25/u30 平台合并（Δ=+0.0199）的读数：")
for lbl, sdv in (("逐轮 SD 取小者(u25=0.0082)", 0.0082),
                 ("逐轮 SD 取大者(u30=0.0118)", 0.0118),
                 ("合并 SD(跨4轮)", sd_pool),
                 ("u20 的离群 SD(0.0312)", 0.0312)):
    se = sdv / math.sqrt(3)
    t = 0.0199 / se
    p = 2 * _t_sf(abs(t), 2)
    print(f"    {lbl:<30} SD={sdv:.4f} → t={t:+.3f} p={p:.4f} "
          f"{'显著' if p < 0.05 else '不显著'}")
print()

print("=" * 72)
print("同号检验（不依赖 SD，只看方向）")
print("=" * 72)
for u, ds, m, sd, se, t, p in rows:
    pos = sum(1 for x in ds if x > 0)
    print(f"  u{u}: {pos}/3 为正  → 符号检验双侧 p = "
          f"{2 * (0.5 ** 3) * sum(math.comb(3, k) for k in range(pos, 4)):.4f}"
          if pos >= 2 else
          f"  u{u}: {pos}/3 为正")
print("  注：n=3、3/3 同号的两侧 p = 0.25 —— **方向一致性本身给不出显著性**。")
print()

print("=" * 72)
print("判决")
print("=" * 72)
print("  用正确的 df=2 临界值 4.303：")
for u, ds, m, sd, se, t, p in rows:
    print(f"    u{u}: t={t:+.3f}  {'达到' if abs(t) > 4.303 else '**未达到**'} 4.303")
print()
print("  ⇒ 即使按最有利的（逐轮）SD：只有 u25 的 t=+4.09 逼近临界值，")
print("    p≈0.055 **仍未过 0.05**；u30 与平台合并都不到。")
print("  ⇒ 正确结论：**方向一致（2/3 种子稳定为正）、量级 +0.02，")
print("    但在 n=3 下未达统计显著**。要下『显著超过』的结论需要更多训练种子。")
