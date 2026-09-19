#!/usr/bin/env python
"""ent01（entropy_coef=0.01，30 轮）**到底有没有超过专家**？两个分母都算。

### 为什么必须两个分母都算

`paired_vs_expert.py` 给出的 SE 是**评估实例级**的（15 个验证种子上逐点配对），
它对 `ent01_s43` 给出 t=+4.28。但 `docs/测试规范.md` ⑦ 明确说：

> 训练种子本身是比评估种子更大的方差来源 …… 至少 3 个训练种子，比分布。

也就是说那个 t 只覆盖了"同一份权重、换验证种子"的噪声，**不覆盖"换个训练种子"**。
真正的分母要按 **3 个训练种子的 Δ** 来算：

    Δ_s = 该种子在 u30 对专家的配对差     s ∈ {42,43,44}
    SE(Δ) = SD(Δ_s)/sqrt(3)

这正是本项目记录里那次"假的配对"（专家 vs RL，配对 SE=0.0110，t=4.07）栽的地方。

### 另外查一件事：这个"超过"是不是靠挑轮次挑出来的

u25–u30 已被证实是平台区（见日志 u50 判决那节）。所以这里把 **u20/u25/u30**
三个点都算出来 —— 若只有某一个点显著，那是挑点；若尾部一致，才是真的。

用法（服务器上）：python scripts/diag/ent01_vs_expert.py
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = OUT / "eval" / "expert_seeds100_240.json"
SEEDS = [42, 43, 44]


def t_crit(df: int, p: float = 0.975) -> float:
    """t 分布上分位数。**n=3 时 df=2 → 4.303**，不是 2 也不是 1.96。

    2026-09-19 订正：本脚本原先用 `t > 2` 判"显著超过"，把 n=3 的噪声
    当成了 n=∞。见 docs/训练诊断记录.md「订正两处判据错误」。
    """
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


def expert() -> dict[int, float]:
    d = json.loads(EXPERT.read_text(encoding="utf-8"))
    return dict(zip(d["seeds"], (float(x) for x in d["success"])))


def points(run: str) -> dict[int, list[float]]:
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        return {}
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


ex = expert()
seeds = sorted(ex)
base = [ex[s] for s in seeds]
print(f"专家 @ 种子 {seeds[0]}-{seeds[-1]}（n={len(seeds)}）")
print(f"  均值 = {statistics.mean(base):.4f}   SD = {statistics.stdev(base):.4f}")
print()

pts = {s: points(f"ent01_s{s}") for s in SEEDS}
for u in (15, 20, 25, 30):
    print(f"=== u{u} ===")
    deltas: list[float] = []
    valid_t: list[float] = []
    for s in SEEDS:
        ps = pts[s].get(u)
        if ps is None or len(ps) != len(seeds):
            print(f"  ent01_s{s}: 缺 u{u} 或长度不符")
            continue
        diffs = [ps[i] - base[i] for i in range(len(seeds))]
        md = statistics.mean(diffs)
        sed = statistics.stdev(diffs) / math.sqrt(len(diffs))
        t = md / sed if sed else float("nan")
        deltas.append(md)
        valid_t.append(t)
        print(f"  ent01_s{s}  均值 {statistics.mean(ps):.4f}  "
              f"配对差 {md:+.4f} (评估侧 t={t:+.2f})")
    if len(deltas) < 2:
        print("  种子不足，跳过\n")
        continue
    m = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    se = sd / math.sqrt(len(deltas))
    t_train = m / se if se else float("nan")
    df = len(deltas) - 1
    tc = t_crit(df)
    p_two = 2 * _t_sf(abs(t_train), df) if se else float("nan")
    print(f"  --- 跨训练种子（**正确的分母**，n={len(deltas)}）---")
    print(f"  Δ = {m:+.4f}   SD(Δ_s) = {sd:.4f}   SE = {se:.4f}")
    print(f"  t = {t_train:+.2f}  (df={df})   临界值 t_0.975 = {tc:.3f}   "
          f"双侧 p = {p_two:.4f}")
    if p_two < 0.05:
        verdict = "显著超过"
    elif p_two < 0.10:
        verdict = "**未达显著**（p<0.10，方向一致）"
    else:
        verdict = "不显著（打平）"
    print(f"  判读: {verdict}")
    print(f"  ⚠ 别用 t>2 判：n=3 时 df=2，临界值是 {tc:.3f}。"
          f"用 2 会把『测不出』读成『测得出』。")
    # 也报一下"把评估侧 t 平均"这个错误口径，用来说明差别有多大
    print(f"  （对照：只看评估侧，t 的均值为 {statistics.mean(valid_t):+.2f} "
          f"—— 这是**偏乐观**的口径，因为忽略训练种子方差）")
    print()

print("=== 结论 ===")
print("  判据是**跨训练种子的双侧 p < 0.05**（df=n−1 的 t 分布），不是 t>2。")
print("  若尾部几轮一致显著，则『ent01 在验证 regime 达到/超过专家』成立；")
print("  若只是方向一致但 p>0.05，只能说『**这次没测出显著差异**』——")
print("  **不等于**『没有差异』（n=3 的功效本身就有限，见 scripts/diag/rho_measure.py）。")
