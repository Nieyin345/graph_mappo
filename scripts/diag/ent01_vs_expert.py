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

用法（服务器上）：
    python scripts/diag/ent01_vs_expert.py                     # 默认 ent01_s{42,43,44}
    python scripts/diag/ent01_vs_expert.py --runs ent01_rerun_s{42,43,44}

### ★★ 2026-09-19 补充：默认的 `ent01_s*` 与 `ent01_t8_*` 是**跨节点搬来的**

`ent01_s{42,43,44}` / `ent01_t8_s{42,43,44}` 跑在**旧节点 amd238（EPYC 7402P）**，
硬件偏置 **+0.0197**（已用两种独立方法测出同一个数，见
`docs/训练诊断记录.md`「★★★ 重标定终判」）。用它当基线会把偏置读成"RL 赢了专家"。

**本节点、8 线程的干净基线是 `ent01_rerun_s{42,43,44}`**（或第五波之后的 `ent01_s*` 重跑）。
所以本脚本加了 `--runs` 参数，比较任何臂之前先确认基线是新节点跑的。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = OUT / "eval" / "expert_seeds100_240.json"
SEEDS = [42, 43, 44]

_ap = argparse.ArgumentParser(description="RL vs 专家：跨训练种子分母（正确的那个）")
_ap.add_argument("--runs", default="ent01",
                 help="臂名前缀，脚本会拼成 <runs>_s{42,43,44}。"
                      "★ 本节点的干净基线是 ent01_rerun；ent01 / ent01_t8 是跨节点搬来的，"
                      "含 +0.0197 硬件偏置。")
ARGS = _ap.parse_args()
STEM = ARGS.runs


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
print(f"  待判臂: {STEM}_s{SEEDS}")
print()

pts = {s: points(f"{STEM}_s{s}") for s in SEEDS}
for u in (15, 20, 25, 30):
    print(f"=== u{u} ===")
    deltas: list[float] = []
    valid_t: list[float] = []
    for s in SEEDS:
        ps = pts[s].get(u)
        if ps is None or len(ps) != len(seeds):
            print(f"  {STEM}_s{s}: 缺 u{u} 或长度不符")
            continue
        diffs = [ps[i] - base[i] for i in range(len(seeds))]
        md = statistics.mean(diffs)
        sed = statistics.stdev(diffs) / math.sqrt(len(diffs))
        t = md / sed if sed else float("nan")
        deltas.append(md)
        valid_t.append(t)
        print(f"  {STEM}_s{s}  均值 {statistics.mean(ps):.4f}  "
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
    if se and se != se:
        verdict = "t 无定义（SE 无效）"
    elif p_two < 0.05:
        # ★★ 判词**必须看符号**。原写法用 p_two = 2*_t_sf(abs(t), df) 把方向抹掉后
        #    只按 p 分档 ⟹ 臂**显著更差**时会打印「显著超过」——一个负结论被翻成
        #    正面主张，而这是 README 指定的权威判读口径。
        #    本项目真的有过 Δ<0 且过线的历史（跨节点偏置下 ent01_s43 逐点配对
        #    t=+4.28，去偏置后转负）⟹ 换上正确基线就可能踩到。
        verdict = "显著超过" if m > 0 else "★ **显著更差**（方向为负且过线）"
    elif p_two < 0.10:
        verdict = ("**未达显著**（p<0.10，方向一致）" if m > 0
                   else "**未达显著**（p<0.10，方向为**负**，不能说「一致」）")
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
print("  判据是**跨训练种子的双侧 p < 0.05**（df=n−1 的 t 分布），不是 t>2，")
print("  且**必须看符号**：p 小只说明「有差异」，方向由 Δ 的正负决定。")
print(f"  若尾部几轮一致**为正**且显著，则『{STEM} 在验证 regime 达到/超过专家』成立；")
print(f"  若一致**为负**且显著，则结论相反——『{STEM} 在验证 regime 显著**差于**专家』。")
print("  若只是方向一致但 p>0.05，只能说『**这次没测出显著差异**』——")
print("  **不等于**『没有差异』（n=3 的功效本身就有限，见 scripts/diag/rho_measure.py）。")
