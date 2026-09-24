#!/usr/bin/env python
"""vcoef1 到底有没有用？—— 在**它自己的基线**上配对算。

背景：vcoef1 跑在 entropy_coef=0.001（`resolved_config.yaml` 确认），
而 0.001 是本项目已判定为错的值。所以问题分两层：

  层一：value_coef 0.5→1.0 在 ent=0.001 上有没有用？
        → vcoef1_s42/43/44  vs  r8_base_s42/43/44（同种子、同 ent、只差 value_coef）
  层二：ent=0.01 才是现用基线，所以在基线上重测才有意义
        → 由层一的符号决定**优先级**，但无论结果都要在 ent=0.01 上重测
          （本项目已确立「跨场景/跨基线不保持」，不能假设可搬）

u5/u10/u15 是 r8_base 与 vcoef1 的公共轮次（r8_base 只跑到 15）。
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
COMMON = (5, 10, 15)


def points(run: str) -> dict[int, list[float]]:
    p = OUT / run / "metrics.jsonl"
    out: dict[int, list[float]] = {}
    if not p.exists():
        return out
    upd = None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "update" in o:
            upd = o["update"]
        ev = o.get("eval_validation")
        if isinstance(ev, dict) and ev.get("per_seed_success") and upd is not None:
            out[upd] = [float(x) for x in ev["per_seed_success"]]
    return out


def t_crit(df: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except ImportError:
        return {2: 4.303, 3: 3.182, 4: 2.776}.get(df, 2.0)


def t_sf(t: float, df: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.sf(abs(t), df))
    except ImportError:
        return float("nan")


SEEDS = (42, 43, 44)


def paired(a_runs, b_runs, label_a, label_b):
    """逐轮配对：Δ = a − b，先按轮配对再跨轮取平台均值。"""
    print(f"  {label_a}  −  {label_b}")
    print(f"    {'轮':>4}{'Δ':>10}{'SD':>9}{'SE':>9}{'t':>8}{'p':>9}  判读")
    per_round = {}
    for u in COMMON:
        ds = []
        for s in SEEDS:
            pa = points(f"{a_runs}{s}").get(u)
            pb = points(f"{b_runs}{s}").get(u)
            if not pa or not pb or len(pa) != len(pb):
                continue
            ds.append(statistics.mean([pa[i] - pb[i] for i in range(len(pa))]))
        if len(ds) < 2:
            print(f"    {u:>4}   (可用种子不足: {len(ds)})")
            continue
        m = statistics.mean(ds)
        sd = statistics.stdev(ds)
        se = sd / math.sqrt(len(ds))
        t = m / se if se else float("nan")
        df = len(ds) - 1
        p = 2 * t_sf(t, df)
        verdict = ("显著" if p < 0.05 else "未达显著")
        per_round[u] = m
        print(f"    {u:>4}{m:>+10.4f}{sd:>9.4f}{se:>9.4f}{t:>+8.2f}{p:>9.4f}  {verdict}")
    if per_round:
        mm = statistics.mean(per_round.values())
        print(f"    u5-15 汇总均值 Δ = {mm:+.4f}")
        print(f"    （n=3，df=2，临界值 {t_crit(2):.3f}——别用 2 判）")
    print()
    return per_round


print("=" * 80)
print("层一：value_coef 0.5→1.0 在 entropy_coef=0.001 上有效吗？")
print("=" * 80)
print("  vcoef1 = ent0.001 + value1.0 ；r8_base = ent0.001 + value0.5")
print()
paired("vcoef1_s", "r8_base_s", "vcoef1", "r8_base")

print("=" * 80)
print("层二：与 ent=0.01 的差距（注意此对照**混淆**了 entropy 与 value 两项）")
print("=" * 80)
print("  vcoef1 = ent0.001+value1.0 ；ent01 = ent0.01+value0.5")
print("  两者差两项，**不能**归因到单一旋钮——只用来判断量级。")
print()
paired("ent01_s", "vcoef1_s", "ent01", "vcoef1")

print("=" * 80)
print("结论")
print("=" * 80)
print("  层一若为「未达显著」→ value_coef=1.0 在旧基线上也没测出效果，")
print("     那么在 ent=0.01 上重测它的优先级**低**，应把时间给 ep2/mini512。")
print("  层一若为「显著为正」→ 值得在 ent=0.01 上重测（可能是被错误基线埋掉的真收益）。")
