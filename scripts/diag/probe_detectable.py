"""各族**能测出的最小效应**（可检测效应量）—— 判据的分辨率不是常数。

## 为什么必须逐族算

我一直引用的"单训练种子分辨率 ~0.035"（`measurement-protocol-two-speeds`）
是**跨种子的 per-seed 成功率 SD**，不是**配对差 Δ_s 的 SD**。
两者差很远：配对把种子间的公共波动消掉了。

实测各族 Δ_s 的 SD：

    cmax         SD = 0.0098   ⟹ 分辨率 0.012
    ent01_rerun  SD = 0.0086   ⟹ 分辨率 0.011
    scratch      SD = 0.1183   ⟹ 分辨率 0.146

⟹ **同一个 n=5，BC 族的门限比从零族紧 12 倍。**
   在 BC 族里"测不出"，是**真的能排除 0.012 以上的效应**；
   在从零族里"测不出"，只排除了 0.146 以上 —— 几乎什么都没说。

## 判据

可检测效应 = t_crit(df) × SE = t_crit × SD / √n
"""
import json
import math
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
T = {2: 4.303, 3: 3.182, 4: 2.776, 8: 2.306, 14: 2.145}


def valpts(run, update=30):
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        return None
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
        if "eval_validation" in o and last == update:
            return list(o["eval_validation"]["per_seed_success"])
    return None


def family(runs):
    """族内逐臂 u30 均值 → 跨臂 SD（= Δ_s 的 SD，因为同种子配对）"""
    vals = []
    for r in runs:
        v = valpts(r)
        if v:
            vals.append(statistics.mean(v))
    return vals


def main():
    fams = {
        "cmax（critic max 池化）": [f"cmax_s{s}" for s in range(42, 47)],
        "ent01_rerun（BC 基线）": [f"ent01_rerun_s{s}" for s in range(42, 47)],
        "v2_bottleneck": [f"v2_bottleneck_s{s}" for s in range(42, 47)],
        "v1_onpath": [f"v1_onpath_s{s}" for s in range(42, 47)],
        "pm_decode": [f"pm_decode_s{s}" for s in range(42, 47)],
        "scratch（从零）": [f"scratch_s{s}" for s in range(42, 47)],
    }

    print("=" * 96)
    print("各族的**可检测效应**（配对差 Δ_s 的口径）")
    print("=" * 96)
    print(f"\n  {'族':<26}{'n':>3}{'均值':>10}{'SD':>10}{'SE':>9}"
          f"{'临界':>7}{'可检测':>10}")
    print("  " + "-" * 76)
    rows = []
    for name, runs in fams.items():
        v = family(runs)
        if len(v) < 2:
            continue
        n = len(v)
        m = statistics.mean(v)
        sd = statistics.stdev(v)
        se = sd / math.sqrt(n)
        crit = T.get(n - 1)
        if crit is None:
            continue
        detect = crit * se
        rows.append((name, n, m, sd, se, crit, detect))
        print(f"  {name:<26}{n:>3}{m:>10.4f}{sd:>10.4f}{se:>9.4f}"
              f"{crit:>7}{detect:>10.4f}")

    print("\n" + "=" * 96)
    print("判读")
    print("=" * 96)
    rows.sort(key=lambda r: r[6])
    print(f"\n  按可检测效应（越小 = 分辨率越细）：")
    for name, n, m, sd, se, crit, detect in rows:
        tag = "★ 能排除 0.012 以下的效应" if detect < 0.015 else (
            "⚠ 只能排除很大的效应" if detect > 0.10 else "")
        print(f"    {name:<26} 可检测 {detect:.4f}  {tag}")

    best = rows[0]
    worst = rows[-1]
    print(f"\n  ★ 最紧（{best[0]}）{best[6]:.4f}  vs  最松（{worst[0]}）{worst[6]:.4f}"
          f"   相差 {worst[6]/best[6]:.0f} 倍")
    print(f"\n  ⟹ 「单种子分辨率 ~0.035」是**跨种子 per-seed SD**的口径；")
    print(f"     配对之后 BC 族的实际门限是 **{best[6]:.3f}**，比它紧 3 倍。")
    print(f"     所以 BC 族的「测不出」**真的排除了 0.012 以上的效应**。")


if __name__ == "__main__":
    main()
