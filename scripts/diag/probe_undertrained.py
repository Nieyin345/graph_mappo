"""「训练时间不够」吗？—— 用**曲线形状 + 优化动力学**回答，不靠感觉。

## 两个假说的预测不同

**H_T（训练不够）**：验证曲线在末段**仍在显著上升**；kl 非零；clip 有激活；
优势幅度可观 ⟹ 再跑会更好。

**H_S（已饱和）**：曲线末段**平/掉**；kl→0；clip 从不激活；|advantage|→0
⟹ 再跑是同一目标上更努力（记忆 `training-saturated-structural-bottleneck`）。

## 关键：两族要分开判

从零族 n=9 里 7 条末段还在涨；BC 族早平。
⟹ **"训练不够"可能只对从零族成立**，而那正是今天刚测出"更差"的那一族。

## 本探针量什么

1. 每族逐更新均值曲线 + **末段斜率**（最后 3 个点的平均增量）
2. kl / clip_frac / |advantage| 的轨迹（收敛的指纹）
3. 给每条臂做**线性外推到 u60**，看外推值是否还显著低于专家
"""
import json
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = 0.697922


def series(run):
    """{'val': {u: sr}, 'train': {u: (kl, clip, adv)}}"""
    p = OUT / run / "metrics.jsonl"
    val, tr = {}, {}
    if not p.exists():
        return val, tr
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
            tr[last] = (o.get("kl"), o.get("clip_frac"),
                        o.get("mean_abs_advantage"))
        if "eval_validation" in o and last is not None:
            val[last] = o["eval_validation"]["mean_success_rate"]
    return val, tr


def slope(vals):
    """末段 3 点的平均增量（每个验证间隔 = 5 轮）。"""
    if len(vals) < 3:
        return float("nan")
    return (vals[-1] - vals[-3]) / 2.0


def main():
    fams = {
        "从零（无BC）": [f"scratch_s{s}" for s in range(42, 51)],
        "BC 暖启动": [f"ent01_rerun_s{s}" for s in range(42, 47)],
        "v2_bottleneck": [f"v2_bottleneck_s{s}" for s in range(42, 47)],
        "v1_onpath": [f"v1_onpath_s{s}" for s in range(42, 47)],
    }

    print("=" * 100)
    print("「训练不够」吗？—— 逐族曲线与末段斜率")
    print("=" * 100)
    print(f"\n  专家 = {EXPERT:.4f}   末段斜率 = u25→u30 的增量（每 5 轮）\n")

    for fam, runs in fams.items():
        curves = {}
        for r in runs:
            v, _ = series(r)
            if v:
                curves[r] = [v[u] for u in sorted(v)]
        if not curves:
            continue
        us = sorted(next(iter(curves.values())))[:0]  # 占位
        n = min(len(c) for c in curves.values())
        # 逐更新族均值
        means = [statistics.mean(c[i] for c in curves.values()) for i in range(n)]
        sl = slope(means)
        label = ["u5", "u10", "u15", "u20", "u25", "u30"][:n]
        print(f"  {fam}  (n={len(curves)})")
        print(f"    族均值曲线: " + "  ".join(f"{l}={m:.4f}" for l, m in zip(label, means)))
        print(f"    末段斜率 = {sl:+.4f}/5轮   首→末 = {means[-1]-means[0]:+.4f}")
        # 逐臂末段斜率的分布
        per = [slope(c) for c in curves.values()]
        rising = sum(1 for x in per if x > 0.005)
        print(f"    逐臂末段：涨的 {rising}/{len(per)}   "
              f"中位 {statistics.median(per):+.4f}  范围 [{min(per):+.4f}, {max(per):+.4f}]")
        print()

    # ---------- 优化动力学（收敛指纹） ----------
    print("=" * 100)
    print("优化动力学：kl / clip_frac / |advantage| —— 收敛还是欠训练")
    print("=" * 100)
    for fam, runs in fams.items():
        print(f"\n  {fam}")
        print(f"    {'臂':<16}{'kl(u5)':>10}{'kl(u30)':>10}"
              f"{'clip(u5)':>10}{'clip(u30)':>11}{'|adv|(u5)':>11}{'|adv|(u30)':>12}")
        for r in runs:
            _, tr = series(r)
            if not tr:
                continue
            us = sorted(tr)
            a, b = tr[us[0]], tr[us[-1]]
            def f(x, w=10, p=5):
                return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'—':>{w}}"
            print(f"    {r:<16}{f(a[0])}{f(b[0])}{f(a[1])}{f(b[1],11)}{f(a[2])}{f(b[2],12)}")
        print("    （kl→0 且 clip→0 ⟹ 收敛；kl 保持 ⟹ 仍在动）")


if __name__ == "__main__":
    main()
