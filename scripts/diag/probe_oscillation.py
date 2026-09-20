"""BC 族续跑 vs 从零族续跑：**验证侧震荡幅度**的对比。

## 核心问题（检查点 2）

从零族续跑后剧烈震荡（§十三）。BC 族续跑是：**平稳**还是**也震荡**？

- BC 族**平稳** ⟹ 再次印证「BC 起点降方差」，震荡是从零族的性质
- BC 族**也震荡** ⟹ 震荡源在**验证协议**（`random_day` 起日随机），不在训练

## 判据：同一窗口内

两侧都用 u30 之后的点（续跑段），量**逐点的变化幅度**：
  · `SD(相邻点差)` —— 震荡的绝对幅度
  · `range` —— 最大最小差
⚠ 必须**同长度同窗口**比，否则是不同口径（`verdict-window-is-part-of-the-claim`）。
"""
import json
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def curve(run, lo=30):
    p = OUT / run / "metrics.jsonl"
    us, vals, last = [], [], None
    if not p.exists():
        return [], []
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
            if last > lo:
                us.append(last)
        if "eval_validation" in o and last is not None and last > lo:
            vals.append(o["eval_validation"]["mean_success_rate"])
    return us, vals


def stats(vals):
    if len(vals) < 2:
        return None
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "sd": statistics.stdev(vals),
        "jitter": statistics.mean(abs(d) for d in diffs),
        "range": max(vals) - min(vals),
    }


def main():
    print("=" * 92)
    print("续跑段（u>30）验证侧震荡幅度对比")
    print("=" * 92)

    for label, runs in (("从零族（scratch）", [f"scratch_s{s}" for s in (42, 43, 44)]),
                        ("BC 族（ent01_rerun）", [f"ent01_rerun_s{s}" for s in (42, 43, 44, 45)])):
        print(f"\n  【{label}】")
        print(f"    {'臂':<18}{'n':>4}{'均值':>9}{'SD':>8}{'平均跳幅':>10}{'极差':>9}")
        allv = []
        for r in runs:
            us, vals = curve(r)
            if not vals:
                continue
            s = stats(vals)
            allv += vals
            print(f"    {r:<18}{s['n']:>4}{s['mean']:>9.4f}{s['sd']:>8.4f}"
                  f"{s['jitter']:>10.4f}{s['range']:>9.4f}")
            print(f"    {'':<18}  点: {[round(v, 3) for v in vals]}")
        if allv:
            s = stats(allv)
            print(f"    {'合并':<18}{s['n']:>4}{s['mean']:>9.4f}{s['sd']:>8.4f}"
                  f"{s['jitter']:>10.4f}{s['range']:>9.4f}")


if __name__ == "__main__":
    main()
