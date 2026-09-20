"""clean vs scratch：**为什么加种子没帮上忙**，以及一个可能更可测的统计量。

## 现象

n=5 → n=8，SD 从 0.0470 涨到 **0.1986**，可检测效应反而变差。
原因：**两个族的方差都极大**（都是从零训），加进来的是两个极端点：
  s49: clean 0.1241 vs scratch 0.6114  ⟹ Δ=−0.4873（灾难性崩塌）
  s48: clean 0.6593 vs scratch 0.4449  ⟹ Δ=+0.2144

## 这暴露了一个实验设计问题

`clean vs scratch` **两侧都是从零训** ⟹ 两侧都带高方差
⟹ 这个对比**在原理上就很难测**（不像 `cmax vs ent01_rerun` 两侧都是 BC 族，
可检测效应只有 0.0081）。

## 本脚本问的第二个问题

均值测不出时，**分布形状**可能仍有信号：
从零训是否会**周期性崩塌**（某个种子学到坏解）？
  · 若 clean 的"崩塌率"（sr < 0.3）与 scratch 显著不同 ⟹ 那才是可测的量
  · 若两者分布相似 ⟹ 先验特征对**稳定性**都无影响
"""
import json
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def final_sr(run, u=30):
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        return None
    last = None
    val = None
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
        if "eval_validation" in o and last == u:
            val = o["eval_validation"]["mean_success_rate"]
    return val


def main():
    fams = {
        "scratch（全特征，从零）": [f"scratch_s{s}" for s in range(42, 51)],
        "clean（无先验特征，从零）": [f"clean_s{s}" for s in range(42, 51)],
        "ent01_rerun（BC 起点）": [f"ent01_rerun_s{s}" for s in range(42, 47)],
        "cmax（BC 起点+max池化）": [f"cmax_s{s}" for s in range(42, 47)],
    }

    print("=" * 92)
    print("各族 u30 的**分布**（不只看均值）")
    print("=" * 92)
    print(f"\n  {'族':<26}{'n':>3}{'均值':>9}{'SD':>9}{'最小':>9}{'最大':>9}"
          f"{'崩塌(<0.3)':>11}")
    print("  " + "-" * 78)
    for name, runs in fams.items():
        vals = [v for v in (final_sr(r) for r in runs) if v is not None]
        if not vals:
            continue
        n = len(vals)
        crash = sum(1 for v in vals if v < 0.3)
        print(f"  {name:<26}{n:>3}{statistics.mean(vals):>9.4f}"
              f"{statistics.stdev(vals) if n > 1 else 0:>9.4f}"
              f"{min(vals):>9.4f}{max(vals):>9.4f}{crash:>7}/{n:<3}")

    print("\n" + "=" * 92)
    print("逐臂 u30（看有没有崩塌的）")
    print("=" * 92)
    for name, runs in fams.items():
        vals = [(r, final_sr(r)) for r in runs]
        vals = [(r, v) for r, v in vals if v is not None]
        print(f"\n  {name}")
        print("    " + "  ".join(f"{r.split('_')[-1]}={v:.3f}" for r, v in vals))

    print("\n" + "=" * 92)
    print("统计检验：clean vs scratch 的**分布是否不同**")
    print("=" * 92)
    a = [v for v in (final_sr(f"scratch_s{s}") for s in range(42, 51)) if v is not None]
    b = [v for v in (final_sr(f"clean_s{s}") for s in range(42, 51)) if v is not None]
    print(f"\n  scratch: n={len(a)} 均值 {statistics.mean(a):.4f} "
          f"SD {statistics.stdev(a):.4f}")
    print(f"  clean:   n={len(b)} 均值 {statistics.mean(b):.4f} "
          f"SD {statistics.stdev(b):.4f}")

    # 崩塌率检验（Fisher 精确检验的两比例版本，手算）
    ca = sum(1 for v in a if v < 0.3)
    cb = sum(1 for v in b if v < 0.3)
    print(f"\n  崩塌率（sr<0.3）：scratch {ca}/{len(a)}  clean {cb}/{len(b)}")
    if len(a) and len(b):
        # 两比例 z 检验（小样本，只作参考）
        pa, pb = ca / len(a), cb / len(b)
        p = (ca + cb) / (len(a) + len(b))
        se = (p * (1 - p) * (1 / len(a) + 1 / len(b))) ** 0.5
        z = (pb - pa) / se if se > 0 else 0.0
        print(f"    两比例 z = {z:+.2f}  （|z|<1.96 ⟹ 看不出差别）")


if __name__ == "__main__":
    main()
