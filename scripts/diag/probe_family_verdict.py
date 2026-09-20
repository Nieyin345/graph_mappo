"""★ 正确判据：**族级** n 种子配对 vs 专家（不是逐臂挑最好的）。

## 为什么必须这样

逐臂看，`scratch_s42` Δ=+0.0266 (t=3.55)「显著」。但：
  · scratch 族均值 = −0.1096（比专家差 11 点）
  · 族内 SD = 0.118（是 ent01 族的 14 倍）
⟹ **从高方差族里挑最好的种子，必然挑出一个显著为正的数。**
   这是 `selection-bias-max-of-k`，我今天已经踩过。

## 正确做法

一族 = 一个配置，其 **n 个训练种子全用**，与专家**逐种子配对**，做**单样本 t**。
这就是 `scripts/diag/paired_verdict.py` 的逻辑，也是项目预注册的判据。

⚠ 独立性限制：同族的 n 条臂共享**同一个 BC 起点**（或同为从零），
  配对差 d_s 在种子间**不完全独立**，但这是项目一贯口径。
"""
import json
import math
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = OUT / "eval" / "expert_seeds100_240.json"
T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}


def valpts(run):
    p = OUT / run / "metrics.jsonl"
    out = {}
    if not p.exists():
        return out
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
        if "eval_validation" in o and last is not None:
            ev = o["eval_validation"]
            out[int(last)] = (ev["seeds"], ev["per_seed_success"])
    return out


def main():
    exp = json.loads(EXPERT.read_text(encoding="utf-8"))
    e_seeds = [int(s) for s in exp["seeds"]]
    e_vals = [float(x) for x in (exp.get("success") or exp["per_seed_success"])]

    fams = {}
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if "_s" not in name:
            continue
        fam, _, s = name.rpartition("_s")
        if not s.isdigit():
            continue
        vp = valpts(name)
        if not vp:
            continue
        u = max(vp)
        if u < 30:
            continue
        seeds, vals = vp[u]
        if list(seeds) != e_seeds:
            continue
        dd = [a - b for a, b in zip(vals, e_vals)]
        fams.setdefault(fam, []).append((int(s), statistics.mean(dd)))

    print("=" * 96)
    print("★ 族级配对 vs 专家（每族用它的**全部**种子，不挑）")
    print("=" * 96)
    print(f"\n  专家 = {statistics.mean(e_vals):.6f}\n")
    print(f"  {'配置族':<18}{'n':>3}{'族均值Δ':>11}{'SD':>9}{'t':>8}{'df':>4}"
          f"{'临界':>7}  判读")
    print("  " + "-" * 76)

    results = []
    for fam, items in sorted(fams.items()):
        ds = [d for _, d in items]
        n = len(ds)
        if n < 2:
            print(f"  {fam:<18}{n:>3}{statistics.mean(ds):>+11.4f}"
                  f"{'—':>9}{'—':>8}{'—':>4}{'—':>7}  n<2 不判")
            continue
        m = statistics.mean(ds)
        sd = statistics.stdev(ds)
        se = sd / math.sqrt(n)
        t = m / se if se > 0 else 0.0
        df = n - 1
        crit = T.get(df)
        if crit is None:
            verd = f"df={df} 无临界值⟹抛错"
        else:
            verd = "★显著为正" if t >= crit else (
                "★显著为负" if t <= -crit else "测不出")
        print(f"  {fam:<18}{n:>3}{m:>+11.4f}{sd:>9.4f}{t:>+8.2f}{df:>4}"
              f"{crit if crit else '—':>7}  {verd}")
        results.append((fam, n, m, sd, t, df, crit, verd))

    print("\n" + "=" * 96)
    print("排序（按族均值）")
    print("=" * 96)
    for fam, n, m, sd, t, df, crit, verd in sorted(results, key=lambda x: -x[2]):
        print(f"  {fam:<18} Δ={m:+.4f}  SD={sd:.4f}  t={t:+.2f}  ({verd})")

    print("\n  ★ 关键：**没有任何一族的 t 超过它的临界值**"
          if all(r[4] < r[6] for r in results if r[6] is not None) else
          "\n  ★ 有族显著")
    print("    ⟹ 「RL 超过专家」在当前 n 下**测不出**。")
    print("    ⟹ 逐臂挑选（我上一版做的）是 max-of-k 选择偏差。")


if __name__ == "__main__":
    main()
