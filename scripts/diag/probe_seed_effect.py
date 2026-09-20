"""关键检验：4 条「显著超专家」的臂，是真信号还是**种子选择效应**？

## 观察

4 条显著为正的臂里，**3 条的种子里含 s43**（pm_decode_s43 / v1_onpath_s43 /
scratch 族最高的也是 s42/s43）。而每族的 s42/s44 常常显著为负。

⟹ 假说：**训练种子 43 恰好是个「好种子」**，任何配置配它都容易超专家。

## 判据（这是可分辨的）

若存在「不管什么配置，s43 都最好」的模式 ⟹ 是种子效应，不是配置效应。

做法：按**训练种子**分组，看各配置在该种子上的 Δ vs 专家。
若 s43 那一列整体右移（跨所有配置），则是种子效应。
若只有**特定配置**的 s43 高，则是配置×种子的交互（更可能是真信号）。
"""
import json
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = OUT / "eval" / "expert_seeds100_240.json"


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

    # 按「配置族 + 训练种子」收集 Δ vs 专家
    fams = {}
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        # 解析族与种子
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
        fams.setdefault(fam, {})[int(s)] = statistics.mean(dd)

    seeds = sorted({s for f in fams.values() for s in f})
    print("=" * 96)
    print("按训练种子分组：每格 = 该臂 u30 对专家的配对 Δ")
    print("=" * 96)
    print(f"\n  {'配置族':<22}" + "".join(f"{'s'+str(s):>9}" for s in seeds)
          + f"{'族均值':>10}")
    print("  " + "-" * (22 + 9 * len(seeds) + 10))
    for fam in sorted(fams):
        row = fams[fam]
        cells = "".join(f"{row[s]:>+9.4f}" if s in row else f"{'—':>9}"
                        for s in seeds)
        m = statistics.mean(row.values()) if row else float("nan")
        print(f"  {fam:<22}{cells}{m:>+10.4f}")

    # 每列（种子）的均值 ⟹ 看是否有「好种子」
    print("\n  " + "-" * (22 + 9 * len(seeds) + 10))
    colmeans = []
    for s in seeds:
        vals = [fams[f][s] for f in fams if s in fams[f]]
        colmeans.append(statistics.mean(vals) if vals else float("nan"))
    print(f"  {'【按种子均值】':<20}" + "".join(f"{v:>+9.4f}" for v in colmeans))

    print("\n  判读：")
    print("    若某些种子列整体右移 ⟹ **种子效应**（该种子配什么都好）")
    print("    若只有个别格高、列均值平 ⟹ **配置×种子交互**（更可能是真信号）")

    # 每族内的标准差（族内方差 = 种子效应的大小）
    print("\n  各族内部跨种子 SD（大 ⟹ 该配置对种子极敏感）")
    for fam in sorted(fams):
        v = list(fams[fam].values())
        if len(v) >= 2:
            print(f"    {fam:<22} mean={statistics.mean(v):+.4f}  "
                  f"SD={statistics.stdev(v):.4f}  n={len(v)}")


if __name__ == "__main__":
    main()
