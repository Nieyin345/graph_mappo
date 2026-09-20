"""全部臂 vs 专家 —— **配对**（均值比不配对是不可比的）。

⚠ 上一版我犯了和今早同一个错：看到 scratch_s42=0.7246 > 专家均值 0.6979
就宣布「首个超过专家」。但表里十几条臂的 u30 均值都 > 0.6979。
均值比**不配对**，而 `docs/测试规范.md` §4 明写不配对时 SE≈0.062。
必须逐种子配对，用 paired_verdict 的同一套逻辑。
"""
import json
import math
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = OUT / "eval" / "expert_seeds100_240.json"
T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
     7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 14: 2.145}


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
    e_mean = statistics.mean(e_vals)

    print("=" * 96)
    print("全部臂 vs 专家 —— **逐种子配对**（u30，种子 100-114）")
    print("=" * 96)
    print(f"\n  专家均值 = {e_mean:.6f}   n={len(e_vals)}")
    print(f"  临界值(df=14) = {T[14]}\n")

    rows = []
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        vp = valpts(d.name)
        if not vp:
            continue
        u = max(vp)
        if u < 30:
            continue
        seeds, vals = vp[u]
        if list(seeds) != e_seeds:
            continue
        dd = [a - b for a, b in zip(vals, e_vals)]
        n = len(dd)
        m = statistics.mean(dd)
        sd = statistics.stdev(dd) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 1 else float("inf")
        t = m / se if se > 0 else 0.0
        rows.append((d.name, statistics.mean(vals), m, t, n - 1,
                     sum(1 for x in dd if x > 0)))

    rows.sort(key=lambda r: -r[2])
    print(f"  {'臂':<24}{'u30均值':>9}{'配对Δ':>10}{'t':>8}{'df':>4}{'同向':>7}  判读")
    print("  " + "-" * 78)
    for name, mean, m, t, df, same in rows:
        crit = T.get(df)
        if crit is None:
            verd = f"df={df} 无临界值⟹抛错"
        else:
            verd = "★显著为正" if t >= crit else (
                "★显著为负" if t <= -crit else "测不出")
        print(f"  {name:<24}{mean:>9.4f}{m:>+10.4f}{t:>+8.2f}{df:>4}{same:>4}/{len(e_vals)}"
              f"  {verd}")

    sig = [r for r in rows if T.get(r[4]) and r[3] >= T[r[4]]]
    print(f"\n  ★ 配对后**显著超过专家**的臂：{len(sig)} 条")
    for name, mean, m, t, df, same in sig:
        print(f"      {name:<24} Δ={m:+.4f}  t={t:+.2f}  ({same}/15 同向)")


if __name__ == "__main__":
    main()
