"""从零训（无 BC）n=9 —— 族级配对判读（正确判据）。

## 数据（u30，验证 regime，种子 100-114）

    scratch_s42 0.7246 | s43 0.5118 | s44 0.5284 | s45 0.6610 | s46 0.7024
    scratch_s47 0.7010 | s48 0.4858 | s49 0.5924 | s50 0.5997
    族均值 0.6119  SD 0.0899  n=9      专家 0.6979

## 本脚本回答三件事

1. **族级配对**（n=9，df=8，临界 2.306）：从零族 vs 专家、vs BC 族
2. **方差对比**：scratch 族 SD 0.0899 vs ent01_rerun 族 SD（≈0.009）
   —— 这是本节最硬的结论，不依赖任何显著性
3. ★ **配对只能取交集**：scratch 有 42-50，ent01_rerun 只有 42-46
   ⟹ 族间配对的 n = 5，**不是 9**。混用会静默错配。

⚠ 独立性限制：同族 9 条共享同一个起点类型（都从零），
  配对差 d_s 在种子间不完全独立；这是项目一贯口径。
"""
import json
import math
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = OUT / "eval" / "expert_seeds100_240.json"
T = {4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 14: 2.145}


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
            out[int(last)] = (list(ev["seeds"]), list(ev["per_seed_success"]))
    return out


def final(run):
    vp = valpts(run)
    if not vp:
        return None
    u = max(vp)
    if u < 30:
        return None
    return vp[u]


def paired_1samp(ds, crit_df=None):
    n = len(ds)
    m = statistics.mean(ds)
    sd = statistics.stdev(ds) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("inf")
    t = m / se if se > 0 else 0.0
    df = n - 1
    crit = T.get(df)
    return m, sd, t, df, crit


def main():
    exp = json.loads(EXPERT.read_text(encoding="utf-8"))
    e_seeds = [int(s) for s in exp["seeds"]]
    e_vals = [float(x) for x in (exp.get("success") or exp["per_seed_success"])]

    print("=" * 96)
    print("从零训（无 BC）n=9 —— 族级配对判读")
    print("=" * 96)
    print(f"\n  专家 = {statistics.mean(e_vals):.6f}\n")

    # ---------- 1. 从零族 vs 专家（n=9，用各自 per_seed） ----------
    sc_runs = [f"scratch_s{s}" for s in range(42, 51)]
    sc = {}
    for r in sc_runs:
        f = final(r)
        if f:
            sc[r] = f

    print(f"  【1】从零族 vs 专家 —— **逐臂**（仅供参考，不作结论）")
    print(f"  {'臂':<16}{'u30':>8}{'配对Δ':>10}{'t':>8}{'同向':>7}")
    ds9 = []
    for r, (seeds, vals) in sc.items():
        if list(seeds) != e_seeds:
            continue
        dd = [a - b for a, b in zip(vals, e_vals)]
        m, sd, t, df, crit = paired_1samp(dd)
        ds9 += dd
        print(f"  {r:<16}{statistics.mean(vals):>8.4f}{m:>+10.4f}{t:>+8.2f}"
              f"{sum(1 for x in dd if x>0):>4}/{len(dd)}")

    # 族级：把每臂的「对专家均值」当一条观测（n=9）
    arm_means = [statistics.mean(v) for _s, v in sc.values()]
    m9 = statistics.mean(arm_means) - statistics.mean(e_vals)
    sd9 = statistics.stdev(arm_means)
    se9 = sd9 / math.sqrt(len(arm_means))
    t9 = m9 / se9
    print(f"\n  ★【族级】从零族（n=9）均值 {statistics.mean(arm_means):.4f}"
          f"  vs 专家 {statistics.mean(e_vals):.4f}")
    print(f"      Δ = {m9:+.4f}   SD = {sd9:.4f}   SE = {se9:.4f}")
    print(f"      t = {t9:+.3f}  (df=8, 临界 {T[8]})   "
          f"{'★显著' if abs(t9) >= T[8] else '测不出'}")

    # ---------- 2. BC 族 ----------
    print(f"\n  【2】BC 族（ent01_rerun，n=5）")
    bc_means, bc_seeds_map = [], {}
    for s in range(42, 47):
        f = final(f"ent01_rerun_s{s}")
        if f:
            bc_means.append(statistics.mean(f[1]))
            bc_seeds_map[s] = f[1]
    m5 = statistics.mean(bc_means) - statistics.mean(e_vals)
    sd5 = statistics.stdev(bc_means)
    se5 = sd5 / math.sqrt(len(bc_means))
    t5 = m5 / se5
    print(f"      均值 {statistics.mean(bc_means):.4f}   Δ = {m5:+.4f}"
          f"   SD = {sd5:.4f}   t = {t5:+.3f} (df=4, 临界 {T[4]})"
          f"   {'★显著' if abs(t5) >= T[4] else '测不出'}")

    # ---------- 3. ★ 方差对比（本节最硬的结论） ----------
    print("\n" + "=" * 96)
    print("★ 方差对比 —— 不依赖显著性，是最可靠的一条")
    print("=" * 96)
    print(f"  {'族':<18}{'n':>3}{'均值':>10}{'SD':>9}{'范围':>20}")
    print(f"  {'从零（无BC）':<18}{len(arm_means):>3}{statistics.mean(arm_means):>10.4f}"
          f"{sd9:>9.4f}   [{min(arm_means):.4f}, {max(arm_means):.4f}]")
    print(f"  {'BC 暖启动':<18}{len(bc_means):>3}{statistics.mean(bc_means):>10.4f}"
          f"{sd5:>9.4f}   [{min(bc_means):.4f}, {max(bc_means):.4f}]")
    print(f"\n  ★ SD 之比 = {sd9/sd5:.1f}×")
    print(f"  ★ 范围宽度：从零 {max(arm_means)-min(arm_means):.4f}"
          f"  vs  BC {max(bc_means)-min(bc_means):.4f}")

    # ---------- 4. 同种子配对（交集 n=5） ----------
    print("\n" + "=" * 96)
    print("★ 同种子配对：从零 vs BC（**只能取交集 n=5**）")
    print("=" * 96)
    dd = []
    for s in range(42, 47):
        a = final(f"scratch_s{s}")
        b = final(f"ent01_rerun_s{s}")
        if a and b and list(a[0]) == list(b[0]):
            d = statistics.mean(a[1]) - statistics.mean(b[1])
            dd.append(d)
            print(f"    s{s}:  scratch {statistics.mean(a[1]):.4f}"
                  f"  BC {statistics.mean(b[1]):.4f}   Δ = {d:+.4f}")
    m, sd, t, df, crit = paired_1samp(dd)
    print(f"\n    Δ = {m:+.4f}   SD = {sd:.4f}   t = {t:+.3f}"
          f"  (df={df}, 临界 {crit})   {'★显著' if abs(t)>=crit else '测不出'}")

    # ---------- 5. 轨迹结构 ----------
    print("\n" + "=" * 96)
    print("轨迹结构：从零族是「慢」还是「分化」？")
    print("=" * 96)
    print(f"  {'臂':<16}{'u5':>8}{'u10':>8}{'u15':>8}{'u20':>8}{'u25':>8}{'u30':>8}{'末段':>9}")
    for r in sc_runs:
        vp = valpts(r)
        if not vp:
            continue
        us = sorted(vp)
        row = [statistics.mean(vp[u][1]) for u in us]
        tail = row[-1] - row[-2] if len(row) >= 2 else 0.0
        print(f"  {r:<16}" + "".join(f"{x:>8.4f}" for x in row) + f"{tail:>+9.4f}")
    print("\n  判读：末段仍在涨的 ⟹ 30 轮不够（慢）；末段掉/平的 ⟹ 分化")


if __name__ == "__main__":
    main()
