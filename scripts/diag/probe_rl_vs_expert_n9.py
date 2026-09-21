"""★ 核心问题：RL 离专家到底多远？—— 用现有全部对照组（n=9）+ 15 个验证种子。

## 为什么现在就能答

「RL vs 专家」的配对单位是**验证种子**（15 个），不是训练种子。
每个训练种子的臂都在**同一批 15 个验证种子**上评过 ⟹ 每臂给 15 个配对点。

本脚本：
  ① 逐训练种子：该臂 u30 vs 专家的配对 Δ 与 t
  ② 合并全部 9 条臂的 135 个配对点（**注意**：同臂内的 15 个点共享
     同一策略 ⟹ 不是独立样本，合并的 t 会虚高 —— 只作参考，
     主要看**逐臂**分布与**族级**均值）
"""
import json, math, statistics
from pathlib import Path
OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = OUT/"eval"/"expert_seeds100_240.json"

def per_seed(run, u=30):
    p = OUT/run/"metrics.jsonl"
    if not p.exists(): return None
    last=None; out=None
    for line in p.read_text(encoding="utf-8",errors="replace").splitlines():
        line=line.strip()
        if not line: continue
        try: o=json.loads(line)
        except: continue
        if "update" in o: last=o["update"]
        if "eval_validation" in o and last==u:
            ev=o["eval_validation"]
            out=(list(ev["seeds"]), list(ev["per_seed_success"]))
    return out

exp = json.loads(EXPERT.read_text(encoding="utf-8"))
e_seeds=[int(s) for s in exp["seeds"]]
e_vals=[float(x) for x in (exp.get("success") or exp["per_seed_success"])]
e_mean=statistics.mean(e_vals)
print("="*94)
print(f"RL vs 专家（u30，15 个验证种子配对）   专家均值 = {e_mean:.6f}")
print("="*94)
print(f"\n  {'臂':<20}{'u30均值':>10}{'配对Δ':>10}{'t':>8}{'同向':>8}")
print("  "+"-"*58)
arm_deltas=[]; arm_means=[]
for s in range(42,51):
    r=f"ent01_rerun_s{s}"
    ps=per_seed(r)
    if not ps: continue
    seeds, vals = ps
    if list(seeds)!=e_seeds:
        print(f"  {r:<20} 种子表不匹配，跳过"); continue
    d=[a-b for a,b in zip(vals,e_vals)]
    m=statistics.mean(d); sd=statistics.stdev(d); se=sd/math.sqrt(len(d))
    arm_deltas.append(m); arm_means.append(statistics.mean(vals))
    print(f"  {r:<20}{statistics.mean(vals):>10.4f}{m:>+10.4f}{m/se:>+8.2f}"
          f"{sum(1 for x in d if x>0):>5}/15")

print(f"\n  ★ 族级（n={len(arm_deltas)} 个训练种子）")
m=statistics.mean(arm_deltas); sd=statistics.stdev(arm_deltas)
se=sd/math.sqrt(len(arm_deltas))
crit={5:2.776,9:2.306}.get(len(arm_deltas),2.776)
print(f"    臂均值 {statistics.mean(arm_means):.6f}  vs  专家 {e_mean:.6f}")
print(f"    Δ = {m:+.6f}   SD(跨训练种子) = {sd:.4f}   SE = {se:.4f}")
print(f"    t = {m/se:+.3f}  (df={len(arm_deltas)-1}, 临界 {crit})")
print(f"    ⟹ {'★显著' if abs(m/se)>=crit else '测不出'}"
      f"   **可检测效应 = {crit*se:.4f}**")
