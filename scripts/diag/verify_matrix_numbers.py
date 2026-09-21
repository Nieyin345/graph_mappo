"""逐条复核 实验矩阵.md 主表的数字（防手抄错误）。"""
import json, statistics
from pathlib import Path
OUT = Path("/opt/qkd/graph_mappo/outputs")
def ev(run, u):
    p=OUT/run/"metrics.jsonl"
    if not p.exists(): return None
    last=None; v=None
    for line in p.read_text(encoding="utf-8",errors="replace").splitlines():
        line=line.strip()
        if not line: continue
        try: o=json.loads(line)
        except: continue
        if "update" in o: last=o["update"]
        if "eval_validation" in o and last==u:
            v=o["eval_validation"]["mean_success_rate"]
    return v
def verdict(ef, seeds, cf="ent01_rerun_s{}"):
    ds=[]
    for s in seeds:
        a,b=ev(ef.format(s),25),ev(ef.format(s),30)
        c,d=ev(cf.format(s),25),ev(cf.format(s),30)
        if None in (a,b,c,d): continue
        ds.append((a+b)/2-(c+d)/2)
    m=statistics.mean(ds); sd=statistics.stdev(ds); se=sd/len(ds)**0.5
    df=len(ds)-1
    crit={2:4.303,4:2.776,8:2.306}.get(df,2.776)
    return len(ds), m, sd, m/se, crit, crit*se

print("="*88)
print("矩阵表数字复算（应与 docs/实验矩阵.md 逐位相符）")
print("="*88)
ROWS=[("#1 demand_edge","demandedge_s{}",range(42,51)),
      ("#2 pool_include_max","cmax_s{}",range(42,47)),
      ("#3 num_layers","layers4_s{}",range(42,47)),
      ("#4 storage10x","storage5_s{}",range(42,47)),
      ("#5 residual","residfalse_s{}",range(42,47))]
for name,ef,seeds in ROWS:
    r=verdict(ef,seeds)
    print(f"  {name:<22} n={r[0]}  Δ={r[1]:+.4f}  SD={r[2]:.4f}  t={r[3]:+.2f}"
          f"  crit={r[4]:.3f}  可检测={r[5]:.4f}")

# #6：从零（scratch）vs BC
r=verdict("scratch_s{}", range(42,51))
print(f"  {'#6 scratch vs BC':<22} n={r[0]}  Δ={r[1]:+.4f}  SD={r[2]:.4f}  t={r[3]:+.2f}"
      f"  crit={r[4]:.3f}  可检测={r[5]:.4f}")
