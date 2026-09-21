"""所有族在 u25 / u30 / 均值 三个窗口下的判决 —— 看"早负晚正"是否普遍。"""
import json, statistics
from pathlib import Path
OUT = Path("/opt/qkd/graph_mappo/outputs")
def pts(run):
    out, last = {}, None
    p = OUT/run/"metrics.jsonl"
    if not p.exists(): return out
    for line in p.read_text(encoding="utf-8",errors="replace").splitlines():
        line=line.strip()
        if not line: continue
        try: o=json.loads(line)
        except: continue
        if "update" in o: last=o["update"]
        if "eval_validation" in o and last is not None:
            out[last]=o["eval_validation"]["mean_success_rate"]
    return out

FAMS = [("demand_edge","demandedge_s{}",range(42,51)),
        ("cmax","cmax_s{}",range(42,47)),
        ("layers4","layers4_s{}",range(42,47)),
        ("storage5","storage5_s{}",range(42,47)),
        ("residfalse","residfalse_s{}",range(42,47))]
print("="*96)
print("窗口敏感性：三个窗口下的 Δ / t")
print("="*96)
print(f"\n  {'族':<14}{'n':>3}  {'u25':>16}  {'u30':>16}  {'u25+u30均值':>16}")
print(f"  {'':<14}{'':>3}  {'Δ':>8}{'t':>8}  {'Δ':>8}{'t':>8}  {'Δ':>8}{'t':>8}")
print("  "+"-"*80)
for name, ef, seeds in FAMS:
    rows=[]
    for s in seeds:
        a=pts(ef.format(s)); b=pts(f"ent01_rerun_s{s}")
        if 25 in a and 25 in b and 30 in a and 30 in b:
            rows.append((a[25]-b[25], a[30]-b[30]))
    if len(rows)<2:
        print(f"  {name:<14}{len(rows):>3}  (数据不足)")
        continue
    def st(idx):
        d=[r[idx] for r in rows]
        m=statistics.mean(d); sd=statistics.stdev(d); se=sd/len(d)**0.5
        return m, m/se if se else 0
    d25=[r[0] for r in rows]; d35=[r[1] for r in rows]
    dm=[(r[0]+r[1])/2 for r in rows]
    out=[]
    for d in (d25, d35, dm):
        m=statistics.mean(d); sd=statistics.stdev(d); se=sd/len(d)**0.5
        out.append((m, m/se if se else 0))
    n=len(rows)
    crit = {5:2.776, 9:2.306}.get(n, 2.776)
    def mark(t): return "*" if abs(t)>=crit else " "
    print(f"  {name:<14}{n:>3}  {out[0][0]:>+8.4f}{out[0][1]:>+7.2f}{mark(out[0][1])} "
          f" {out[1][0]:>+8.4f}{out[1][1]:>+7.2f}{mark(out[1][1])} "
          f" {out[2][0]:>+8.4f}{out[2][1]:>+7.2f}{mark(out[2][1])}")
print(f"\n  （* = 该窗口下超过临界值；n=5 临界 2.776，n=9 临界 2.306）")
