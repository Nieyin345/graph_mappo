import json, statistics
from pathlib import Path
OUT = Path("/opt/qkd/graph_mappo/outputs")

def rows(run):
    out = []
    for line in (OUT/run/"metrics.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line: continue
        try: o = json.loads(line)
        except: continue
        if "update" in o: out.append(o)
    return out

def vals(run):
    out, last = [], None
    for line in (OUT/run/"metrics.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line: continue
        try: o = json.loads(line)
        except: continue
        if "update" in o: last = o["update"]
        if "eval_validation" in o: out.append((last, o["eval_validation"]["mean_success_rate"]))
    return out

print("="*90)
print("训练侧体征（末轮）")
print("="*90)
print(f"  {'臂':<22}{'kl':>10}{'entropy':>9}{'clip':>9}{'train_sr':>10}")
for pre in ("demandedge","ent01_rerun"):
    for s in range(47,51):
        r = f"{pre}_s{s}"
        rs = rows(r)
        if not rs: continue
        o = rs[-1]
        def f(x,w=9,p=5): return f"{x:>{w}.{p}f}" if isinstance(x,(int,float)) else f"{'—':>{w}}"
        print(f"  {r:<22}{f(o.get('kl'),10)}{f(o.get('entropy'))}{f(o.get('clip_frac'))}{f(o.get('mean_success_rate'),10)}")

print()
print("="*90)
print("验证侧曲线")
print("="*90)
for pre in ("demandedge","ent01_rerun"):
    print(f"\n  【{pre}】")
    means = []
    for s in range(47,51):
        r = f"{pre}_s{s}"
        v = vals(r)
        if not v: continue
        print(f"    {r:<22}{[round(x[1],3) for x in v]}")
        means.append(statistics.mean([x[1] for x in v][-2:]))
    if means:
        print(f"    末两点均值 {statistics.mean(means):.4f} 范围[{min(means):.4f},{max(means):.4f}]")
