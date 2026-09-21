"""9 个对照臂的 u30 分布：+0.0214 的两批差是否在种子的正常波动内？

看每臂 u30 的**验证侧**逐种子 SD —— 若单臂内部的种子间波动就有 0.05+，
那不同训练种子之间差 0.021 完全正常。
"""
import json, statistics
from pathlib import Path
OUT = Path("/opt/qkd/graph_mappo/outputs")
def per_seed(run):
    out, last = None, None
    for line in (OUT/run/"metrics.jsonl").read_text(encoding="utf-8",errors="replace").splitlines():
        line=line.strip()
        if not line: continue
        try: o=json.loads(line)
        except: continue
        if "update" in o: last=o["update"]
        if "eval_validation" in o and last==30:
            out=o["eval_validation"]["per_seed_success"]
    return out

print("="*90)
print("9 个对照臂的 u30")
print("="*90)
means=[]; within_sd=[]
for s in range(42,51):
    v=per_seed(f"ent01_rerun_s{s}")
    if v:
        means.append(statistics.mean(v)); within_sd.append(statistics.stdev(v))
        print(f"  ent01_rerun_s{s}: 均值 {statistics.mean(v):.4f}  臂内种子 SD {statistics.stdev(v):.4f}")
print()
print(f"  ★ 跨训练种子 SD（9 个臂均值）= {statistics.stdev(means):.4f}")
print(f"  ★ 臂内验证种子 SD（平均）    = {statistics.mean(within_sd):.4f}")
print(f"  ★ 两批差 +0.0214 / 跨臂 SD {statistics.stdev(means):.4f} = "
      f"{0.0214/statistics.stdev(means):.2f} 个 SD")
print(f"\n  判读：两批差 = {0.0214/statistics.stdev(means):.2f}×跨臂SD")
print(f"  在 {len(means)} 个种子里，最大最小差 = {max(means)-min(means):.4f}")
