#!/usr/bin/env python
"""扫所有 run 的 eval 轨迹：这个算法到底有没有在学？

只读 metrics.jsonl，不加载 checkpoint。输出每个 run 的 eval 序列、
首末值、以及训练侧成功率（update 行的 mean_success_rate）。
"""
import json
from pathlib import Path

MAIN = Path("/opt/qkd/graph_mappo")
OUT = MAIN / "outputs"


def read_run(d):
    rows, evals = [], []
    p = d / "metrics.jsonl"
    if not p.exists():
        return rows, evals
    for line in p.open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(o, dict):
            continue
        if "eval_validation" in o:
            e = o["eval_validation"]
            evals.append((o.get("update"), float(e.get("mean_success_rate", 0.0))))
        elif "update" in o:
            rows.append(o)
    return rows, evals


runs = sorted([d for d in OUT.iterdir() if d.is_dir() and (d / "metrics.jsonl").exists()])
print(f"共 {len(runs)} 个 run\n")

hdr = f"{'run':<34} {'轮':>3} {'训练succ首/末':>16} {'eval 轨迹':<44} {'Δeval':>8}"
print(hdr)
print("-" * len(hdr))

for d in runs:
    rows, evals = read_run(d)
    if not rows and not evals:
        continue
    n = len(rows)
    t0 = rows[0].get("mean_success_rate") if rows else None
    t1 = rows[-1].get("mean_success_rate") if rows else None
    tstr = f"{t0:.3f}/{t1:.3f}" if t0 is not None and t1 is not None else "  --"
    eseq = " ".join(f"{v:.3f}" for _u, v in evals)
    if len(eseq) > 44:
        eseq = eseq[:41] + "..."
    if len(evals) >= 2:
        dlt = f"{evals[-1][1] - evals[0][1]:+.3f}"
    else:
        dlt = "   --"
    name = d.name if len(d.name) <= 34 else d.name[:31] + "..."
    print(f"{name:<34} {n:>3} {tstr:>16} {eseq:<44} {dlt:>8}")
