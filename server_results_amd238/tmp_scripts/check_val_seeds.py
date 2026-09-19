#!/usr/bin/env python
"""检查验证点用的是不是同一批种子、以及每点几个实例。

动机：s42 的 u35/u40/u45 = 0.6950/0.7045/0.6951，其中 u35 与 u45 几乎相同。
需要排除"验证机制本身"造成的交替（种子集变、实例数变），
再谈"是不是训练动力学"。
"""
import json
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")

for run in ("ent01_s42_u30to50", "ent01_s43_u30to50", "ent01_s44_u30to50"):
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        print(f"--- {run}: 无文件")
        continue
    print(f"--- {run}")
    u = None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "update" in r:
            u = r["update"]
        ev = r.get("eval_validation")
        if isinstance(ev, dict):
            ps = ev.get("per_seed_success") or []
            seeds = ev.get("seeds") or []
            mean = ev.get("mean_success_rate")
            print(f"  u{u}: n={len(ps)} seeds={seeds}")
            print(f"        per_seed={[round(float(x), 4) for x in ps]}")
            print(f"        mean={mean}")
    print()
