#!/usr/bin/env bash
# 看两线进度：每个 run 的轮数、最后一次验证、验证序列。
# 单独一个脚本是因为从 PowerShell 经 ssh 内联下发带引号的命令会被吞掉引号
# （`for f in outputs/r3_*/` + `python3 -c "..."` 实测 syntax error）。
set -u
cd /opt/qkd/graph_mappo || exit 1
exec python3 - <<'PY'
import json
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")

for pat in ("r3_*", "real_*"):
    print(f"===== {pat} =====")
    for d in sorted(ROOT.glob(f"outputs/{pat}")):
        p = d / "metrics.jsonl"
        if not p.exists():
            print(f"  {d.name:<18} (无 metrics)")
            continue
        tr, ev = [], []
        for line in p.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "update" in r:
                tr.append(r)
            elif "eval_validation" in r:
                ev.append(r["eval_validation"])
        series = " ".join(f"{e['mean_success_rate']:.4f}" for e in ev) or "—"
        ent = f"{tr[0]['entropy']:.3f}->{tr[-1]['entropy']:.3f}" if tr else "—"
        kl = sum(r.get("kl", 0.0) for r in tr) / len(tr) if tr else float("nan")
        print(f"  {d.name:<18} 轮={len(tr):>3}  熵={ent:<16} 平均kl={kl:.5f}")
        print(f"      验证: {series}")
PY