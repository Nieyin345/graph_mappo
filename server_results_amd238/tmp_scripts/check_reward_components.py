"""B 线探针：奖励组件量级分解（diag vs real）+ 信用分配视野检查。

读 .tmp/{r4_base,real_base,r4_ent_ep2}/rollout_debug.jsonl，
打印各奖励组件随 update 的均值序列与占比。

用法：python .tmp/check_reward_components.py
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / ".tmp"

COMPONENTS = [
    "mean_reward_served", "mean_reward_failed", "mean_reward_generated",
    "mean_reward_dense", "mean_reward_storage", "mean_reward_keep_active",
    "mean_reward_waiting", "mean_reward_switch", "mean_reward_expired",
    "mean_reward_conflict",
]

EXTRA = [
    "mean_qkp_utilization", "mean_activated_edges",
    "mean_generated_keys", "mean_served_keys", "mean_arrived_keys",
    "mean_waiting_keys", "mean_failed_keys",
]


def load(name: str) -> list[dict]:
    p = BASE / name / "rollout_debug.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


for name in ("r4_base", "r4_ent_ep2", "real_base"):
    rows = load(name)
    if not rows:
        print(f"==== {name}: 无文件")
        continue
    print(f"==== {name}  updates={len(rows)}  steps/update={rows[0].get('steps')}")
    # 组件量级：首、中、末三段平均
    n = len(rows)
    segs = (rows[: max(1, n // 3)], rows[n // 3: 2 * max(1, n // 3)], rows[2 * max(1, n // 3):])
    for tag, chunk in zip(("前1/3", "中1/3", "后1/3"), segs):
        if not chunk:
            continue
        total = sum(abs(r["mean_reward"]) for r in chunk) / len(chunk)
        parts = []
        for c in COMPONENTS:
            v = sum(r.get(c, 0.0) for r in chunk) / len(chunk)
            if abs(v) > 1e-9:
                parts.append(f"{c[12:]}={v:+.2e}({v / total * 100 if total else 0:.0f}%)")
        print(f"  {tag} |reward|={total:.4f}  " + "  ".join(parts))
    # 关键物理量
    for tag, chunk in zip(("前1/3", "后1/3"), (rows[: max(1, n // 3)], rows[-max(1, n // 3):])):
        if not chunk:
            continue
        phys = "  ".join(
            f"{k[5:]}={sum(r.get(k, 0.0) for r in chunk) / len(chunk):.4g}" for k in EXTRA
        )
        print(f"  {tag} {phys}")
    print()
