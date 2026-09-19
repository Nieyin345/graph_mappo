"""第六浪中期速览：eval@5 一览 + stor10 物理签名初查（不改结论，只看量级）。

用法：python .tmp/check_r6_peek.py
"""
from __future__ import annotations

import json
from pathlib import Path

M = Path(__file__).resolve().parents[1] / ".tmp"

ARMS = ("r6_vcoef1_long", "r6_lowv025", "r6_base_lr1e4", "r6_g999", "r6_stor10",
        "real_base", "real_vcoef1")


def load(name):
    p = M / name / "metrics.jsonl"
    if not p.exists():
        return [], []
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    tr = [r for r in rows if "update" in r]
    ev = [r["eval_validation"] for r in rows if "eval_validation" in r]
    return tr, ev


for name in ARMS:
    tr, ev = load(name)
    if not ev and not tr:
        print(f"{name:<16} (无数据)")
        continue
    evs = " ".join(f"{e['mean_success_rate']:.4f}" for e in ev)
    edges = ""
    if len(tr) >= 4:
        seg = max(1, len(tr) // 3)
        a = sum(x.get("mean_activated_edges", 0.0) for x in tr[:seg]) / seg
        b = sum(x.get("mean_activated_edges", 0.0) for x in tr[-seg:]) / seg
        edges = f"  激活边 {a:.1f}->{b:.1f}"
    print(f"{name:<16} updates={len(tr):<3} evals: {evs}{edges}")
