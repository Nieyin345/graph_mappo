#!/usr/bin/env python
"""查看某个 run 的 H1 特征逐轮轨迹（优势塌陷诊断）。

H1：critic 拟合回报越来越好（corr 上升）→ 真实优势趋零（|A| 下降）→
per-minibatch 归一化把噪声放大回单位尺度 → 策略学到的是噪声。
"""
import json
import sys
from pathlib import Path

MAIN = Path("/opt/qkd/graph_mappo")
RUNS = sys.argv[1:] or ["r8_base_s42"]

COLS = [
    ("update", "u", 3, "d"),
    ("mean_abs_advantage", "|A|", 8, ".4f"),
    ("value_return_corr", "corr", 7, ".3f"),
    ("value_std", "V_std", 7, ".3f"),
    ("return_std", "R_std", 6, ".3f"),
    ("entropy", "entropy", 8, ".3f"),
    ("kl", "kl", 8, ".5f"),
    ("mean_success_rate", "succ", 7, ".4f"),
]

for run in RUNS:
    path = MAIN / "outputs" / run / "metrics.jsonl"
    if not path.exists():
        print(f"!! {run}: 无 metrics.jsonl")
        continue
    rows = []
    evals = []
    for line in path.open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        if "update" in d:
            rows.append(d)
        elif "eval_validation" in d:
            evals.append(d["eval_validation"])

    print(f"=== {run} （{len(rows)} 轮）===")
    header = " ".join(f"{label:>{width}}" for _k, label, width, _f in COLS)
    print(header)
    print("-" * len(header))
    for d in rows:
        cells = []
        for key, _label, width, fmt in COLS:
            v = d.get(key)
            if v is None:
                cells.append(f"{'--':>{width}}")
            elif isinstance(v, int):
                cells.append(f"{v:>{width}d}")
            else:
                cells.append(f"{v:>{width}{fmt}}")
        print(" ".join(cells))

    if evals:
        print()
        print("  eval_validation:")
        for i, e in enumerate(evals, 1):
            sr = e.get("mean_success_rate")
            rew = e.get("mean_reward")
            print(f"    #{i}  success={sr!r}  reward={rew!r}")
    print()
