#!/usr/bin/env python
"""汇总 r8 多种子实验的验证曲线与配对统计。

每个 run 的 metrics.jsonl 里 eval_validation 行给出每 5 轮一次的验证成功率
（15 个请求种子上的均值）。对比 BC 起点（=第 0 轮，从 checkpoint 扫描已知
验证 regime 为 0.6483）与专家（0.698）。
"""
import json
import sys
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
RUNS = sys.argv[1:] or ["r8_base_s42", "r8_base_s43", "r8_base_s44"]

BC0 = 0.6483   # BC 起点在验证 regime 的成功率（probe 扫描）
EXPERT = 0.6980  # 专家在验证 regime 的成功率

for run in RUNS:
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        print(f"{run}: 缺 metrics.jsonl")
        continue
    vals = []  # (update, mean_success_rate)
    train_sr = []
    for line in p.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "eval_validation" in row:
            ev = row["eval_validation"]
            vals.append((row.get("update", len(vals) + 1),
                         float(ev.get("mean_success_rate", 0.0))))
        if "mean_success_rate" in row:
            train_sr.append(float(row["mean_success_rate"]))
    print(f"\n== {run} ==")
    print(f"  训练侧（随机日，仅参考）: u1={train_sr[0]:.4f}"
          + (f"  u{len(train_sr)}={train_sr[-1]:.4f}" if len(train_sr) > 1 else ""))
    if vals:
        print(f"  验证曲线（每 5 轮）:")
        for u, v in vals:
            bar = "█" * int((v - 0.60) * 200) if v > 0.60 else ""
            print(f"    u{u:<3} {v:.4f}  Δvs-BC={v-BC0:+.4f}  {bar}")
        best = max(vals, key=lambda x: x[1])
        print(f"  最佳: u{best[0]} = {best[1]:.4f}  "
              f"(BC {BC0:.4f}, 专家 {EXPERT:.4f})")
    else:
        print("  无 eval_validation 行")

print("\n参照: BC 起点 0.6483 / 专家 0.6980（验证 regime，15 种子）")
