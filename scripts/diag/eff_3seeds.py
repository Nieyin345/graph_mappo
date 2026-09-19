# -*- coding: utf-8 -*-
"""把「RL 比专家省 22% 密钥」从 1 个训练种子扩到 3 个（§4⑦ 的规矩）。

单种子（s42）已经量出 RL/专家 生成-服务比 = 0.777（逐验证种子 15/15 同向）。
但按 docs/测试规范.md §4⑦，**训练种子本身是更大的方差来源**，
单种子不能定论。这里把 s43/s44 的 u30 checkpoint 也在**同一验证 regime**
上评估，看 0.777 是不是稳定。

判据（与主结论一致）：
  · 若三个种子的 生成/服务 都显著低于专家，且**方向一致** → 可信
  · 逐种子对专家的比值就是三个独立读数，可直接看其散布
注意：success_rate 那个 +0.0070 仍然远低于 0.035，本脚本**不**声称
"RL 超过专家"，只声称**密钥效率更高**。
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
VAL = ROOT / "outputs/eval"
expert = json.loads((VAL / "expert_seeds100_240_gen.json").read_text(encoding="utf-8"))
EXP_SEEDS = [int(x) for x in expert["seeds"]]
exp_gen = [float(x) for x in expert["generated"]]
exp_srv = [float(x) for x in expert["served"]]
exp_ok = [float(x) for x in expert["success"]]

print("=" * 96)
print("专家（同一验证 regime，种子 100-114）")
print("=" * 96)
print(f"  生成/服务 = {sum(exp_gen)/sum(exp_srv):.2f}   成功率 = {st.mean(exp_ok):.4f}")
print()

rows = {}
for s in (42, 43, 44):
    d = VAL / f"rl_ent01_s{s}_u30" / f"rl_ent01_s{s}_u30"
    csvp = d / "steps.csv"
    if not csvp.exists():
        print(f"  ent01_s{s}: 缺 {csvp}")
        continue
    by_seed = defaultdict(lambda: {"gen": 0.0, "served": 0.0})
    with open(csvp, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            b = by_seed[int(r["seed"])]
            b["gen"] += float(r["generated_keys"])
            b["served"] += float(r["served_keys"])
    # 逐种子的 gen/srv 比
    ratios = []
    for sd in EXP_SEEDS:
        if sd in by_seed and by_seed[sd]["served"]:
            ratios.append(by_seed[sd]["gen"] / by_seed[sd]["served"])
    if not ratios:
        continue
    g = sum(by_seed[sd]["gen"] for sd in EXP_SEEDS)
    v = sum(by_seed[sd]["served"] for sd in EXP_SEEDS)
    rows[s] = {"ratio": g / v, "per_seed": ratios}

exp_ratio = sum(exp_gen) / sum(exp_srv)
print("=" * 96)
print("三个训练种子的 u30，同一验证 regime")
print("=" * 96)
print(f"  {'种子':<10}{'生成/服务':>12}{'RL/专家':>10}{'逐验证种子低于专家的个数':>26}")
for s, r in rows.items():
    exp_per = [exp_gen[i] / exp_srv[i] for i in range(len(EXP_SEEDS))]
    below = sum(1 for a, b in zip(r["per_seed"], exp_per) if a < b)
    print(f"  ent01_s{s:<4}{r['ratio']:>12.2f}{r['ratio']/exp_ratio:>10.3f}"
          f"{below:>18}/{len(r['per_seed'])}")
print(f"  {'专家':<10}{exp_ratio:>12.2f}{1.0:>10.3f}")

if rows:
    rs = [r["ratio"] for r in rows.values()]
    print()
    print("  RL 三种子 生成/服务: " + "  ".join(f"{x:.2f}" for x in rs))
    print(f"    均值 {st.mean(rs):.2f}   SD {st.stdev(rs) if len(rs)>1 else 0:.3f}")
    print(f"  专家 {exp_ratio:.2f}")
    print(f"  ⟹ 三个种子的比值相对专家: "
          + "  ".join(f"{x/exp_ratio:.3f}" for x in rs))
    print()
    saved = [1 - x / exp_ratio for x in rs]
    print(f"  ★ 省下的密钥比例，逐种子: "
          + "  ".join(f"{x:.1%}" for x in saved))
    print(f"    均值 **{st.mean(saved):.1%}**"
          f"   （最小 {min(saved):.1%}，最大 {max(saved):.1%}）")
    if len(saved) > 1:
        sd = st.stdev(saved)
        print(f"    SD {sd:.3f}  —— 三个种子都为正 ⟹ 方向稳定")
