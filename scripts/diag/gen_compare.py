# -*- coding: utf-8 -*-
"""同一个 regime 上，RL 与专家的生成/等待/利用率对照。

**这一步必须做，因为跨 regime 比是错的**：
  专家在**验证 regime**（天 330-365、240 步）上 生成/服务 = 214.5；
  而 RL 那个 97.3 是**训练 regime**（天 0-295、1440 步）。
  两边的天数、需求量、回合长度都不同，比值不可比。

所以本脚本把 RL 也在**验证 regime** 上跑一遍（steps.csv 里有逐步的
generated_keys），与专家的同一组量并列。

回答的问题：训练侧看到的"生成量掉 41%"，
  (a) 在验证 regime 上 RL 的生成量相对专家是**低**的 → 漂到了"少生成"那侧
  (b) 还是**高**的 → 漂的方向与专家相反
没有这个对照，41% 那个数就只是一个孤立的变化，不知道好坏。
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
exp = {k: [float(x) for x in expert[k]] for k in
       ("seeds", "success", "served", "failed", "arrived",
        "generated", "waiting_mean", "qkp_util_mean")}

# RL：聚合 steps.csv
rl_dir = VAL / "rl_ent01_s42_u30" / "rl_ent01_s42_u30"
by_seed = defaultdict(lambda: {"gen": 0.0, "served": 0.0, "fail": 0.0,
                               "wait": 0.0, "util": 0.0, "n": 0})
with open(rl_dir / "steps.csv", newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        s = int(row["seed"])
        d = by_seed[s]
        d["gen"] += float(row["generated_keys"])
        d["served"] += float(row["served_keys"])
        d["fail"] += float(row["failed_keys"])
        d["wait"] += float(row["waiting_keys"])
        d["util"] += float(row["qkp_utilization"])
        d["n"] += 1

seeds = sorted(set(by_seed) & set(int(s) for s in exp["seeds"]))
print(f"共同种子 {len(seeds)} 个：{seeds[:5]}...")
print()
print("=" * 104)
print("同一个验证 regime（天 330-365，240 步，种子 100-114）")
print("=" * 104)
hdr = (f"  {'seed':>5}{'succ_E':>9}{'succ_RL':>9}"
       f"{'gene_E':>16}{'gene_RL':>16}{'gen/srv_E':>11}{'gen/srv_RL':>11}")
print(hdr)
ge = gr = se = sr = 0.0
we = wr = ue = ur = 0.0
for s in seeds:
    i = [int(x) for x in exp["seeds"]].index(s)
    d = by_seed[s]
    e_gen, e_srv = exp["generated"][i], exp["served"][i]
    r_served = d["served"]
    row = (f"  {s:>5}{exp['success'][i]:>9.4f}"
           f"{r_served / (exp['arrived'][i] if exp['arrived'][i] else 1):>9.4f}"
           f"{e_gen:>16,.0f}{d['gen']:>16,.0f}"
           f"{e_gen / e_srv if e_srv else 0:>11.1f}"
           f"{d['gen'] / r_served if r_served else 0:>11.1f}")
    print(row)
    ge += e_gen; gr += d["gen"]
    se += e_srv; sr += r_served
    we += exp["waiting_mean"][i]; wr += d["wait"] / max(d["n"], 1)
    ue += exp["qkp_util_mean"][i]; ur += d["util"] / max(d["n"], 1)

n = len(seeds)
print()
print("=" * 104)
print("汇总（同一 regime，可直接比）")
print("=" * 104)
print(f"  {'量':<24}{'专家':>20}{'RL(u30)':>20}{'RL/专家':>12}")
print(f"  {'生成量/局':<24}{ge/n:>20,.0f}{gr/n:>20,.0f}{(gr/n)/(ge/n):>12.3f}")
print(f"  {'服务量/局':<24}{se/n:>20,.0f}{sr/n:>20,.0f}{(sr/n)/(se/n):>12.3f}")
print(f"  {'生成/服务':<24}{ge/se:>20.2f}{gr/sr:>20.2f}{(gr/sr)/(ge/se):>12.3f}")
print(f"  {'等待量均值':<24}{we/n:>20,.0f}{wr/n:>20,.0f}{(wr/n)/(we/n):>12.3f}")
print(f"  {'利用率均值':<24}{ue/n:>20.5f}{ur/n:>20.5f}{(ur/n)/(ue/n):>12.3f}")
print()
print("  ★ 判读：")
print("    RL/专家的 生成/服务 若 <1 → RL 用相对更少的密钥服务同样的需求，")
print("      在验证 regime 上也是**效率更高**，训练侧那个 -41% 是漂对了方向。")
print("    若 >1 → RL 生成得更浪费，训练侧的漂移是坏的。")
