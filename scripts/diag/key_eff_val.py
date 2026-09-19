#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同 regime 上，RL vs 专家的密钥效率（三训练种子）。

### 这个缺口为什么非补不可

「RL 比专家省 36.5% 密钥」这条在任何产物里都**没有被实测过**：

  · `outputs/eval/` 原本只有 `expert_seeds100_240.json`（**旧版，无 generated**）
    + `stop_sensitivity.json`。我 2026-09-19 18:39 重跑专家才补上 `generated`
    （逐位复现 61/61，确定性坐实）。
  · `eff_3seeds.py` 引用 `outputs/ent01_s${s}/checkpoint_update_000030.pt`
    —— 实测该目录 **`.pt` 数为 0**，脚本从来没成功。
  · `gen_compare.py` 引用 `outputs/eval/rl_ent01_s42_u30/.../steps.csv`
    —— 实测不存在。
  · 我 15:53 的 `/tmp/keyeff.py` 自己就写了「专家 json **没有 generated**
    ⟹ 无法从它算……那个数来自另一次探针」。

⟹ 36.5% 的**出处不明**（探针脚本已不在或不可复现）。本脚本给出
   **同 regime、可复现、三训练种子** 的版本，替代那个数。

### 同 regime 是硬前提

`run_baselines.py` 与 `eval_expert.py` 都走
`load_validation_profile` + `build_validation_env_config` + `start_seed+seed`，
且 `--policies __none__` 不改环境 ⟹ 两边同 regime（天 330–365、240 步、
请求种子 100–114）。**跨 regime 比是错的**（专家验证 214.5 vs RL 训练 97.3，
回合长度差 6 倍）。

### 判据

  · 逐请求种子配对（RL − 专家），三个训练种子各自给一个 Δ
  · 三个训练种子的**方向必须一致**（§4⑦：训练种子本身是更大的方差源）
  · 「省 X%」= 1 − (RL 生成/服务) / (专家 生成/服务)
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

VAL = Path("/opt/qkd/graph_mappo/outputs/eval")
EXP = VAL / "expert_seeds100_240_gen.json"
TRAIN_SEEDS = [42, 43, 44]


def load_expert():
    d = json.loads(EXP.read_text(encoding="utf-8"))
    return {
        "seeds": [int(x) for x in d["seeds"]],
        "gen": [float(x) for x in d["generated"]],
        "served": [float(x) for x in d["served"]],
        "success": [float(x) for x in d["success"]],
        "arrived": [float(x) for x in d["arrived"]],
    }


def agg_steps(p: Path):
    """把 steps.csv 按请求种子聚合 **generated**（它不在 episodes.csv 里）。

    ⚠ 不要用 steps.csv 反推 success_rate：`served/(served+failed)` 是**错的**。
    实测 s42：arrived=20,475,667 而 served+failed=20,263,684 ⟹ 有 211,983 个密钥
    **既没服务也没失败（还在排队）** ⟹ 反推会高估约 1.8 点
    （0.7093 vs 真值 0.6780）。success_rate 一律取自 `episodes.csv`。
    """
    by = defaultdict(lambda: {"gen": 0.0, "n": 0})
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            s = int(row["seed"])
            by[s]["gen"] += float(row["generated_keys"])
            by[s]["n"] += 1
    return by


def read_episodes(p: Path):
    """episodes.csv 是 per-seed 的权威读数（含真 success_rate 与 arrived）。"""
    by = {}
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by[int(row["seed"])] = {
                "success": float(row["success_rate"]),
                "arrived": float(row["arrived_keys"]),
                "served": float(row["served_keys"]),
                "failed": float(row["failed_keys"]),
            }
    return by


def main():
    exp = load_expert()
    print("=" * 92)
    print("专家锚（验证 regime：天 330-365、240 步、请求种子 100-114）")
    print("=" * 92)
    eg, es = sum(exp["gen"]), sum(exp["served"])
    print("  生成/服务 = %.2f   成功率 = %.4f   served/步 = %.0f"
          % (eg / es, st.mean(exp["success"]), es / len(exp["seeds"]) / 240))
    print()

    print("=" * 92)
    print("RL（同 regime，三训练种子，u30 checkpoint = ent01_rerun_s*）")
    print("=" * 92)
    print("  %-20s %13s %13s %11s %11s" %
          ("臂", "生成/步", "服务/步", "生成/服务", "成功率"))
    print("  " + "-" * 74)
    print("  %-20s %13.0f %13.0f %11.2f %11.4f" %
          ("专家锚", eg / 15 / 240, es / 15 / 240, eg / es,
           st.mean(exp["success"])))

    rl = {}
    for s in TRAIN_SEEDS:
        # run_baselines 的输出目录结构：<out>/<policy_name>/{steps,episodes}.csv
        name = "rl_ent01rerun_s%d_u30" % s
        d = VAL / name / name
        if not (d / "steps.csv").exists() or not (d / "episodes.csv").exists():
            print("  %-20s 缺数据（%s）" % (name, d))
            continue
        rl[s] = {"gen": agg_steps(d / "steps.csv"), "ep": read_episodes(d / "episodes.csv")}

    if not rl:
        print("没有 RL 数据")
        return

    print()
    perarm = {}
    for s, dd in sorted(rl.items()):
        by, ep = dd["gen"], dd["ep"]
        seeds = sorted(by)
        g = sum(by[x]["gen"] for x in seeds)
        sv = sum(ep[x]["served"] for x in seeds)
        ar = sum(ep[x]["arrived"] for x in seeds)
        perarm[s] = {
            "seeds": seeds,
            "gen": [by[x]["gen"] for x in seeds],
            "served": [ep[x]["served"] for x in seeds],
            "arrived": [ep[x]["arrived"] for x in seeds],
            "success": [ep[x]["success"] for x in seeds],
            "ratio": g / sv,
        }
        print("  %-20s %13.0f %13.0f %11.2f %11.4f" %
              ("ent01_rerun_s%d" % s, g / len(seeds) / 240,
               sv / len(seeds) / 240, g / sv, st.mean(perarm[s]["success"])))

    print()
    print("=" * 92)
    print("判读 1：相对专家省了多少（同 regime，训练种子为单位）")
    print("=" * 92)
    print("  %-20s %11s %13s %s" % ("臂", "生成/服务", "vs 专家", "省"))
    print("  " + "-" * 74)
    print("  %-20s %11.2f %13s %s" % ("专家锚", eg / es, "—", "—"))
    pct = []
    for s in sorted(perarm):
        r = perarm[s]["ratio"]
        rel = r / (eg / es)
        pct.append((1 - rel) * 100)
        print("  %-20s %11.2f %12.3fx %+10.1f%%"
              % ("ent01_rerun_s%d" % s, r, rel, (1 - rel) * 100))
    print()
    m = st.mean(pct)
    print("  三训练种子省密钥：%s" % "  ".join("%.1f%%" % x for x in pct))
    print("  ★ 均值 = %.1f%%   SD = %.1f%%   极差 = %.1f%%"
          % (m, st.stdev(pct) if len(pct) > 1 else 0, max(pct) - min(pct)))
    print("  方向一致（全部为正）？ %s" % ("是 ✓" if all(x > 0 for x in pct) else "否 ✗"))
    print()
    print("  记忆库里的 36.5%%（= mean(22.3, 41.9, 45.4)，三训练种子）：")
    print("    本次实测 %s ⟹ %s" % ("%.1f%%" % m,
          "与记忆同量级" if abs(m - 36.5) < 12 else "**与记忆不同量级，记忆该订正**"))

    print()
    print("=" * 92)
    print("判读 2：逐请求种子配对（最强检验，配对消掉需求侧的种子噪声）")
    print("=" * 92)
    print("  %-20s %11s %11s %10s %9s" % ("臂", "Δ生成/服务", "SD(逐种子)", "t", "同向"))
    print("  " + "-" * 74)
    exps = exp["seeds"]
    for s in sorted(perarm):
        # 逐请求种子配对：RL 与专家同一 seed 的 生成/服务
        rseeds = perarm[s]["seeds"]
        if rseeds != exps:
            print("  %-20s !! 种子序列不一致，跳过" % ("ent01_rerun_s%d" % s))
            continue
        d = []
        for i, _ in enumerate(exps):
            er = exp["gen"][i] / exp["served"][i]
            rr = perarm[s]["gen"][i] / perarm[s]["served"][i]
            d.append(rr - er)
        md = st.mean(d)
        sd = st.stdev(d)
        sdw = st.stdev(d)
        t = md / (sd / len(d) ** 0.5) if sd > 0 else float("inf")
        nz = sum(1 for x in d if x < 0)
        print("  %-20s %+11.2f %11.2f %+9.2f %5d/%d"
              % ("ent01_rerun_s%d" % s, md, sdw, t, nz, len(d)))

    print()
    print("=" * 92)
    print("判读 3：成功率 vs 专家（**必须配对**，且这是臂-vs-专家 ⟹ 硬件偏置不抵消）")
    print("=" * 92)
    print("  ⚠ 未配对均值会误导：本项目记忆 `rl-vs-expert-only-0.0086` 记的是")
    print("     **配对 +0.0086**，而本次未配对均值算出来是 RL 低 ~0.014 ⟹ 符号相反。")
    print("     不配对时 15 个请求种子的 SE≈0.062、配对后≈0.012（§4 规矩）⟹ 必须配对。")
    print()
    print("  %-20s %11s %11s %10s %9s" % ("臂", "Δ(配对)", "SD(逐种子)", "t", "同向"))
    print("  " + "-" * 74)
    for s in sorted(perarm):
        rseeds = perarm[s]["seeds"]
        if rseeds != exps:
            print("  %-20s !! 种子序列不一致，跳过" % ("ent01_rerun_s%d" % s))
            continue
        d = [perarm[s]["success"][i] - exp["success"][i] for i in range(len(exps))]
        md, sd = st.mean(d), st.stdev(d)
        t = md / (sd / len(d) ** 0.5) if sd > 0 else float("inf")
        npos = sum(1 for x in d if x > 0)
        print("  %-20s %+11.4f %11.4f %+9.2f %5d/%d"
              % ("ent01_rerun_s%d" % s, md, sd, t, npos, len(d)))
    print()
    print("  未配对均值：RL=%.4f  专家=%.4f  差=%+.4f"
          % (st.mean([x for s in perarm for x in perarm[s]["success"]]),
             st.mean(exp["success"]),
             st.mean([x for s in perarm for x in perarm[s]["success"]])
             - st.mean(exp["success"])))

    print()
    print("=" * 92)
    print("=== 结论 ===")
    print("=" * 92)
    print("· 「省密钥」若三个训练种子同向为正、且逐请求种子配对 15/15 同向 ⟹")
    print("  **现象成立**，可以写进日志。")
    print("· 但**奖励对生成量完全不敏感**（reward-is-the-metric 已核实：")
    print("  served 101.4%% + failed 1.7%%，generated 恰好 0.000000）⟹ 即使现象")
    print("  成立，它也是**无人看守的副产品**，不是学到的能力——正因如此才可能漂回去。")
    print("· 要把它变成能力，得先把密钥效率写进奖励；那需要重新预注册（新目标）。")


if __name__ == "__main__":
    main()
