#!/usr/bin/env python
"""专家 vs RL：**在 RL 实际使用的验证种子上**做配对比较。

背景（一个我自己制造的假警报，已核对）：
`server_results/runs/outputs/eval/` 里的 expert*.json 全是种子 **7-21**，
而 RL 的 `eval_validation` 用的是种子 **100-114**（52/66 个 run）。
两者完全不相交，于是"RL vs 专家"看起来不可比。
但补测专家 @ 100-114 得 **0.6979**，与日志记录的 0.698 **逐位吻合**
—— 说明原始测量本来就是对的，只是那个 JSON 没被归档下来（节点换过）。
本脚本把这条唯一同口径的对照**配对**算出来（此前从未配对过）。

### ⚠ 2026-09-19：这个 t **不能**用来判"该配置超过了专家"

本脚本算的是**配对差**：同一次训练、在 15 个验证种子上逐点配对。它的 SE
只覆盖"换验证种子"的噪声，**不覆盖"换训练种子"**——而训练种子才是本项目
单种子分辨率 ~0.035 的来源（`docs/测试规范.md` ⑦）。

所以 `t=+4.28` 的正确读法是「**在这 15 个留出实例上可辨**」，
不是「该配置超过了专家」。同一个 run 换成跨训练种子的分母，t 从 +4.28
掉到 **+3.03、p=0.094**（未达显著）。

**要回答"有没有超过专家"，用 `scripts/diag/ent01_vs_expert.py`**，
它按 Δ_s（n=3）算，并用 df=2 的临界值 4.303（**不是 2**）。
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def expert_perseed(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    return dict(zip(d["seeds"], d["success"])), d


def rl_points(run: str):
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        return []
    last_u, pts = None, []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "update" in r:
                last_u = r["update"]
            ev = r.get("eval_validation")
            if isinstance(ev, dict) and ev.get("per_seed_success"):
                pts.append((last_u, float(ev["mean_success_rate"]),
                            list(ev["per_seed_success"])))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", default="eval/expert_seeds100_240.json",
                    help="相对 outputs/ 的路径")
    ap.add_argument("--round", type=int, default=15,
                    help="取哪个 update 的验证点做配对")
    ap.add_argument("--runs", nargs="*", default=[
        "r8_base_s42", "r8_base_s43", "r8_base_s44",
        "ent01_s42", "ent01_s43", "ent01_s44",
        "r7_base", "r7_fix_ent", "ent01_g999_s42",
    ])
    args = ap.parse_args()

    ep, ed = expert_perseed(OUT / args.expert)
    seeds = list(ed["seeds"])
    base = [ep[s] for s in seeds]
    bm = sum(base) / len(base)
    print(f"专家 @ 种子 {seeds[0]}-{seeds[-1]} (n={len(seeds)}, "
          f"{ed['steps']} 步):  mean = {bm:.4f}  "
          f"SD = {statistics.stdev(base):.4f}")
    print()

    print(f"{'run':<20}{'u':>4}{'均值':>9}{'配对差':>10}{'SE':>8}{'t':>7}"
          f"{'胜/负':>9}")
    print("-" * 68)
    for run in args.runs:
        pts = rl_points(run)
        if not pts:
            print(f"{run:<20}  (无验证点)")
            continue
        sel = [p for p in pts if p[0] == args.round]
        u, m, ps = sel[0] if sel else pts[-1]
        if len(ps) != len(seeds):
            print(f"{run:<20}{u:>4}  逐种子数 {len(ps)} != {len(seeds)}，跳过")
            continue
        diffs = [ps[i] - base[i] for i in range(len(seeds))]
        md = sum(diffs) / len(diffs)
        sed = statistics.stdev(diffs) / len(diffs) ** 0.5
        t = md / sed if sed else float("nan")
        win = sum(1 for d in diffs if d > 0)
        print(f"{run:<20}{u:>4}{m:>9.4f}{md:>+10.4f}{sed:>8.4f}{t:>+7.2f}"
              f"{win:>5}/{len(diffs)-win:<3}")
        print(f"{'':<20}     逐种子差: " + " ".join(f"{d:+.3f}" for d in diffs))


if __name__ == "__main__":
    main()
