"""多跑几个验证种子能带来多少分辨率？—— 用**已有的** per_seed_success 直接算。

不需要新实验：metrics.jsonl 里存了每个 run 各验证点的 15 个逐种子成功率。
把它们当"更大的种子池"抽样，就能算 n=15/30/45/60 时的评估 SE，
以及那个 SE 换算成"配对分辨率"是多少。

为什么要它：`evaluate_validation` 每 eval_interval 轮跑一次 15 episode × 240 步，
相对于 1440 步的训练 rollout 是**便宜的**。若加验证种子能把评估 SE 从 0.036
降到 0.02，那是纯配置改动（训练器已经在做这件事，只改 episodes/seeds 数量）。
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
random.seed(0)

# 收集所有验证点的 per_seed_success（每个 15 个）
pts = []
for d in sorted(OUT.iterdir()):
    p = d / "metrics.jsonl" if d.is_dir() else None
    if not p or not p.exists():
        continue
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = o.get("eval_validation")
        if isinstance(ev, dict) and ev.get("per_seed_success"):
            v = [float(x) for x in ev["per_seed_success"]]
            if len(v) == 15:
                pts.append(v)

print(f"验证点总数 {len(pts)}（每个 15 个种子）")
allseeds = [x for v in pts for x in v]
sd_hat = statistics.stdev(allseeds)
print(f"合并后 SD(per-seed) = {sd_hat:.4f}")
print()

print("=" * 78)
print("1. 评估 SE 随验证种子数 n")
print("=" * 78)
print(f"  {'n':>4}{'SE = SD/√n':>14}{'相对 n=15':>12}")
for n in (15, 30, 45, 60, 90, 120):
    print(f"  {n:>4}{sd_hat / n ** 0.5:>14.4f}{sd_hat / 15 ** 0.5 / (sd_hat / n ** 0.5):>11.2f}×")

print()
print("=" * 78)
print("2. 但**配对**后能到多少？—— 这才是判据用的量")
print("=" * 78)
print("  配对 SE 取决于**逐种子差的 SD**，不只看单侧 SE。")
print("  用实测的配对 SE（两条策略同种子相减，见 .tmp/eval_se_check.py）：0.0063（n=15）。")
print()
print("  配对差的 SD 与样本量的关系：SE_pair(n) = SD_pair / √n")
sd_pair = 0.0063 * 15 ** 0.5
print(f"  反推 SD_pair = {sd_pair:.4f}")
for n in (15, 30, 45, 60):
    print(f"    n={n:>3} → 配对 SE = {sd_pair / n ** 0.5:.4f}")

print()
print("=" * 78)
print("3. 成本侧（要诚实）")
print("=" * 78)
print("  evaluate_validation 每 eval_interval 轮跑一次 n episode × 240 步。")
print("  训练一轮是 8 episode × 1440 步 = 11520 步 rollout + 175s update。")
print("  验证一轮 n=15 是 15×240 = 3600 步，**约训练 rollout 的 31%**。")
print("  翻到 n=30 就是 62% —— **不是免费的**，会直接吃掉训练速度。")
print()
print("  ⚠ 而且上面第 2 节的配对 SE 是**评估侧**的；换训练种子的分辨率")
print("     ~0.035 是**另一个**方差来源，加验证种子**动不了它**。")
print("     所以：加验证种子只能压评估噪声，压不动主要方差来源。")
