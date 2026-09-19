#!/usr/bin/env python
"""我的功效估算用了 u30 的 SD=0.0118，但规范说逐种子能晃 0.104 —— 谁对？

`docs/测试规范.md` ⑦ 写：「换一个**训练种子**，同一条 `eval_validation` 的
**update 5** 值可以在 **0.71–0.82** 之间晃，跨度 **0.104**」。

而我从 `ent01_s42/43/44` 在 **u30** 量到的 SD 只有 **0.0118**（跨度 0.0206）。
两者差 5 倍。这个差如果没搞清楚，我那个「3 个种子 t=3.65，够用」的结论
就是**建在一个可疑的 SD 上**。

最可能的解释：**u5 是训练早期，方差大；u30 已到平台，方差小。**
规范那句量的是 u5。这不是矛盾，是"方差随收敛缩小"。

本脚本直接验证这一点：对有多种子、且验证点覆盖多轮的 run 家族，
逐轮算**跨种子 SD**，看它是否随 update 下降。

用法（服务器上）：python /tmp/var_vs_update.py
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def series(run: str) -> dict[int, float]:
    """{轮次: 验证均值}"""
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        return {}
    out: dict[int, float] = {}
    last = None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "update" in o:
            last = o["update"]
        ev = o.get("eval_validation")
        if isinstance(ev, dict) and ev.get("per_seed_success") and last is not None:
            out[last] = float(ev["mean_success_rate"])
    return out


# 已知的多训练种子家族：前缀 → 种子列表
FAMILIES = {
    "ent01": [42, 43, 44],
    "ent01_g999": [42],          # 只有 1 个，跳过
    "r2_base": [42, 43, 44],
    "r2_ep2": [42, 43, 44],
    "diag_seed": [42, 43, 44, 45],
}

print("=== 逐轮跨种子 SD（同配置、不同 --seed）===")
for fam, seeds in FAMILIES.items():
    runs = [f"{fam}_s{s}" for s in seeds]
    ss = {r: series(r) for r in runs}
    ss = {r: v for r, v in ss.items() if v}
    if len(ss) < 2:
        print(f"  {fam}: 可用 run 不足（{list(ss)}），跳过")
        continue
    common = set.intersection(*(set(v) for v in ss.values()))
    if not common:
        print(f"  {fam}: 无共同轮次，跳过")
        continue
    print(f"  {fam}  ({len(ss)} 个种子: {', '.join(r.split('_s')[-1] for r in ss)})")
    print(f"    {'轮':>4} {'均值':>8} {'SD':>8} {'跨度':>8}")
    for u in sorted(common):
        vals = [ss[r][u] for r in ss]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"    u{u:<3} {statistics.mean(vals):8.4f} {sd:8.4f} "
              f"{max(vals) - min(vals):8.4f}")
    print()

print("=== 交叉验证：规范说的 u5 跨度 0.104 ===")
print("  上面 ent01 家族的 u5 跨度若明显大于 u30，就证实了『方差随收敛缩小』；")
print("  若 u5 跨度也在 0.02 量级，那规范那句 0.104 量的是**别的东西**")
print("  （旧代码 / 默认 env_seed=7 那条流），要单独标出来。")
