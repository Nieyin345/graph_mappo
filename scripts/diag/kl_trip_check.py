#!/usr/bin/env python
"""KL 早停在这些 run 里**实际触发过吗**？

AST 已经证明：`mappo_trainer.py:731` 的 break 隶属于 **epoch 循环**，所以
KL 一 Trio 就中止**整轮更新**——epochs=1 和 epochs=2 在这一点上**行为相同**。
`configs/train_safe_ep2.yaml` 里那句「epochs=2 时可以只停掉第二个 epoch」是错的。

但那句话的**实际后果**取决于 KL 到底触发不触发。如果所有 run 的 `total_batches`
都等于"该轮应有的 batch 数"，说明从没提前停过，那句话就是**无害的错误**；
如果确实有轮次早停，那它影响的是**所有**臂（不只 ep2），读指标时要注意。

`total_batches` 不在 metrics.jsonl 里，但 `kl` 是**对已跑 batch 的均值**，
而早停会让 batch 数变少 → 同一轮的 kl 会因早停而偏高、且 update_s 会明显偏小。
所以这里查两件事：
  1. 逐轮 `update_s` 有没有突然塌陷的轮次（早停的指纹）；
  2. 若日志里能拿到 batch 数，直接对比。

用法（服务器上）：python /tmp/kl_trip_check.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")

RUNS = [
    "ent01_s42", "ent01_s43", "ent01_s44",
    "vcoef1_s42", "vcoef1_s43", "vcoef1_s44",
]


def rows(run: str):
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return [r for r in out if "update" in r]


print("=== 1. 逐轮 update_s 是否有塌陷（早停的指纹）===")
for run in RUNS:
    rs = rows(run)
    if not rs:
        print(f"  {run}: 无数据")
        continue
    us = [(r["update"], r.get("update_s")) for r in rs if isinstance(r.get("update_s"), (int, float))]
    if not us:
        continue
    vals = sorted(v for _, v in us)
    med = vals[len(vals) // 2]
    # 早停会让 update_s 掉到中位数的一半以下
    odd = [(u, v) for u, v in us if v < 0.6 * med]
    mark = "  ".join(f"u{u}={v:.0f}s" for u, v in odd) if odd else "无"
    print(f"  {run:<12} 中位 {med:6.1f}s ({len(us)} 轮)  疑似早停: {mark}")

print()
print("=== 2. kl 的分布（早停会让 kl 偏高）===")
import statistics
for run in RUNS:
    rs = rows(run)
    kls = [r["kl"] for r in rs if isinstance(r.get("kl"), (int, float))]
    if not kls:
        print(f"  {run}: 无 kl")
        continue
    mx = max(kls)
    print(f"  {run:<12} kl 中位 {statistics.median(kls):.5f}  最大 {mx:.5f}  "
          f"(target_kl=0.02, 均值口径)")

print()
print("=== 3. 日志里有没有 KL 早停的痕迹 ===")
for run in RUNS:
    log = Path("/tmp") / f"{run}.log"
    if not log.exists():
        continue
    txt = log.read_text(encoding="utf-8", errors="replace")
    hits = re.findall(r"(?i)(kl[^\n]{0,40}(stop|early|trip)[^\n]{0,40})", txt)
    print(f"  {run:<12} {'命中 ' + str(len(hits)) + ' 处' if hits else '无 KL 早停字样'}")

print()
print("=== 判读 ===")
print("  若各 run 的 update_s 都没有塌陷、kl 都在 target 之下，")
print("  说明 KL 早停**从没触发**——那么 train_safe_ep2.yaml 里那句错误描述")
print("  是**无害的**（不影响读数），但仍要改：一句错误的代码语义描述")
print("  会让人在真正需要它的时候按错的模型去读。")
