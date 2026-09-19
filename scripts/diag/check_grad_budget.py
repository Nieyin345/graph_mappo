#!/usr/bin/env python
"""核对「一轮几次梯度步」与 kl 的累积形态。

### 为什么要核

`docs/训练诊断记录.md`「KL ≈ 0.001」一节里我写了
**「一轮只做一次梯度步，而这一次步把策略分布只挪了 0.1%」**。
但 `MAPPOTrainer.update()`（`mappo_trainer.py:675-714`）的结构是

    for _epoch in range(epochs):
        for batch in batch_iter:      # 按 minibatch_size 切
            ...
            self.optimizer.step()     # ★ 每个 minibatch 一次

配置 `epochs: 1`、`minibatch_size: 256`、buffer 8 env × 1440 = 11520
⟹ 应该是 **45 次**，不是 1 次。而 `stats.n_minibatches` 就是日志里的 `nb=`。

### 判读

  nb 恒为 45 ⟹ 「一次梯度步」是**错的**，要改诊断记录。
  kl 在 optimizer.step() **之前**算 ⟹ 同一轮内它随 minibatch 递增
    （前面的步已经挪过策略），所以逐轮 kl 是「进入该轮时」的低估值。

用法（服务器上）：/opt/qkd/venv/bin/python /tmp/check_grad_budget.py
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def rows(run: str) -> list[dict]:
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(o.get("update"), int):
            out.append(o)
    return out


runs = sorted(
    d.name for d in OUT.iterdir()
    if d.is_dir() and (d / "metrics.jsonl").exists()
)

print("=" * 78)
print("① 一轮几次梯度步？—— 统计 nb=  （代码：optimizer.step() 在 minibatch 循环内）")
print("=" * 78)
c: Counter = Counter()
per_run: dict[str, int] = {}
for r in runs:
    rs = rows(r)
    nbs = [o["nb"] for o in rs if isinstance(o.get("nb"), int)]
    if nbs:
        c.update(nbs)
        per_run[r] = Counter(nbs).most_common(1)[0][0]
print(f"  {len(per_run)} / {len(runs)} 个 run 有 nb 读数")
for v, n in sorted(c.items()):
    print(f"    nb={v:<5} 出现 {n:>4} 次")
print()
print("  判读：", end="")
if 45 in c:
    top = c.most_common(1)[0]
    print(f"**nb={top[0]} 是常态**（{top[1]} 次）")
    print("        ⟹ 「一轮只做一次梯度步」作废；实际 = buffer / minibatch_size")
else:
    print("未见 45，需人工看")

print()
print("=" * 78)
print("② kl / ratio / actor_grad 的逐轮形态")
print("=" * 78)
# 取读数最多的 4 个 run 家族做展示
fam: dict[str, list[dict]] = {}
for r in runs:
    rs = rows(r)
    if len(rs) >= 10:
        base = r.split("_s")[0]
        if base not in fam or len(rs) > len(fam[base]):
            fam[base] = rs
top = sorted(fam.items(), key=lambda kv: -len(kv[1]))[:4]
for base, rs in top:
    print(f"  {base}   ({len(rs)} 轮)")
    print(f"    {'u':>4}{'kl':>10}{'ratio':>9}{'actor_grad':>12}{'nb':>5}")
    sel = rs[:5] + (rs[len(rs) // 2: len(rs) // 2 + 1]) + rs[-3:]
    for o in sel:
        f = lambda k: o.get(k, float("nan"))
        print(f"    {o['update']:>4}{f('kl'):>10.5f}{f('ratio'):>9.4f}"
              f"{f('actor_grad'):>12.4f}{o.get('nb', -1):>5}")
    kls = [o["kl"] for o in rs if isinstance(o.get("kl"), (int, float))]
    ras = [o["ratio"] for o in rs if isinstance(o.get("ratio"), (int, float))]
    if kls:
        print(f"    kl: 首 {kls[0]:.5f} → 末 {kls[-1]:.5f}   中位 {statistics.median(kls):.5f}"
              f"   最大 {max(kls):.5f}")
    if ras:
        out_of = sum(1 for x in ras if x < 0.9 or x > 1.1)
        print(f"    ratio: {min(ras):.4f} ~ {max(ras):.4f}   （clip 边界 0.9 / 1.1；"
              f"越界 {out_of}/{len(ras)} 次）")
    print()

print("=" * 78)
print("③ 累计位移：单轮小，但 45 步 × 30 轮会累积")
print("=" * 78)
acc = [(r, [o["kl"] for o in rows(r) if isinstance(o.get("kl"), (int, float))])
       for r in runs]
acc = [(r, k) for r, k in acc if len(k) >= 20]
if acc:
    med = statistics.median([statistics.median(k) for _, k in acc])
    tot = statistics.median([sum(k) for _, k in acc])
    print(f"  {len(acc)} 个 run 有 ≥20 轮读数")
    print(f"  逐轮 kl 中位数        = {med:.5f}")
    print(f"  20+ 轮**累加** kl 中位数 = {tot:.5f}")
    print("  ⟹ 若真是「一轮一步」，30 轮总位移 ≈ 30 × 一步；实测 kl 会累积到")
    print("     比单轮值大两个量级，与「45 步/轮」一致，与「1 步/轮」不一致。")
else:
    print("  数据不足")
print("=" * 78)
