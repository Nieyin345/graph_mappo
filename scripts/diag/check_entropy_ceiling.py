#!/usr/bin/env python
"""熵 4.2 nats 到底是不是"接近均匀分布"？

### 为什么这个问题重要

`entropy` 是**每个决策的均值**，决策 = 在「当时可行的弧 + STOP」上做 softmax。
若有 K 个选项，最大熵 = ln K。所以：

  · 若 K ≈ 66，则 ln 66 ≈ 4.19 ⟹ **4.2 就是均匀分布**，
    策略在可行集上几乎等概率 —— 那是"没学到"，不是"探索得好"。
  · 若 K >> 66，则 4.2 相对很小，策略其实是尖的。

**这两种情况的结论完全相反**，所以必须先量 K。

数据来源：`rollout_debug.jsonl`（每轮 rollout 的分解，免费）。
先看它有哪些键，再找弧数/匹配数的读数。

用法（服务器上）：/opt/qkd/venv/bin/python /tmp/check_entropy_ceiling.py
"""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def first_jsonl(run: str, fn: str, n: int = 1):
    p = OUT / run / fn
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(out) >= n:
            break
    return out


print("=" * 78)
print("① rollout_debug.jsonl 有哪些键")
print("=" * 78)
for run in ("ent01_s42", "ep2e1_s42"):
    rows = first_jsonl(run, "rollout_debug.jsonl", 1)
    if not rows:
        print(f"  {run}: 无 rollout_debug.jsonl")
        continue
    o = rows[0]
    print(f"  {run}:")
    for k in sorted(o):
        v = o[k]
        if isinstance(v, (int, float)):
            print(f"    {k:<36} {v}")
        elif isinstance(v, (list, tuple)):
            print(f"    {k:<36} list[{len(v)}]  例 {v[:3]}")
        elif isinstance(v, dict):
            print(f"    {k:<36} dict{list(v)[:6]}")
        else:
            print(f"    {k:<36} {type(v).__name__}")
    print()

print("=" * 78)
print("② 找「每决策可选弧数」的读数")
print("=" * 78)
CAND = ("n_arcs", "num_arcs", "arcs", "n_feasible", "feasible", "candidates",
        "n_candidates", "choices", "n_choices", "matched", "n_matched",
        "edges", "n_edges", "decisions", "n_decisions", "stop")
for run in ("ent01_s42", "ep2e1_s42"):
    rows = first_jsonl(run, "rollout_debug.jsonl", 1)
    if not rows:
        continue
    o = rows[0]
    hits = {k: o[k] for k in o if any(c in k.lower() for c in CAND)}
    print(f"  {run}: {hits if hits else '（没有明显相关的键）'}")

print()
print("=" * 78)
print("③ 所有 round 里这些键的分布（看是否稳定）")
print("=" * 78)
run = "ent01_s42"
p = OUT / run / "rollout_debug.jsonl"
if p.exists():
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    print(f"  {run}: {len(rows)} 轮")
    if rows:
        CAND_KEYS = [k for k in rows[0]
                     if any(c in k.lower() for c in CAND)
                     and isinstance(rows[0][k], (int, float))]
        for k in CAND_KEYS:
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
            if vals:
                print(f"    {k:<30} 中位 {statistics.median(vals):.2f}  "
                      f"范围 [{min(vals):.2f}, {max(vals):.2f}]")
        if CAND_KEYS:
            print()
            print("  ★ 用上面最大的那个候选数当作 K，算 ln K 与实测熵 4.2 比：")
            for k in CAND_KEYS:
                vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
                if not vals:
                    continue
                K = statistics.median(vals)
                if K > 1:
                    print(f"    K={K:>8.0f} ({k:<24}) ⟹ ln K = {math.log(K):.3f}   "
                          f"实测熵 4.2 占 {4.2 / math.log(K):.1%}")

print()
print("=" * 78)
print("④ 实测熵（从 metrics.jsonl）")
print("=" * 78)
for r in ("ent01_s42", "ent01_s43", "vcoef1_s42"):
    q = OUT / r / "metrics.jsonl"
    if not q.exists():
        continue
    ents = []
    for line in q.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(o.get("entropy"), (int, float)):
            ents.append(o["entropy"])
    if ents:
        print(f"  {r:<14} 首 {ents[0]:.3f}  末 {ents[-1]:.3f}  最大 {max(ents):.3f}")
print("=" * 78)
