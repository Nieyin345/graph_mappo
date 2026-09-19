#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""u1 指纹大盘：把所有 run 的第一行 mean_success_rate 摊开，看谁与谁同簇。

u1 发生在**第一次梯度步之前**，只取决于 **权重 + 种子 + 代码 + 硬件**
（与臂自己的超参无关，也与线程数无关——记忆 `node-parallelism-profile`）。
所以：同 BC + 同种子 + 同节点 + 同代码 的臂 **必须 u1 逐位相同**。
不同就说明代码或权重有差异，配对不干净。
"""
import json
import os
import re
import sys
from collections import defaultdict

R = "/opt/qkd/graph_mappo/outputs"


def u1(d):
    p = os.path.join(R, d, "metrics.jsonl")
    try:
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                o = json.loads(ln)
                return o.get("mean_success_rate"), o.get("seed")
    except OSError:
        pass
    return None, None


def cfgfield(d, key):
    p = os.path.join(R, d, "resolved_config.yaml")
    try:
        for ln in open(p, encoding="utf-8"):
            m = re.match(r"\s*%s:\s*(\S+)" % key, ln)
            if m:
                return m.group(1)
    except OSError:
        pass
    return "?"


rows = []
for d in sorted(os.listdir(R)):
    if not os.path.isdir(os.path.join(R, d)):
        continue
    v, seed = u1(d)
    if v is None:
        continue
    rows.append((v, d, seed))

# 按 u1 值聚类（同一指纹的放一起）
by = defaultdict(list)
for v, d, seed in rows:
    by[round(v, 9)].append(d)

print("=== u1 指纹聚类（同指纹 = 同权重+同种子+同代码+同硬件）===")
for v in sorted(by, reverse=True):
    ds = by[v]
    print("  %.10f  ×%d" % (v, len(ds)))
    for d in sorted(ds):
        print("       %s" % d)

print()
print("=== 只看本次关心的一组（seed 42）===")
for v, d, seed in sorted(rows, reverse=True):
    if d.endswith("_s42") or "s42" in d:
        print("  %-26s u1=%.10f  seed=%s" % (d, v, seed))
