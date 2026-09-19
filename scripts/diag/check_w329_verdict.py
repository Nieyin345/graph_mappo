#!/usr/bin/env python
"""w329（训练窗口 0-295 → 0-329）到底有没有判读过？

### 为什么查这个

`configs/train_window_329.yaml` 论证得很完整（实测 296-329 那 34 天与训练窗
统计同质、且策略难度也同质），它的**唯一收益是"+11.5% 起日多样性"**。
服务器上 `outputs/w329_s62/s62r/s63/s63r/s64` 五个 run 都存在。

但 `docs/训练诊断记录.md` 里搜 "w329" 只有三处，且全在 OOM/infra 语境里，
**没有一条判读**。要么是做了没记，要么是被 OOM 打断了没做成。

这个脚本量出结论：w329 各臂 vs 同种子基线，逐种子配对。

### 配对对象怎么选

w329 是"纯数据侧、不动算法/模型/超参"的改动，所以对照应当是
**同配置链、同 BC 起点、同种子**的非 w329 run。脚本会把每个 w329 臂
按种子去找对应的对照（优先 `ent01_s*`，其次列出来让人看）。

用法（服务器上）：/opt/qkd/venv/bin/python /tmp/check_w329_verdict.py
"""
from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
PLATEAU = (25, 30)


def points(run: str) -> dict[int, list[float]]:
    p = OUT / run / "metrics.jsonl"
    out: dict[int, list[float]] = {}
    if not p.exists():
        return out
    last = None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
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
            out[last] = [float(x) for x in ev["per_seed_success"]]
    return out


def plateau(run: str):
    pts = points(run)
    us = [u for u in PLATEAU if u in pts]
    if not us:
        return None
    n = len(pts[us[0]])
    if any(len(pts[u]) != n for u in us):
        return None
    return [statistics.mean(pts[u][i] for u in us) for i in range(n)]


print("=" * 78)
print("w329 有没有可判读的数据")
print("=" * 78)

w = sorted(d.name for d in OUT.glob("w329*") if d.is_dir())
if not w:
    print("  ✗ outputs 里没有 w329* —— 从未跑过")
    raise SystemExit(0)

print(f"  {len(w)} 个 w329 run：")
for r in w:
    p = plateau(r)
    npts = len(points(r))
    tail = ""
    if p is None:
        tail = "  ← **无 u25/u30 平台读数，不可判读**"
    else:
        tail = f"  平台 {statistics.mean(p):.4f}"
    print(f"    {r:<16} eval点 {npts:>2}{tail}")

print()
print("=" * 78)
print("按种子找对照（同配置链、同 BC、同种子；优先 ent01_*）")
print("=" * 78)
cands = {}
for d in sorted(OUT.iterdir()):
    if not d.is_dir():
        continue
    m = re.match(r"^(ent01|base|r8_base|r6_base)_s(\d+)(_r\d+)?$", d.name)
    if m:
        cands.setdefault(int(m.group(2)), []).append(d.name)

diffs, used = [], []
for r in w:
    m = re.search(r"w329_s(\d+)", r)
    if not m:
        continue
    s = int(m.group(1))
    pw = plateau(r)
    if pw is None:
        print(f"  {r}: 无平台读数，跳过")
        continue
    pool = cands.get(s, [])
    if not pool:
        print(f"  {r}: 种子 {s} 没有对照（现有对照种子：{sorted(cands)}）")
        continue
    ref = max(pool, key=lambda x: len(points(x)))
    pr = plateau(ref)
    if pr is None:
        print(f"  {r}: 对照 {ref} 无平台读数")
        continue
    if len(pw) != len(pr):
        print(f"  {r}: 种子数不一致（{len(pw)} vs {len(pr)}），跳过")
        continue
    d = [b - a for a, b in zip(pr, pw)]
    dm = statistics.mean(d)
    diffs.append(dm)
    used.append((r, ref, s))
    print(f"  {r:<16} vs {ref:<12} Δ = {dm:+.4f}   "
          f"（{statistics.mean(pw):.4f} − {statistics.mean(pr):.4f}）")

print()
print("=" * 78)
print("判读")
print("=" * 78)
if len(diffs) >= 2:
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(len(diffs))
    df = len(diffs) - 1
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(df, 2.0)
    t = m / se if se > 0 else float("nan")
    print(f"  n={len(diffs)}（{len({u[2] for u in used})} 个不同种子）"
          f"  Δ = {m:+.4f}  SD = {sd:.4f}  SE = {se:.4f}")
    print(f"  t = {t:+.3f}  (df={df}, 临界 {tcrit})")
    try:
        from scipy import stats
        print(f"  p = {2 * float(stats.t.sf(abs(t), df)):.4f}")
    except ImportError:
        pass
    if len({u[2] for u in used}) < 3:
        print("  ⚠ 不同种子数 < 3 ⟹ 分辨率不足，只能说方向")
    elif abs(t) >= tcrit:
        print("  ★ 显著 ⟹ 扩窗确实改变结果")
    else:
        print(f"  ✗ |t| < {tcrit} ⟹ 以 n={len(diffs)} 分辨率测不出")
else:
    print(f"  ✗ 只有 n={len(diffs)} 个可比对 ⟹ **无法判读**")
    print("    这解释了为什么日志里没有 w329 的判读：数据不足以支持一条结论。")
print("=" * 78)
