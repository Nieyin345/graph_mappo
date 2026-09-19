#!/usr/bin/env python
"""设计下一轮实验前的事实核对（服务器上跑）。只读，不改任何东西。

要回答三个问题：
  1. 每轮训练实际耗时多久 → 决定 15 小时能排几浪；
  2. `train.ppo.epochs` 这个字段**真的被代码读了吗** → 否则整个实验是空跑；
  3. ent01 三条臂的耗时与最终验证值 → 作为新臂的对照基线。
"""
import json
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def rows(run):
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
        except Exception:
            pass
    return out


print("=== 1. ent01 三条臂的每轮耗时（秒）===")
for s in (42, 43, 44):
    rs = rows(f"ent01_s{s}")
    tr = [r for r in rs if "update" in r]
    if not tr:
        print(f"  ent01_s{s}: 无数据")
        continue
    # 前 3 轮（受缓存/预热影响）与后 3 轮分开看
    def avg(key, sel):
        v = [r.get(key) for r in sel if isinstance(r.get(key), (int, float))]
        return sum(v) / len(v) if v else float("nan")
    last = tr[-1]
    print(f"  ent01_s{s}: 共 {len(tr)} 轮  末轮 update={last.get('update')}")
    print(f"    前3轮  rollout_s {avg('rollout_s', tr[:3]):7.1f}  "
          f"update_s {avg('update_s', tr[:3]):7.1f}  "
          f"elapsed_s {avg('elapsed_s', tr[:3]):8.1f}")
    print(f"    后3轮  rollout_s {avg('rollout_s', tr[-3:]):7.1f}  "
          f"update_s {avg('update_s', tr[-3:]):7.1f}  "
          f"elapsed_s {avg('elapsed_s', tr[-3:]):8.1f}")
    tot = sum(r.get("elapsed_s", 0) for r in tr if isinstance(r.get("elapsed_s"), (int, float)))
    print(f"    合计 elapsed_s ≈ {tot:.0f}s = {tot/3600:.2f}h  "
          f"→ 每轮均值 {tot/len(tr):.0f}s")

print()
print("=== 2. 训练总墙钟（从日志首尾时间戳，含收尾）===")
for s in (42, 43, 44):
    lg = Path(f"/tmp/ent01_s{s}.log")
    if not lg.exists():
        print(f"  ent01_s{s}: 无日志")
        continue
    txt = lg.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"  ent01_s{s}: {len(txt)} 行")

print()
print("=== 3. epochs 字段是否真的被代码读 ===")
import re
src_root = Path("/opt/qkd/graph_mappo")
hits = []
for f in src_root.rglob("*.py"):
    if "/.git/" in str(f) or "__pycache__" in str(f):
        continue
    try:
        t = f.read_text(encoding="utf-8", errors="replace")
    except OSError:
        continue
    for i, ln in enumerate(t.splitlines(), 1):
        if re.search(r'\[\s*["\']epochs["\']\s*\]|get\(\s*["\']epochs["\']|epochs\s*=', ln):
            hits.append((str(f.relative_to(src_root)), i, ln.strip()[:120]))
for h in hits[:25]:
    print(f"  {h[0]}:{h[1]}  {h[2]}")
if not hits:
    print("  ⚠ 没有任何地方读 epochs —— 若是这样，改这个字段是空跑")

print()
print("=== 4. 确认 value_coef / minibatch_size 被读 ===")
for key in ("value_coef", "minibatch_size", "target_kl"):
    n = 0
    for f in src_root.rglob("*.py"):
        if "/.git/" in str(f) or "__pycache__" in str(f):
            continue
        try:
            t = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        n += len(re.findall(rf'["\']{key}["\']', t))
    print(f"  {key}: 代码里出现 {n} 次")
