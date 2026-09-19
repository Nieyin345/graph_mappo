# -*- coding: utf-8 -*-
"""训练更久有用吗？—— 两个已有的免费证据合起来看。

A. ent01 的验证曲线 u5→u30 是还在爬，还是已经平台？
   （平台 ⟹ 加轮数没用；还在爬 ⟹ 加轮数是最直接的杠杆）
B. outputs/ent01_s44_u30to50 —— 从 u30 续跑到 u50，直接看 u30→u50 的增量。

B 特别有价值：它**已经跑完了**，不需要新实验，而且正好回答"续训有没有用"。
注意它的种子必须与对照一致才能配对。
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
EXPERT = OUT / "eval" / "expert_seeds100_240.json"
d = json.loads(EXPERT.read_text(encoding="utf-8"))
ex = dict(zip(d["seeds"], (float(x) for x in d["success"])))
base = [ex[s] for s in sorted(ex)]


def load_val(name):
    p = OUT / name / "metrics.jsonl"
    if not p.exists():
        return None
    out, last = {}, None
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
    return out or None


def delta(v):
    return st.mean([v[i] - base[i] for i in range(len(base))])


print("=" * 88)
print("A. ent01 三个种子的验证曲线（对专家的配对差）")
print("=" * 88)
runs = ["ent01_s42", "ent01_s43", "ent01_s44"]
curves = {}
for r in runs:
    c = load_val(r)
    if c:
        curves[r] = c
    else:
        print(f"  {r}: 无数据")
if curves:
    us = sorted(set.intersection(*[set(c) for c in curves.values()]))
    print(f"  共同验证轮: {us}")
    print(f"  {'轮':>4}" + "".join(f"{r[-3:]:>12}" for r in curves) + f"{'均值':>12}")
    for u in us:
        cells = [delta(curves[r][u]) for r in curves]
        print(f"  {u:>4}" + "".join(f"{c:>+12.4f}" for c in cells)
              + f"{st.mean(cells):>+12.4f}")
    print()
    if len(us) >= 4:
        first = st.mean([delta(curves[r][us[0]]) for r in curves])
        mid = st.mean([delta(curves[r][us[len(us)//2]]) for r in curves])
        last = st.mean([delta(curves[r][us[-1]]) for r in curves])
        print(f"  首点 {us[0]}: {first:+.4f}   中点 {us[len(us)//2]}: {mid:+.4f}"
              f"   末点 {us[-1]}: {last:+.4f}")
        print(f"  后半程增量（中点→末点）= {last - mid:+.4f}")
        print(f"  ⟹ {'**仍在爬**：加轮数是最直接的杠杆' if last - mid > 0.005 else '**已平台**：单纯加轮数没用'}")

print()
print("=" * 88)
print("B. ent01_s44_u30to50（从 u30 续跑到 u50，已有数据）")
print("=" * 88)
ext = load_val("ent01_s44_u30to50")
if ext:
    us = sorted(ext)
    print(f"  验证轮: {us}")
    for u in us:
        print(f"    u{u:<4} 对专家 {delta(ext[u]):+.4f}")
    rc = OUT / "ent01_s44_u30to50" / "resolved_config.yaml"
    if rc.exists():
        txt = rc.read_text(encoding="utf-8", errors="replace")
        for key in ("global_seed", "num_updates", "entropy_coef"):
            for ln in txt.splitlines():
                if key in ln:
                    print(f"    [{key}] {ln.strip()}")
                    break
else:
    print("  无数据")

print()
print("=" * 88
      )
print("C. 与 ent01_s44 的 u30 对照（确认续跑确实是从同一状态出发）")
print("=" * 88)
if ext and "ent01_s44" in curves:
    c44 = curves["ent01_s44"]
    for u in sorted(set(c44) & set(ext)):
        print(f"  u{u:<4} 原始 {delta(c44[u]):+.4f}   续跑 {delta(ext[u]):+.4f}"
              f"   差 {delta(ext[u]) - delta(c44[u]):+.4f}")
