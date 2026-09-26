"""PPO gain over BC, both windows. BC is seed-INDEPENDENT (one shared
screen_bc per model), so every training seed's PPO is paired against the SAME
BC-start vector; the replication unit is the training seed (n=3, crit 4.303).

Usage (server): /opt/qkd/venv/bin/python .tmp/cmp_bc_gain.py
"""
from __future__ import annotations
import json, math, statistics as st
from pathlib import Path

FRZ = Path("outputs/frozen_eval")
BCE = Path("outputs/bc_eval")
SEEDS = (42, 43, 44)


def rd(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def bc(model, win="A"):
    """The shared BC-start evaluation for this model (seed-independent)."""
    return rd(BCE / f"rl_outputs_experiments_{model}_screen_bc_{win}.json")


def ppo(model, tag, seed, win):
    return rd(FRZ / f"rl_outputs_experiments_{model}_{tag}_s{seed}_{win}.json")


ARMS = [("v3", "v3", "scr240_v3"), ("v6", "v6", "scr240_v6")]

print("=== 窗 A/B：BC 起点 vs PPO 终点（专家锚 A 0.7674 / B 0.8674）===")
for label, model, tag in ARMS:
    b = bc(model, "A")
    bm = b["mean_success_rate"] if b else None
    print(f"\n[{label}]  BC 起点(窗A) = {bm:.4f}" if bm else f"\n[{label}]  BC 起点缺失")
    for win in ("A", "B"):
        ps = [ppo(model, tag, s, win) for s in SEEDS]
        ps = [p for p in ps if p]
        if not ps:
            continue
        m = st.mean(p["mean_success_rate"] for p in ps)
        print(f"  窗{win}: PPO {m:.4f}  逐种子 "
              f"{[round(p['mean_success_rate'], 4) for p in ps]}")

print()
print("=== PPO 增益（窗 A；每个训练种子 vs 同一份 BC）===")
for label, model, tag in ARMS:
    b = bc(model, "A")
    if not b:
        print(f"  {label}: 缺 BC 起点"); continue
    rb = b["per_seed_success"]
    gains = []
    for s in SEEDS:
        p = ppo(model, tag, s, "A")
        if not p:
            continue
        rp = p["per_seed_success"]
        if len(rp) != len(rb):
            print(f"  {label} s{s}: 种子数不匹配 {len(rp)} vs {len(rb)}"); continue
        gains.append(sum(x - y for x, y in zip(rb, rp)) / len(rb))
    n = len(gains)
    if n == 0:
        continue
    m = sum(gains) / n
    if n > 1:
        sd = st.stdev(gains); se = sd / math.sqrt(n)
        t = m / se if se else float("inf")
        crit = {1: 12.706, 2: 4.303, 3: 3.182}.get(n - 1, 1.96)
        v = "可分辨" if abs(t) >= crit else "**测不出**"
        print(f"  {label}: Δ={m:+.4f}  逐种子 {[round(x,4) for x in gains]}  "
              f"t={t:+.2f} (n={n}, 临界 {crit}) {v}")
    else:
        print(f"  {label}: Δ={m:+.4f} (n={n})")
