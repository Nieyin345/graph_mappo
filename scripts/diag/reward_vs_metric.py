# -*- coding: utf-8 -*-
"""奖励还在涨吗？—— 若是，而 success_rate 不动，那瓶颈是**目标错位**，
不是"学不动了"。

这是对上一节（训练饱和是结构性的）的**机制追问**：
  · 若 reward 也平了 → 学习真的停了（优化到顶）
  · 若 reward 还在涨、success_rate 平 → 策略仍在改善**代理目标**，
    只是这个代理已经与指标脱钩 ⟹ 该改的是**奖励**，不是优化器

数据全在已有日志里（每轮都打 reward= 和 success_rate=）。

再把 reward 按 rollout_debug.jsonl 的分解拆开看是哪一项在涨——
env_full.yaml 的 shaped 奖励有 served/storage/keep_active/failed 等多项，
若涨的是 storage 这类与"服务成功"无关的项，就是直接的错位证据。
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def parse(name):
    p = Path("/tmp") / f"{name}.log"
    if not p.exists():
        return None
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if "update=" not in line or "actor_loss" not in line:
            continue
        d = {}
        for k in ("reward", "success_rate", "entropy", "kl"):
            m = re.search(rf"(?:^|\s){k}=(-?[\d.eE+]+)", line)
            if m:
                d[k] = float(m.group(1))
        if d:
            rows.append(d)
    return rows or None


print("=" * 92)
print("1. reward 与 success_rate 的轨迹（训练 regime）")
print("=" * 92)
FAMS = {
    "ent01": ["ent01_s42", "ent01_s43", "ent01_s44"],
    "vcoef1(ent=0.001)": ["vcoef1_s42", "vcoef1_s43", "vcoef1_s44"],
}
for fam, names in FAMS.items():
    series = [parse(n) for n in names]
    series = [s for s in series if s]
    if not series:
        continue
    L = min(len(s) for s in series)
    print(f"\n  --- {fam}（{len(series)} 个种子，共同长度 {L}）---")
    print(f"  {'轮':>4}{'reward':>14}{'success_rate':>16}")
    for i in range(L):
        if i < 6 or (i + 1) % 5 == 0 or i == L - 1:
            r = [s[i].get("reward") for s in series if s[i].get("reward") is not None]
            c = [s[i].get("success_rate") for s in series
                 if s[i].get("success_rate") is not None]
            print(f"  {i+1:>4}{st.mean(r):>14.3f}"
                  f"{st.mean(c):>16.4f}" if r and c else "")
    # 半段均值
    for key in ("reward", "success_rate"):
        vals = [[s[i].get(key) for i in range(L) if s[i].get(key) is not None]
                for s in series]
        vals = [v for v in vals if v]
        if not vals:
            continue
        h = min(len(v) for v in vals)
        per_seed = [st.mean(v[h // 2:]) - st.mean(v[:h // 2]) for v in vals]
        print(f"    {key:<14} 前半→后半 增量: "
              + "  ".join(f"{x:+.4f}" for x in per_seed)
              + f"   均值 {st.mean(per_seed):+.4f}")

print()
print("=" * 92)
print("2. 奖励分解（rollout_debug.jsonl）—— 涨的是哪一项？")
print("=" * 92)
for name in ("ent01_s42", "ent01_s43", "ent01_s44"):
    p = OUT / name / "rollout_debug.jsonl"
    if not p.exists():
        print(f"  {name}: 无 rollout_debug.jsonl")
        continue
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        continue
    print(f"\n  --- {name}（{len(rows)} 条）---")
    # 找出所有数值键
    def numkeys(o, prefix=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from numkeys(v, f"{prefix}{k}.")
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            yield prefix.rstrip(".")
    keys = sorted(set(numkeys(rows[0])))
    reward_keys = [k for k in keys if "reward" in k.lower() or "served" in k.lower()
                   or "failed" in k.lower() or "storage" in k.lower()
                   or "generated" in k.lower() or "keep" in k.lower()]
    print(f"    数值键 {len(keys)} 个，其中奖励相关 {len(reward_keys)} 个")
    for k in reward_keys[:14]:
        def get(r):
            cur = r
            for part in k.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return None
                cur = cur[part]
            return cur if isinstance(cur, (int, float)) else None
        vals = [get(r) for r in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) < 4:
            continue
        h = len(vals) // 2
        print(f"      {k:<44} 首 {vals[0]:>12.3f}  末 {vals[-1]:>12.3f}"
              f"   半段增量 {st.mean(vals[h:]) - st.mean(vals[:h]):>+12.4f}")
