#!/usr/bin/env python
"""量 checkpoint 之间的权重变化：训练到底改没改模型？

对两两 checkpoint 算每个参数的相对 L2 变化 ||Δ||/||w||，
按 actor / critic 分组汇总。若 critic 大幅变化而 actor 几乎不动，
就说明 actor 拿到的是无效梯度 —— 与 KL≈0.001 一致。
"""
import sys
from pathlib import Path

import torch

MAIN = Path("/opt/qkd/graph_mappo/outputs")


def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        for key in ("model_state", "model", "state_dict", "model_state_dict", "policy"):
            if key in ck and isinstance(ck[key], dict):
                return ck[key]
    return ck


def rel_delta(a, b):
    """逐参数相对变化，返回 {name: (rel, abs_norm)}。"""
    out = {}
    for k in a:
        if k not in b:
            continue
        x, y = a[k], b[k]
        if not torch.is_tensor(x) or not torch.is_tensor(y):
            continue
        if x.shape != y.shape or not x.is_floating_point():
            continue
        d = (y.float() - x.float()).norm().item()
        n = x.float().norm().item()
        out[k] = (d / n if n > 1e-12 else 0.0, d)
    return out


def group(name):
    low = name.lower()
    if "value" in low or "critic" in low:
        return "critic"
    if "actor" in low or "policy" in low:
        return "actor"
    return "other"


def summarize(label, a, b):
    r = rel_delta(a, b)
    if not r:
        print(f"{label}: 无可比参数")
        return
    buckets = {}
    for k, (rel, ab) in r.items():
        buckets.setdefault(group(k), []).append((k, rel, ab))
    print(f"=== {label} ===")
    for g in ("actor", "critic", "other"):
        items = buckets.get(g)
        if not items:
            continue
        rels = [x[1] for x in items]
        print(f"  {g:<7} n={len(items):<4} 相对L2变化 均值={sum(rels)/len(rels):.3e} "
              f"最大={max(rels):.3e}")
        for k, rel, ab in sorted(items, key=lambda x: -x[1])[:3]:
            print(f"      {k:<52} {rel:.3e}")
    print()


pairs = [
    ("初始 -> r6_base 第20轮(final)",
     MAIN / "supervised_pg_phased/supervised_pg_phased_latest.pt",
     MAIN / "r6_base/checkpoint_final.pt"),
    ("r6_base 第5轮 -> 第10轮",
     MAIN / "r6_base/checkpoint_update_000005.pt",
     MAIN / "r6_base/checkpoint_update_000010.pt"),
    ("r6_base 第15轮 -> 第20轮",
     MAIN / "r6_base/checkpoint_update_000015.pt",
     MAIN / "r6_base/checkpoint_update_000020.pt"),
    ("初始 -> r7_fix_ent 第20轮(final)  [entropy_coef 0.01]",
     MAIN / "supervised_pg_phased/supervised_pg_phased_latest.pt",
     MAIN / "r7_fix_ent/checkpoint_final.pt"),
]

for label, pa, pb in pairs:
    if not pa.exists() or not pb.exists():
        print(f"!! 缺文件: {label}\n   {pa}\n   {pb}\n")
        continue
    summarize(label, load(pa), load(pb))

# 顺带打印一次参数名全貌，便于后续定位
print("=== 参数名全貌 ===")
sd = load(MAIN / "r6_base/checkpoint_final.pt")
for k, v in sd.items():
    if torch.is_tensor(v):
        print(f"  {k:<60} {tuple(v.shape)}")
