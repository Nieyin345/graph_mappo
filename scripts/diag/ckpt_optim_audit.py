# -*- coding: utf-8 -*-
"""BC checkpoint 的 optimizer_state 里到底有什么？

三个问题，都直接影响"热启动是否真的热"：
  1. param_groups 有几组？各组 lr 是多少？（actor_lr bug 的现场）
  2. Adam 的一二阶矩（exp_avg / exp_avg_sq）在不在、非不非空？
     —— 若为空，所谓"热启动"其实只是权重热启动，优化器是冷的。
  3. defaults（betas/eps/weight_decay）存的是什么？
     —— Optimizer.load_state_dict **连 defaults 一起恢复**，所以这些也会盖掉配置。

注：本脚本只读不写，不加载模型到 GPU。
"""
from __future__ import annotations

import torch
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
cands = [
    ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt",
    ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_best.pt",
]
cands += sorted((ROOT / "outputs").glob("*/checkpoint_final.pt"))[:3]

for p in cands:
    if not p.exists():
        print(f"--- {p.name}: 不存在")
        continue
    print("=" * 78)
    print(f"{p.relative_to(ROOT)}  ({p.stat().st_size/1048576:.1f} MB)")
    print("=" * 78)
    try:
        blob = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"  读取失败: {exc}")
        continue

    if not isinstance(blob, dict):
        print(f"  类型 {type(blob)}，跳过")
        continue
    print(f"  顶层键: {sorted(blob.keys())[:12]}")

    opt = blob.get("optimizer_state")
    if opt is None:
        print("  optimizer_state: **None** → 热启动时优化器是全新的")
        continue

    print(f"  optimizer_state 是 {type(opt).__name__}")
    if isinstance(opt, dict):
        pg = opt.get("param_groups")
        st = opt.get("state")
        print(f"    param_groups: {len(pg) if pg else 0} 组")
        if pg:
            for i, g in enumerate(pg):
                keys = {k: v for k, v in g.items() if k != "params"}
                npar = len(g.get("params", []))
                print(f"      组{i}: {npar} 个参数张量, {keys}")
        print(f"    defaults: {opt.get('defaults')}")
        if st:
            nz = 0
            shapes = []
            for k, v in list(st.items())[:3]:
                if isinstance(v, dict):
                    ea = v.get("exp_avg")
                    if ea is not None and hasattr(ea, "numel"):
                        nz += int(ea.abs().sum() > 0)
                        shapes.append(tuple(ea.shape))
            print(f"    state: {len(st)} 个条目；前 3 个的 exp_avg 形状 {shapes}")
            print(f"    state 里 exp_avg 非零的条目数（前3）: {nz}")
        else:
            print("    state: **空** → 有 lr 但没有矩，等于冷的")
    print()
