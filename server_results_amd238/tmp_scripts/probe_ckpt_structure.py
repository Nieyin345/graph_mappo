#!/usr/bin/env python
"""检查 checkpoint 的顶层结构。"""
import torch
from pathlib import Path

MAIN = Path("/opt/qkd/graph_mappo/outputs")
for rel in ["r6_base/checkpoint_final.pt",
            "r6_base/checkpoint_update_000005.pt",
            "supervised_pg_phased/supervised_pg_phased_latest.pt"]:
    p = MAIN / rel
    print(f"=== {rel} ===")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print(f"  顶层类型: {type(ck).__name__}")
    if isinstance(ck, dict):
        for k, v in ck.items():
            if torch.is_tensor(v):
                print(f"    {k:<44} Tensor{tuple(v.shape)}")
            elif isinstance(v, dict):
                print(f"    {k:<44} dict(n={len(v)})  样例键: {list(v)[:3]}")
            else:
                s = repr(v)
                print(f"    {k:<44} {type(v).__name__} {s[:70]}")
    print()
