#!/usr/bin/env python
"""打印 configs/*.yaml 顶层键的并集，用于校准 check_configs.py 的 KNOWN_TOP。

一个总是触发的警告等于没有警告——所以白名单要按实际配置校准。
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path("/opt/qkd/graph_mappo")
keys: dict[str, list[str]] = {}
for p in sorted((ROOT / "configs").glob("*.yaml")):
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if isinstance(d, dict):
        for k in d:
            keys.setdefault(k, []).append(p.name)

for k in sorted(keys):
    print(f"{k:<20} {len(keys[k]):>3} 个文件   e.g. {keys[k][0]}")
print()
print("python 集合字面量：")
print("{")
for k in sorted(keys):
    print(f'    "{k}",')
print("}")
