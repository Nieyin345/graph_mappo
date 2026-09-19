#!/usr/bin/env python
"""把「双重编码」损坏的文件还原：UTF-8 字节被当 GBK 读后又存成了 UTF-8。

还原 = text.encode('gbk').decode('utf-8')。
默认演练（只打印结论）；加 --apply 才落盘。
"""
import sys
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
TARGETS = [
    "docs/启发式与强化学习算法说明.md",
    "docs/方法原理与实验总结.md",
    "README.md",
    "docs/BFS引导强化学习预训练汇报.md",
    "configs/README.md",
    "scripts/eval/README.md",
]
APPLY = "--apply" in sys.argv

for rel in TARGETS:
    p = ROOT / rel
    if not p.exists():
        print(f"!! 缺失 {rel}")
        continue
    raw = p.read_text(encoding="utf-8")
    try:
        fixed = raw.encode("gbk").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError) as e:
        print(f"!! {rel}: 无法往返（{type(e).__name__}: {e}）")
        continue
    print(f"=== {rel} ===")
    print(f"  长度 {len(raw)} -> {len(fixed)}")
    print(f"  修复前: {raw[:90]!r}")
    print(f"  修复后: {fixed[:90]!r}")
    if APPLY:
        p.write_text(fixed, encoding="utf-8")
        print("  [已写回]")
    print()
