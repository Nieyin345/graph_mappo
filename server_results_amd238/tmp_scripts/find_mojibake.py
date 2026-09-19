#!/usr/bin/env python
"""检测仓库里有多少文本文件是「双重编码」损坏的（UTF-8 被当 GBK 读后又存成 UTF-8）。

判据：文件是合法 UTF-8，但内容里高频出现「锛」「銆」「鐨」「璇」这类
GBK 双字节被误读产生的特征字。
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
MARKERS = "锛銆鐨璇鍜屾垜浜嗕綘鏄涓嶅湪鏈夎繖"
EXTS = {".md", ".py", ".yaml", ".yml", ".txt", ".json", ".sh", ".rst"}

hits, total = [], 0
for p in ROOT.rglob("*"):
    if not p.is_file() or p.suffix.lower() not in EXTS:
        continue
    if any(part in {".git", "__pycache__", "outputs", "node_modules", ".tmp"} for part in p.parts):
        continue
    total += 1
    try:
        t = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    if not t:
        continue
    n = sum(t.count(c) for c in MARKERS)
    if n >= 3:
        hits.append((n, str(p.relative_to(ROOT))))

hits.sort(reverse=True)
print(f"扫描 {total} 个文本文件，检出双重编码损坏 {len(hits)} 个：\n")
for n, rel in hits:
    print(f"  {n:>6}  {rel}")
