#!/usr/bin/env python
"""定位损坏文件里第一个无法用 GBK 回编的字符，并试其它候选编码。"""
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
REL = "scripts/eval/README.md"
raw = (ROOT / REL).read_text(encoding="utf-8")

bad = None
for i, ch in enumerate(raw):
    try:
        ch.encode("gbk")
    except UnicodeEncodeError:
        bad = (i, ch)
        break
print(f"第一个非 GBK 字符: 位置 {bad[0]}  U+{ord(bad[1]):04X}  码位名={__import__('unicodedata').name(bad[1], '?')}")
print(f"  上下文: {raw[max(0,bad[0]-25):bad[0]+25]!r}")
print()

# 统计所有非法字符
from collections import Counter
cnt = Counter()
for ch in raw:
    try:
        ch.encode("gbk")
    except UnicodeEncodeError:
        cnt[ch] += 1
print(f"非法字符种类 {len(cnt)}，共 {sum(cnt.values())} 个：")
for ch, n in cnt.most_common(12):
    print(f"  U+{ord(ch):04X}  {n:>4}  {__import__('unicodedata').name(ch,'?')}")
print()

for codec in ("gbk", "gb18030", "cp936", "big5"):
    try:
        fixed = raw.encode(codec).decode("utf-8")
        print(f"[OK] {codec}: {fixed[:70]!r}")
    except Exception as e:
        print(f"[--] {codec}: {type(e).__name__}: {str(e)[:80]}")
