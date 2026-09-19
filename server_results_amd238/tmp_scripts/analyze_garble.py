#!/usr/bin/env python
"""不猜编码链，直接看数据：损坏文档里到底有哪些字符。"""
from collections import Counter
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo")
TARGETS = [
    "README.md",
    "configs/README.md",
    "docs/方法原理与实验总结.md",
]

# 候选解码器：把乱码字符编回字节时试这些
import codecs
CANDIDATES = ["gbk", "gb18030", "gb2312", "big5", "cp950", "shift_jis",
              "euc_jp", "euc_kr", "cp932", "latin-1", "cp1252", "cp1251"]

for rel in TARGETS:
    p = ROOT / rel
    raw = p.read_text(encoding="utf-8")
    print(f"\n{'='*70}\n=== {rel}  长度 {len(raw)} ===")
    hist = Counter(ord(c) for c in raw if ord(c) > 0x7F)
    print(f"  非 ASCII 字符 {sum(hist.values())} 个，去重 {len(hist)} 种")
    print("  最常见 25 种：")
    for cp, n in hist.most_common(25):
        ch = chr(cp)
        try:
            b = ch.encode("gbk")
            bs = b.hex(" ")
        except Exception as e:
            bs = f"<{type(e).__name__}>"
        print(f"    U+{cp:04X} {ch!r:>10} ×{n:<5} gbk={bs}")
    # 试各解码器能否把它编回字节
    print("  各候选编码器可编码率：")
    tot = sum(hist.values())
    for enc in CANDIDATES:
        ok = 0
        for cp, n in hist.items():
            try:
                chr(cp).encode(enc)
                ok += n
            except Exception:
                pass
        print(f"    {enc:<10} {ok}/{tot} = {ok/max(1,tot):.1%}")
    # 样本
    print("  样本（第 3-8 行）：")
    for line in raw.splitlines()[2:8]:
        print(f"    {line[:90]}")
