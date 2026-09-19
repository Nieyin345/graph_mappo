#!/usr/bin/env python
"""还原「双重编码」损坏的文档：UTF-8 字节被当 cp936 读后又存成了 UTF-8。

为什么普通 `encode('gbk')` 不行：Windows 的 cp936 解码器遇到**未定义**的
双字节序列时，不报错，而是映射到 Unicode 私用区（PUA）U+E000+offset。
所以损坏的文件里有 65 个 PUA 字符，回编时必须把它们映射回原始字节。

映射规则（微软官方 Best-Fit / PUA 表）：
  U+E000–U+E4C5  对应单字节 0x80–0xA0 及部分双字节起始
  U+E4C6–U+E765  对应 0xA1–0xFE 的双字节第二字节等
本脚本用「查表 + 数值还原」两条路径，对每个 PUA 字符取可用的那条。

默认演练；--apply 落盘。
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


def build_pua_tables():
    """从 Python 的 cp936 编解码器反向构造 PUA -> 原始字节 的表。

    做法：对 cp936 能表示的所有字节序列，解码后若落到 PUA，就记下
    「该 PUA 字符 -> 该字节序列」。这样不用手抄微软的表。
    """
    import codecs
    dec = codecs.getdecoder("cp936")
    table = {}
    # 单字节 0x80-0xFF 与全部双字节组合
    seqs = [bytes([b]) for b in range(0x80, 0x100)]
    seqs += [bytes([a, b]) for a in range(0x81, 0xFF) for b in range(0x40, 0xFF)]
    for raw in seqs:
        try:
            ch, _ = dec(raw, "strict")
        except (UnicodeDecodeError, ValueError):
            continue
        if len(ch) == 1 and 0xE000 <= ord(ch) <= 0xF8FF:
            table.setdefault(ch, raw)
    return table


PUA2RAW = build_pua_tables()
print(f"cp936 PUA 映射表：{len(PUA2RAW)} 条")


def ungarble(text):
    """把损坏文本还原成原始 UTF-8 字节，再解码。"""
    out = bytearray()
    for ch in text:
        if ord(ch) < 0x80:
            out.extend(ch.encode("ascii"))
        elif ch in PUA2RAW:
            out.extend(PUA2RAW[ch])
        else:
            try:
                out.extend(ch.encode("gbk"))
            except UnicodeEncodeError:
                return None, f"无法编码 U+{ord(ch):04X}"
    try:
        return out.decode("utf-8"), None
    except UnicodeDecodeError as e:
        return None, f"UTF-8 解码失败 @ {e.start}: {e.reason}"


for rel in TARGETS:
    p = ROOT / rel
    if not p.exists():
        print(f"!! 缺失 {rel}")
        continue
    raw = p.read_text(encoding="utf-8")
    fixed, err = ungarble(raw)
    n_pua = sum(1 for c in raw if 0xE000 <= ord(c) <= 0xF8FF)
    print(f"\n=== {rel} ===")
    print(f"  PUA 字符 {n_pua} 个，长度 {len(raw)}")
    if fixed is None:
        print(f"  !! 失败：{err}")
        continue
    print(f"  修复后长度 {len(fixed)}")
    print(f"  修复前: {raw[:70]!r}")
    print(f"  修复后: {fixed[:70]!r}")
    if APPLY:
        p.write_text(fixed, encoding="utf-8")
        print("  [已写回]")
