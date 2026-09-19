#!/usr/bin/env python
"""还原双重编码损坏的文档。

损坏链条：原始 UTF-8 字节 --(按 GBK 解码)--> CJK/PUA 乱码 --(存成 UTF-8)--> 现在这样

还原就是反过来：把每个字符编回 GBK 字节，再按 UTF-8 读。

难点只有一个：Windows 的 cp936 解码器遇到**未定义**的双字节序列时，会把它映射到
Unicode 私用区（PUA，U+E000 起），而不是报错。Python 的 cp936 不做这个映射，
所以拿不到「PUA -> 原字节」的反表。

本脚本不查表，改用**约束求解**：既然还原后的字节必须是合法 UTF-8，那就对每个 PUA
字符试遍所有可能的 1~2 字节组合，用「末尾仍是合法的 UTF-8 前缀」做剪枝，回溯搜索。
真解唯一时必然找到。

另有 ASCII '?'（0x3F）——那是原 GBK 解码器丢弃字节留下的墓碑，**不可恢复**，
只能计数报告。

用法：
  python .tmp/ungarble.py                # 演练，只报告
  python .tmp/ungarble.py --apply        # 写回文件
"""
from __future__ import annotations

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
FULL = "--full" in sys.argv

# PUA 候选字节：优先双字节（GBK 主体），再单字节
_CANDS = [bytes([a, b]) for a in range(0x81, 0xFF) for b in range(0x40, 0x100) if b != 0x7F]
_CANDS += [bytes([b]) for b in range(0x81, 0x100)]
# 按「更像 GBK 主体」排序：双字节在前，且 trail 在常用区间的优先
_CANDS.sort(key=lambda c: (len(c), c[0], abs(c[-1] - 0xA1) if len(c) == 2 else 99))


def tail_ok(buf: bytearray) -> bool:
    """buf 的末尾必须是一个合法的（可能尚未完整的）UTF-8 序列。"""
    n = len(buf)
    if n == 0:
        return True
    k = n - 1
    while k >= 0 and 0x80 <= buf[k] <= 0xBF:
        k -= 1
    if k < 0 or n - 1 - k > 3:
        return False
    b = buf[k]
    if b < 0x80:
        return k == n - 1
    if 0xC2 <= b <= 0xDF:
        need = 2
    elif 0xE0 <= b <= 0xEF:
        need = 3
    elif 0xF0 <= b <= 0xF4:
        need = 4
    else:
        return False
    if n - k > need:
        return False
    # 完整性检查交给最终整体解码，这里只查后续 continuation 的取值域
    for t in range(k + 1, n):
        if not (0x80 <= buf[t] <= 0xBF):
            return False
    return True


def segments(text):
    """切成 [固定字节段 | PUA 槽] 的序列，减少搜索深度。"""
    segs = []
    cur = bytearray()

    def flush():
        if cur:
            segs.append(("fixed", bytes(cur)))
            cur.clear()

    n_pua = 0
    bad = None
    for ch in text:
        o = ord(ch)
        if 0xE000 <= o <= 0xF8FF:
            flush()
            segs.append(("pua", None))
            n_pua += 1
        elif o < 0x80:
            cur.append(o)
        else:
            try:
                cur.extend(ch.encode("gbk"))
            except UnicodeEncodeError:
                flush()
                bad = f"U+{o:04X} 无法编回 GBK"
                break
    flush()
    return segs, n_pua, bad


def solve_simple(segs):
    """深度优先回溯：只在 PUA 处分支，用「末尾仍是合法 UTF-8 前缀」剪枝。"""
    buf = bytearray()
    stack = []
    i = 0
    nxt = 0
    while True:
        if i == len(segs):
            try:
                return bytes(buf).decode("utf-8"), None
            except UnicodeDecodeError:
                pass
            if not stack:
                return None, "无解"
            i, nxt = stack.pop()
            kind, val = segs[i]
            del buf[-len(_CANDS[nxt - 1]):]
            continue
        kind, val = segs[i]
        if kind == "fixed":
            buf.extend(val)
            if tail_ok(buf):
                i += 1
                nxt = 0
                continue
            del buf[-len(val):]
            if not stack:
                return None, "无解（固定段冲突）"
            i, nxt = stack.pop()
            del buf[-len(_CANDS[nxt - 1]):]
            continue
        ok = False
        for ci in range(nxt, len(_CANDS)):
            c = _CANDS[ci]
            buf.extend(c)
            if tail_ok(buf):
                stack.append((i, ci + 1))
                i += 1
                nxt = 0
                ok = True
                break
            del buf[-len(c):]
        if ok:
            continue
        if not stack:
            return None, "无解（回溯穷尽）"
        i, nxt = stack.pop()
        del buf[-len(_CANDS[nxt - 1]):]


def main():
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    targets = [t for t in TARGETS if not only or any(o in t for o in only)]
    print(f"待还原 {len(targets)} 个文件；模式={'写回' if APPLY else '演练'}\n")
    total_q = 0
    for rel in targets:
        p = ROOT / rel
        if not p.exists():
            print(f"!! 缺失 {rel}")
            continue
        raw = p.read_text(encoding="utf-8")
        segs, n_pua, bad = segments(raw)
        n_q = raw.count("?")
        total_q += n_q
        print(f"=== {rel} ===")
        print(f"  长度 {len(raw)}，PUA {n_pua} 个，ASCII '?' {n_q} 个")
        if bad:
            print(f"  !! 中止：{bad}")
            continue
        fixed, err = solve_simple(segs)
        if fixed is None:
            print(f"  !! 求解失败：{err}")
            continue
        # 校验：还原结果里不应再有 PUA
        left = sum(1 for c in fixed if 0xE000 <= ord(c) <= 0xF8FF)
        head = fixed[:60].replace("\n", "\\n")
        print(f"  还原成功：长度 {len(fixed)}，残留 PUA {left}")
        print(f"  还原后开头: {head}")
        if APPLY:
            p.write_text(fixed, encoding="utf-8")
            print("  [已写回]")
    print(f"\n合计 ASCII '?'（不可恢复的墓碑）= {total_q}")


if __name__ == "__main__":
    main()
