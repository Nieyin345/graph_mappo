#!/usr/bin/env python
"""受控实验：拿干净中文串走疑似损坏链条，复现出 '锛?' 这类形态，
从而判定 (a) 链条是什么 (b) 尾部的 '?' 是不是真的丢了一个字节。

已知事实：损坏文件里
  '锛' = U+951B  (gbk bytes EF BC)
  '銆' = U+9286  (gbk bytes E3 80)
且两者后面常常各跟一个 ASCII '?' (0x3F)。

推测：原 UTF-8 三字节 E? xx yy 中，前两字节被 gbk 解成一个汉字，
第三字节是 0x80-0xBF 的孤字节，被解码器丢弃/替换成 '?'。
"""
import codecs

SAMPLES = [
    "每个时间，分钟）",
    "（Tx-out ≥ 1、Rx-in ≥ 1",
    "这是测试。",
    "智能体要为每个节点决定一条密钥生成链路，约束是**双端**（",
]

# 各种可能的链条
CHAINS = {
    "utf8 -> gbk(replace)": lambda b: b.decode("gbk", errors="replace"),
    "utf8 -> gbk(ignore)": lambda b: b.decode("gbk", errors="ignore"),
    "utf8 -> cp936(replace)": lambda b: b.decode("cp936", errors="replace"),
    "utf8 -> gb18030(replace)": lambda b: b.decode("gb18030", errors="replace"),
    "utf8 -> gbk(surrogateescape)": lambda b: b.decode("gbk", errors="surrogateescape"),
}

for name, fn in CHAINS.items():
    print(f"\n===== 链条：{name} =====")
    for s in SAMPLES[:3]:
        b = s.encode("utf-8")
        try:
            g = fn(b)
        except Exception as e:
            print(f"  原文 {s!r}\n    -> {type(e).__name__}: {e}")
            continue
        # 再存成 utf-8（模拟落盘）再读回，就是文件里的样子
        stored = g
        has_q = "?" in stored
        print(f"  原文 {s!r}")
        print(f"    gbk字节 {b.hex(' ')}")
        print(f"    -> {stored!r}   含'?'={has_q}")
        # 试还原
        try:
            back = g.encode("gbk").decode("utf-8")
            print(f"    还原: {back!r}  {'✓ 一致' if back == s else '✗ 不一致'}")
        except Exception as e:
            print(f"    还原失败: {type(e).__name__}: {e}")

# 关键判定：'锛' 后面的字节到底是什么
print("\n" + "=" * 70)
print("=== 判定：'，' / '（' / '。' 在三字节 UTF-8 下的第 3 字节 ===")
for ch in "，。、（）！？：；":
    b = ch.encode("utf-8")
    print(f"  {ch}  U+{ord(ch):04X}  utf8={b.hex(' ')}  第1-2字节->gbk: ", end="")
    try:
        print(f"{b[:2].decode('gbk')!r}  第3字节={b[2]:02X}")
    except Exception as e:
        print(f"<{e}>  第3字节={b[2]:02X}")
