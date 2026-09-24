"""进程内文件补丁（带 `finally` 还原）—— 造反证工具。

存在的理由：造反证要**故意改坏**代码，但服务器上的树是 peer 正在跑的，
不能碰；而且改完必须还原，否则下一次实验会踩到上一轮的残留。
`patched()` 是上下文管理器，退出时**无条件**写回原文（含异常路径）。

用法：
    import apply_patch
    with apply_patch.patched(path, anchor, injected) as ok:
        ...   # ok=False 表示锚点不唯一/找不到，**没有**动过文件
"""
from __future__ import annotations

import contextlib
from pathlib import Path


def _count(text: str, anchor: str) -> int:
    return text.count(anchor)


@contextlib.contextmanager
def patched(path, anchor: str, injected: str, *, label: str = ""):
    """把 `anchor` 的**第一次**（且必须是唯一一次）出现替换成 `injected`。

    ★ 锚点不唯一就拒绝打补丁（`ok=False`），因为"替换了哪一个"本身
      不确定 ⟹ 实验结果无法归因。这和 `find()` 的零命中/多命中同一条规矩：
      探针必须能说出它到底改了什么。
    """
    p = Path(path)
    orig = p.read_text(encoding="utf-8")
    n = _count(orig, anchor)
    if n != 1:
        yield False
        return
    new = orig.replace(anchor, injected, 1)
    if new == orig:
        yield False
        return
    try:
        p.write_text(new, encoding="utf-8")
        yield True
    finally:
        # ★ 无条件还原：即使被测代码抛异常也不能把改坏的源码留在盘上
        cur = p.read_text(encoding="utf-8")
        if cur != orig:
            p.write_text(orig, encoding="utf-8")
