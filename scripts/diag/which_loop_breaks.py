#!/usr/bin/env python
"""决定性问题：KL 早停的 `break` 到底跳出**哪个**循环？

背景：`configs/train_safe_ep2.yaml`（我写的）声称
    「epochs=1 时早停会中止整轮更新，epochs=2 时**可以只停掉第二个 epoch**」
并据此说"本臂的 stop_for_kl 行为与基线不可直接比"。

但肉眼看缩进，`mappo_trainer.py` 的两处 break 是：
    729  break          # 缩进 20，跳出 batch 循环
    731  break          # 缩进 16，跳出的那个循环是……？
如果 731 跳的是 **epoch 循环**，那 epochs=2 和 epochs=1 一样会中止**整轮更新**，
上面那句"可以只停掉第二个 epoch"就是**错的**，而它会导致我读 ep2 结果时
错误地认为两侧指标不可比。

**缩进肉眼数容易错，所以用 ast 直接问解释器**：找出每个 break 所属的
最近一层循环，打印它的源码行。这比数空格可靠，也比读文档可靠。

用法（服务器上）：python /tmp/which_loop_breaks.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path("/opt/qkd/graph_mappo/qkd_rl/rl/algos/mappo_trainer.py")

tree = ast.parse(SRC.read_text(encoding="utf-8"))
lines = SRC.read_text(encoding="utf-8").splitlines()

LOOPS = (ast.For, ast.While, ast.AsyncFor)


def func_name(node) -> str:
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(c is node for c in ast.walk(n)):
                return n.name
    return "<module>"


def enclosing_loop(node, parents):
    """从内往外找最近的一层循环。"""
    for p in reversed(parents):
        if isinstance(p, LOOPS):
            return p
    return None


print(f"文件: {SRC}")
print(f"行数: {len(lines)}")
print()

found = 0
for parent_chain in ast.walk(tree):
    pass

# 手动遍历，维护祖先栈 —— 才能回答"这个 break 属于哪个循环"
def visit(node, stack):
    global found
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Break):
            loop = enclosing_loop(child, stack)
            found += 1
            ln = child.lineno
            print(f"break @ line {ln}")
            print(f"  源码     : {lines[ln-1].strip()!r}")
            if loop is None:
                print("  ⚠ 找不到所属循环（异常结构）")
            else:
                ll = loop.lineno
                print(f"  所属循环 : {type(loop).__name__} @ line {ll}")
                print(f"  循环源码 : {lines[ll-1].strip()!r}")
            print()
        visit(child, stack + [child])


visit(tree, [])
print(f"共 {found} 处 break")

print()
print("=== 判读 ===")
# 另外验证：epoch 循环与 batch 循环的行号
epoch_line = batch_line = None
for n in ast.walk(tree):
    if isinstance(n, ast.For) and n.lineno in (663, 673):
        if n.lineno == 663:
            epoch_line = n.lineno
        else:
            batch_line = n.lineno
print(f"  epoch 循环 @ {epoch_line}: {lines[epoch_line-1].strip()!r}")
print(f"  batch 循环 @ {batch_line}: {lines[batch_line-1].strip()!r}")
print()
print("  若 line 731 的 break 隶属于 **epoch 循环**，则 epochs=2 时 KL 早停同样")
print("  中止**整轮更新** → 配置文件里那句'可以只停掉第二个 epoch'是**错的**。")
