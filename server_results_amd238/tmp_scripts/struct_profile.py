#!/usr/bin/env python
"""代码结构画像：函数长度、参数个数、重复定义——为"结构优化"提供依据。

只读，不改任何东西。目的是把"代码很乱"变成可核对的数字，
避免凭感觉重构（本项目对"凭感觉"已经踩过多次）。
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

TARGETS = [
    "qkd_rl/rl/algos/mappo_trainer.py",
    "qkd_rl/rl/models/graph_mappo.py",
    "qkd_rl/env/env.py",
    "qkd_rl/env/action_resolver.py",
    "qkd_rl/rl/algos/policy.py",
]

print("=" * 78)
print("函数长度画像（>60 行的列出；这类函数是重构的主要对象）")
print("=" * 78)

total_funcs = 0
long_funcs = []
for rel in TARGETS:
    p = ROOT / rel
    if not p.exists():
        print(f"  (缺 {rel})")
        continue
    src = p.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"  !! {rel} 解析失败: {e}")
        continue
    lines = src.splitlines()
    print(f"\n--- {rel}  ({len(lines)} 行)")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n = (node.end_lineno or node.lineno) - node.lineno + 1
            total_funcs += 1
            nargs = len(node.args.args) + len(node.args.kwonlyargs)
            if n > 60:
                long_funcs.append((n, rel, node.name, node.lineno, nargs))
    # 只打印本文件的统计
    local = [f for f in long_funcs if f[1] == rel]
    if local:
        for n, _, name, lineno, nargs in sorted(local, reverse=True):
            print(f"    {n:>4} 行  {name:<45} (:L{lineno}, {nargs} 参数)")
    else:
        print("    （没有 >60 行的函数）")

print()
print("=" * 78)
print(f"5 个文件共 {total_funcs} 个函数，其中 >60 行的 {len(long_funcs)} 个")
print("=" * 78)

# 类的大小
print()
print("=" * 78)
print("类画像（方法与行数）")
print("=" * 78)
for rel in TARGETS:
    p = ROOT / rel
    if not p.exists():
        continue
    tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            n = (node.end_lineno or node.lineno) - node.lineno + 1
            meths = [x for x in node.body
                     if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if n > 150:
                print(f"  {n:>4} 行  (L{node.lineno:<5}) {rel}::{node.name}"
                      f"   {len(meths)} 个方法")

# 重复的函数名（跨文件同名，可能是复制粘贴）
print()
print("=" * 78)
print("跨文件同名的函数/方法（复制粘贴的线索）")
print("=" * 78)
names = defaultdict(list)
for rel in TARGETS:
    p = ROOT / rel
    if not p.exists():
        continue
    tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names[node.name].append(f"{rel}:L{node.lineno}")
dups = {k: v for k, v in names.items() if len(v) > 1}
if dups:
    for k, v in sorted(dups.items()):
        print(f"  {k}:")
        for loc in v:
            print(f"      {loc}")
else:
    print("  （无）")
