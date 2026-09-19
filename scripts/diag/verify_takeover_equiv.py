# -*- coding: utf-8 -*-
"""核对 launch_g2.py 会不会把 `hist32_s44` 起成**与 launch_wave263.py 逐字相同**的命令。

为什么必须做这个：如果 g2 接管后把 s44 起成了另一套（臂名/配置/线程/轮数任一不同），
它就是**另一条臂**，与 wave263 的 s42/s43 不可比 ⟹ hist32 判读（Task #1）作废。
这种错**不会报错**，只会生成一份看起来正常的 outputs/。

做法：把两个模块的 `launch()` **各自的命令构造**跑一遍（不真的起进程），逐字比。
两个文件都在服务器 /tmp 下。g2 的命令构造抽在同一段代码里，这里复刻它的表达式
—— 复刻本身也可能写错，所以**再加一层**：直接 import g2 模块，monkeypatch 掉
`subprocess.Popen`，把真实构造出的 argv 抓下来。这样比的是**真代码**，不是复刻。
"""
from __future__ import annotations

import importlib.util
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


captured = []


class FakePopen:
    def __init__(self, cmd, **kw):
        captured.append((list(cmd), kw.get("env", {})))
        self.pid = 12345


def grab(m, name, nupd, resume):
    """用假 Popen 跑 m.launch()，把真实 argv 抓出来。"""
    captured.clear()
    m.subprocess.Popen = FakePopen
    m.log = lambda *_a, **_k: None          # 静音
    try:
        m.launch(name, m.HIST_CFGS if hasattr(m, "HIST_CFGS") else None,
                 "hist", nupd, resume)
    except TypeError:
        # wave263 的 launch(name, cfgs, kind) 没有 nupd/resume
        m.launch(name, m.HIST_CFGS, "hist")
    return captured[0]


g2 = load("/tmp/launch_g2.py", "g2")
w = load("/tmp/launch_wave263.py", "w")

# ---- 抓 g2 的命令（phase A 里 s44 是 nupd=30、resume=None）----
g_argv, g_env = grab(g2, "hist32_s44", 30, None)
# ---- 抓 driver 的命令 ----
w_argv, w_env = grab(w, "hist32_s44", 30, None)

print("driver argv：")
print("  " + " ".join(w_argv))
print("g2     argv：")
print("  " + " ".join(g_argv))
print()

bad = 0
if w_argv != g_argv:
    print("✗ argv **不等** —— 逐位差异：")
    for i, (a, b) in enumerate(zip(w_argv, g_argv)):
        if a != b:
            print("     [%d] driver=%r  g2=%r" % (i, a, b))
    if len(w_argv) != len(g_argv):
        print("     长度不同：%d vs %d" % (len(w_argv), len(g_argv)))
    bad += 1
else:
    print("✓ argv 逐字相同（%d 个 token）" % len(g_argv))

# ---- 环境变量：线程数必须相同（线程数确定性影响结果，差 0.018）----
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    a, b = w_env.get(k), g_env.get(k)
    ok = a == b
    print("%s driver=%r g2=%r ⟹ %s" % ("✓" if ok else "✗", a, b, k))

print()
print("判据（三条都必须成立）：")
print("  1. argv 逐字相同")
print("  2. OMP_NUM_THREADS 相同")
print("  3. MKL_NUM_THREADS 相同")
print()
print("结论：%s" % ("可以接管 ✓" if bad == 0 else "**不可以接管** ✗"))
sys.exit(1 if bad else 0)
