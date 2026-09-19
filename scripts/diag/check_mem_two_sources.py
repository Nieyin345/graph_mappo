# -*- coding: utf-8 -*-
"""对表：`launch_g2.py` 的 `pss()` 与 `mem_pss.py` 的 `pss_of()` 是不是同一个量。

### 为什么要对表

记忆 `meminfo-kb-is-kib-not-gb` 的教训：**两个工具量同一物理量时必须对表**。
2026-09-21 那次是 `launch_g2` 用十进制 GB、`mem_pss` 用 GiB ⟹ 同一台机器读出
两个 `MemTotal`（263.4 vs 251.2）⟹ 内存门**静默偏松**、余量虚报近一倍。

那个 bug 修了，但**没有任何常设检查**保证它不再回来——两个文件各写各的
`/proc` 解析，改一个忘一个就会重新漂移（记忆 `duplicate-implementation-drifts`）。
本脚本就是那道检查：**对同一批在跑进程，两个实现各量一次，逐进程打印差值**。

### 同时校准模型常数

`launch_g2` 的门用**模型常数** `PSS_BASE=25.0` / `PSS_HIST=63.0` 算 Σ稳态，
而日志里打印的"在跑值"是**实测**。本脚本把两者并排，看模型偏哪一侧：

  · 模型 > 实测 ⟹ 门**偏保守**（安全，代价是少放臂）
  · 模型 < 实测 ⟹ 门**偏松**（危险 ⟹ 必须上调常数）

⚠ 2026-09-21 更正：原写法在这里塞了一句「父进程一个 8.26 GB 单段的嫌疑是
`log_probs`/`entropies` 的每节点一张量」。**那句话是错的，已删**：
90 个 `.detach()` 对象**共享同一份 storage**（只多出 ~180 个对象头/步），
量级是几百 MB，**解释不了 8.26 GB**。详见记忆
`run-memory-23gb-and-growing`。**别在脚本的说明文字里写没核过的因果**——
那会变成下一个"照文档猜"的坑（`whitelist-from-docs-not-from-keys`）。

⚠ 注意「在跑的实测」**不是稳态**：hist 在 u10 时 ~55 < 63 是正常的。
所以 `差(实-模)` 为负**不直接**说明常数该降——要校准常数得看**跑到末轮**的臂。

只读，不改任何东西。可在任意时刻运行（无训练进程时打印提示后退出）。
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

G2 = next((p for p in ("/tmp/launch_g2.py",
                       "/opt/qkd/graph_mappo/scripts/diag/launch_g2.py")
           if os.path.exists(p)), None)
MEM_PSS = next((p for p in ("/tmp/mem_pss.py",
                            "/opt/qkd/graph_mappo/scripts/diag/mem_pss.py")
                if os.path.exists(p)), None)
if G2 is None or MEM_PSS is None:
    print("✗ 找不到 launch_g2.py（%s）或 mem_pss.py（%s）" % (G2, MEM_PSS))
    sys.exit(1)
print("用 %s 与 %s" % (G2, MEM_PSS))

spec = importlib.util.spec_from_file_location("g2", G2)
g2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g2)


def load_mem_pss():
    """mem_pss.py 可能是脚本形态（模块级就跑）。只取它的 pss_of 函数源码。"""
    src = open(MEM_PSS, encoding="utf-8").read()
    # 抽出 pss_of 定义（含到下一个顶层 def）
    m = re.search(r"^def pss_of\(.*?\n(?=^def |\Z)", src, re.S | re.M)
    if not m:
        return None
    ns = {"os": os, "KB": 1024.0}
    # pss_of 内部用了 pathlib.Path 与模块级 KB —— 抽函数时要把**它依赖的名字**
    # 一起给上，否则 `NameError: name 'Path'/'KB' is not defined`
    # （我连栽了两次：先 Path 后 KB。抽函数不连带依赖 = 拆一半的东西。）
    from pathlib import Path
    ns["Path"] = Path
    exec(m.group(0), ns)          # noqa: S102 —— 只执行这一个函数定义
    return ns["pss_of"]


pss_of = load_mem_pss()
if pss_of is None:
    print("✗ 抽不出 mem_pss.pss_of，退出")
    sys.exit(1)

print("=" * 74)
print("校准：g2 的内存模型 vs 实测 PSS")
print("=" * 74)
print("模型的 steady_of: base=%.1f  hist=%.1f  （**常数，不是实测**）"
      % (g2.PSS_BASE, g2.PSS_HIST))
print()

runs = g2.live_runs()
if not runs:
    print("（此刻没有训练进程在跑，无法校准）")
    sys.exit(0)

# g2.live_runs() 返回 (name, kind, 模型稳态)。要拿 PID 才能实测。
# 训练进程的 cmdline 带 --run-name；worker 不带，用 PPid 归组。
trainers = {}
for p in g2._pids():
    cl = g2.cmdline(p)
    m = re.search(r"--run-name\s+(\S+)", cl)
    if m:
        trainers[m.group(1)] = p

print("%-22s %-6s %8s %8s %8s %8s" %
      ("run", "kind", "模型", "实测crew", "mem_pss", "差(实-模)"))
print("-" * 74)


def crew_of(pid):
    """父 + 全部 spawn worker 的 pid 列表（worker 的 cmdline 不带 run-name，
    所以只能按 PPid 归组 —— 记忆 `oom-orphan-workers`）。"""
    return [pid] + [q for q in g2._pids() if g2.ppid(q) == pid]


rows = []
for name, kind, model in sorted(runs):
    pid = trainers.get(name)
    if pid is None:
        print("%-22s %-6s %8.1f %8s %8s %8s" %
              (name, kind, model, "(无PID)", "--", "--"))
        continue
    crew = crew_of(pid)
    # ★ 两个实现对**同一批 pid** 各量一次 —— 这才是同一物理量的对表。
    #   第一版我拿"全 crew 求和"去比"只量父进程"，两个量不同，比不出单位错。
    meas = sum(g2.pss(q) for q in crew)                 # launch_g2 的实现
    ref = sum(pss_of(q) for q in crew)                  # mem_pss 的实现
    rows.append((name, kind, model, meas, crew))
    flag = "" if abs(meas - ref) < 0.05 else "  ✗ **两个实现不一致**"
    print("%-22s %-6s %8.1f %8.1f %8.1f %+8.1f   (%d 进程)%s" %
          (name, kind, model, meas, ref, meas - model, len(crew), flag))

if not rows:
    print("\n（没有能对上的进程）")
    sys.exit(0)

print()
print("=" * 74)
print("汇总")
print("=" * 74)
sum_model = sum(r[2] for r in rows)
sum_meas = sum(r[3] for r in rows)
avail = g2.avail_gib()
print("  Σ模型稳态 = %.1f GiB   Σ实测 PSS = %.1f GiB   **差 %+.1f**"
      % (sum_model, sum_meas, sum_meas - sum_model))
print("  MemAvailable = %.1f GiB（视角 A 的输入，这个是**实测**）" % avail)
print("  其它占用 = %.1f GiB" % (g2.memtotal_gib() - g2.memfree_gib() - sum_meas))
print()

for kind in ("base", "hist"):
    vs = [r[3] for r in rows if r[1] == kind]
    if vs:
        print("  %-5s 实测均值 %.1f GiB（n=%d），模型常数 %.1f  ⟹ 差 %+.1f"
              % (kind, sum(vs) / len(vs), len(vs), g2.steady_of(kind),
                 sum(vs) / len(vs) - g2.steady_of(kind)))

print()
print("读法：")
print("  · `实测crew` 与 `mem_pss` 两列**必须一致** —— 不一致就是单位/口径漂移，")
print("    正是 `meminfo-kb-is-kib-not-gb` 那个 bug 复发的信号。")
print("  · `差(实-模)` 是**当前实测** − **模型常数**。")
print("    负值不代表常数该降：在跑的臂**不是稳态**（hist 在 u10 时 ~55 < 63 正常）。")
print("    要校准常数，看**跑到末轮**的臂。")
print("  · 门的两个视角用的是 MemAvailable（实测）+ 模型常数，**混用**是刻意的：")
print("    稳态只能靠模型（当前值不代表未来），可用量必须靠实测。")
