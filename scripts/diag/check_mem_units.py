# -*- coding: utf-8 -*-
"""核对 `launch_g2.py` 与 `mem_pss.py` 的内存**单位**是否一致。

为什么值得单写一个检查：两个文件的 `MemTotal` 读数对不上（263.4 vs 251.2），
说明一个用十进制 GB（÷1e9 / ÷1e6），一个用 GiB（÷2^30 / ÷2^20）。
而 `PSS_BASE=25.0` / `PSS_HIST=63.0` 这两个**模型常数是从 mem_pss 量出来的**
⟹ 如果 launch_g2 拿 GiB 的数当 GB 用，稳态就被**系统性低估 7.4%**，
门会**偏松** —— 正是记忆里
`silent-lenient-fallback-in-thresholds` / `thresholds-and-transcribed-numbers`
那一类静默错误。

判据用**同一份原始字节**两种算法各算一遍，看差多少；再拿一个在跑进程的 PSS
对两个工具做交叉验证。不猜、不外推。

## ★ 退出码必须分辨「判不出」与「通过」（B-13，2026-09-21 修）

旧实现里 **三条 `return 0` 全都是「跳过」**（没训练进程 / 读不到 mem_pss /
文件不存在）⟹ **rc 结构性恒为 0**（只有读 smaps 失败才 1）⟹ 无论查没查成，
调用方看到的都是"通过"。同族：`gate-must-print-its-inputs`（恒真的门不报错）、
`failed-launch-must-be-loud`（失败被当成"还在跑"）。

约定（与 `watch_liveness.py` 的 0/1/2/3 一致）：

    0 = 查了，且通过
    1 = 查了，且有不合
    2 = **判不出**（前提不具备：没有在跑的训练进程、或找不到被检文件）
"""

from __future__ import annotations

import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

KB_IN_MIB = 1024.0
MIB_IN_GIB = 1024.0
# ★★ 两个**不同**的常数，原脚本把它们混为一个（2026-09-21 修）。
#
#   `KB_PER_GB        = 1e6`        ← kB 是 KiB，除以它得到十进制 GB
#   `MIB_PER_GB       = 2**20/1e6`  ← **把已经是 MiB 的值**换算成十进制 GB
#   `KB_PER_GIB       = 2**20`      ← kB（=KiB）除以它得到 GiB
#
#   本脚本第 42 行打印的「两者比值」= (kb/1e6)/(kb/2^20) = 2^20/1e6 = **1.048576**。
#   而旧标签写的「理论 2^30/1e9 = 1.073742」是 `MIB_PER_GB` —— **另一个量**。
#   两个数不一样，所以那行**从写下那天起标签就是错的**，却从没人发现：
#   **它只是打印，从来没有任何断言读过它**。
#   同族：`thresholds-and-transcribed-numbers`（手抄常数）——
#   本条是它的变体：**手抄的常数写对了数字，但配错了场景**。
KB_PER_GB = 1e6
MIB_PER_GB = 2 ** 30 / 1e9          # = 1.073741824（用于"已是 MiB 的值"）
KB_PER_GIB = 2 ** 20
# 第 42 行那个比值的理论值 —— **是下面这个**，不是 MIB_PER_GB
RATIO_GB_OVER_GIB = KB_PER_GIB / KB_PER_GB      # = 1.048576

RC_OK, RC_FAIL, RC_UNDECIDED = 0, 1, 2
_fails: list[str] = []


def read_memtotal_kb() -> int:
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemTotal:"):
            return int(ln.split()[1])
    return 0


def main() -> int:
    kb = read_memtotal_kb()
    if kb <= 0:
        # ★ 旧实现这里也 `return 0` —— 连 MemTotal 都读不到却报"通过"。
        print("✗ 读不到 MemTotal ⟹ **判不出**（不是通过）")
        return RC_UNDECIDED
    as_gib = kb / KB_IN_MIB / MIB_IN_GIB          # mem_pss.py 的算法
    as_gb = kb / 1e6                              # launch_g2.py 的算法
    print("MemTotal 原始值 = %d kB" % kb)
    print("  mem_pss.py  算法 ÷1024÷1024 = %.1f GiB" % as_gib)
    print("  launch_g2.py 算法 ÷1e6      = %.1f GB" % as_gb)
    print("  两者比值 = %.6f（理论 %.6f = 2^20/1e6）"
          % (as_gb / as_gib, RATIO_GB_OVER_GIB))
    # ★ 这条以前只是**打印**，从不判定 —— 于是它的标签错了好几年没人发现。
    #   现在它是一条断言：比值偏离理论值 ⟹ 两个实现真的不同源，
    #   这正是本脚本存在的理由（记忆 `meminfo-kb-is-kib-not-gb`）。
    ratio_err = abs(as_gb / as_gib - RATIO_GB_OVER_GIB)
    if ratio_err > 1e-9:
        _fails.append("比值 %.9f ≠ 理论 %.9f（2^20/1e6）"
                      % (as_gb / as_gib, RATIO_GB_OVER_GIB))
    # 正对照：把**另一个**常数也印出来，免得下次再有人把两个搞混
    print("     ⚠ 别跟 %.6f（2^30/1e9）搞混 —— 那个是「已是 MiB 的值」换算成 GB"
          % MIB_PER_GB)
    print("       的系数，**不是**本行这个比值的理论值。旧标签配错了场景。")
    print()

    # ---- 交叉验证：取一个在跑的 run，两个工具各量一次 PSS ----
    # mem_pss.py 的 pss_of: kB/1024/1024 (GiB)；launch_g2 的 pss: bytes/1e9 (GB)
    target = None
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            cl = open("/proc/%s/cmdline" % p, "rb").read().decode(
                "utf-8", "replace").replace("\0", " ")
        except OSError:
            continue
        m = re.search(r"--run-name\s+(\S+)", cl)
        if m:
            target = (int(p), m.group(1))
            break
    if target is None:
        print("（此刻没有训练进程，跳过 PSS 交叉验证）")
        return _verdict(undecided="没有在跑的训练进程，PSS 交叉验证这一段没查")

    pid, name = target
    tot_kb = 0
    try:
        for ln in open("/proc/%d/smaps_rollup" % pid):
            if ln.startswith("Pss:"):
                tot_kb += int(ln.split()[1])
    except OSError as e:
        # ★ 旧实现这里 `return 1`（报"有不合"），但**读不到 ≠ 有不合** ——
        #   那是"判不出"。1 是留给真发现的。
        print("读不到 %d 的 smaps_rollup: %s ⟹ **判不出**" % (pid, e))
        return RC_UNDECIDED
    bytes_ = tot_kb * 1024
    print("在跑进程 %d (%s) 的 PSS：原始 %d kB" % (pid, name, tot_kb))
    print("  mem_pss.py  算法 = %.2f GiB" % (tot_kb / 1024 / 1024))
    print("  launch_g2.py 算法 = %.2f GB" % (bytes_ / 1e9))
    print()

    print("== 对模型常数的影响 ==")
    for label, v_gib in (("PSS_BASE", 25.0), ("PSS_HIST", 63.0)):
        # ★ 这里用的才是 MIB_PER_GB（值**已经是 GiB**，要换算成十进制 GB），
        #   与第 42 行那个比值的理论值 RATIO_GB_OVER_GIB 是**两个不同的量**。
        v_gb = v_gib * MIB_PER_GB
        print("  %s = %.1f（按 mem_pss 量的 GiB）⟹ 折算成 GB 是 %.1f（+%.1f）"
              % (label, v_gib, v_gb, v_gb - v_gib))
    print()
    print("判据：若 launch_g2 内部混用两种单位，则**稳态被低估 7.4%**，门偏松。")
    print("     修法：让 launch_g2 **全程用 GiB**（与 mem_pss 及模型常数同单位）。")
    print()

    # ---- 反向检查：mem_pss.py 自己的**标签**写错了 ----
    # 它内部用 ÷1024÷1024（GiB），却把标签打成 "GB" ⟹ 读者会把它的读数
    # 当十进制 GB 用，于是 ≈ +5% 的静默偏松。这正是 2026-09-20 那次单位错
    # 的**传播路径**：不是谁抄错了数，是**标签**在骗人。
    mp = "/opt/qkd/graph_mappo/scripts/diag/mem_pss.py"
    try:
        s = open(mp, encoding="utf-8").read()
    except OSError:
        print("（读不到 %s，跳过标签检查）" % mp)
        return _verdict(undecided="读不到 mem_pss.py，标签自检这一段没查")
    import re as _re
    gb_labels = len(_re.findall(r"GB", s))
    gib_labels = len(_re.findall(r"GiB", s))
    divides_gib = ("/ KB / KB" in s) or ("/1024/1024" in s)
    print("== mem_pss.py 的标签自检 ==")
    print("  内部算法是 GiB（÷1024÷1024）：%s" % divides_gib)
    print("  文中出现 'GB' %d 次、'GiB' %d 次" % (gb_labels, gib_labels))
    if divides_gib and gb_labels > gib_labels:
        print("  ⚠ **标签与算法不符** —— 它量的是 GiB 却标成 GB。读者照标签用会偏松 ≈5%。")
        print("     这不只是笔误：单位错就是**这样传播**的（读的人信任标签）。")
        _fails.append("mem_pss.py 标签与算法不符（GB %d > GiB %d）"
                      % (gb_labels, gib_labels))
    else:
        print("  ✓ 标签与算法一致")
    return _verdict()


def _verdict(undecided: str | None = None) -> int:
    """把「发现了不合」与「有一段没查成」分开报，并给对应的 rc。

    ★ 顺序有意：**先报不合**（那是真读数），再把"没查成的段"降级为不确定。
      反之（有不合却因为某段没查成而报 2）会把真发现静默掉。
    """
    if _fails:
        print()
        print("✗ **有不合**：")
        for f in _fails:
            print("   · %s" % f)
        return RC_FAIL
    if undecided:
        print()
        print("⚠ **判不出**：%s" % undecided)
        print("   —— 这既不是通过也不是失败。rc=2 让调用方能分辨。")
        return RC_UNDECIDED
    print()
    print("✓ 两处实现同单位、标签与算法一致。rc=0")
    return RC_OK


if __name__ == "__main__":
    sys.exit(main())
