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
"""
from __future__ import annotations

import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

KB_IN_MIB = 1024.0
MIB_IN_GIB = 1024.0
GiB_PER_GB = 2 ** 30 / 1e9          # = 1.073741824


def read_memtotal_kb() -> int:
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemTotal:"):
            return int(ln.split()[1])
    return 0


def main() -> int:
    kb = read_memtotal_kb()
    as_gib = kb / KB_IN_MIB / MIB_IN_GIB          # mem_pss.py 的算法
    as_gb = kb / 1e6                              # launch_g2.py 的算法
    print("MemTotal 原始值 = %d kB" % kb)
    print("  mem_pss.py  算法 ÷1024÷1024 = %.1f GiB" % as_gib)
    print("  launch_g2.py 算法 ÷1e6      = %.1f GB" % as_gb)
    print("  两者比值 = %.4f（理论 2^30/1e9 = %.4f）" % (as_gb / as_gib, GiB_PER_GB))
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
        return 0

    pid, name = target
    tot_kb = 0
    try:
        for ln in open("/proc/%d/smaps_rollup" % pid):
            if ln.startswith("Pss:"):
                tot_kb += int(ln.split()[1])
    except OSError as e:
        print("读不到 %d 的 smaps_rollup: %s" % (pid, e))
        return 1
    bytes_ = tot_kb * 1024
    print("在跑进程 %d (%s) 的 PSS：原始 %d kB" % (pid, name, tot_kb))
    print("  mem_pss.py  算法 = %.2f GiB" % (tot_kb / 1024 / 1024))
    print("  launch_g2.py 算法 = %.2f GB" % (bytes_ / 1e9))
    print()

    print("== 对模型常数的影响 ==")
    for label, v_gib in (("PSS_BASE", 25.0), ("PSS_HIST", 63.0)):
        v_gb = v_gib * GiB_PER_GB
        print("  %s = %.1f（按 mem_pss 量的 GiB）⟹ 折算成 GB 是 %.1f（+%.1f）"
              % (label, v_gib, v_gb, v_gb - v_gib))
    print()
    print("判据：若 launch_g2 内部混用两种单位，则**稳态被低估 7.4%**，门偏松。")
    print("     修法：让 launch_g2 **全程用 GiB**（与 mem_pss 及模型常数同单位）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
