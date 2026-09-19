#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单 run 的内存构成：父进程 vs 8 个 worker，以及父进程最大的匿名映射。

为什么要量而不是推：并发上限 =(250−17)/PSS，所以**每一个 GB 都直接换成并发数**。
"降内存换吞吐"是这台机器上唯一还剩的路（CPU 只用 28%），
而要走这条路必须先知道**内存花在哪个结构上**——猜错方向就白跑一轮 A/B。
只读，不碰任何 run。
"""
import glob
import os
import re
import subprocess


def cmdline(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            return f.read().decode("utf-8", "replace").replace("\0", " ")
    except OSError:
        return ""


def pss(pid):
    try:
        for ln in open("/proc/%d/smaps_rollup" % pid):
            if ln.startswith("Pss:"):
                return int(ln.split()[1]) / 1e6
    except OSError:
        return 0.0


def rss(pid):
    try:
        for ln in open("/proc/%d/status" % pid):
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) / 1e6
    except OSError:
        return 0.0


# 找 trainer 父进程
parents = []
for d in glob.glob("/proc/[0-9]*"):
    pid = int(d.rsplit("/", 1)[1])
    c = cmdline(pid)
    if "train_graph_mappo.py" in c and "-u" in c:      # 只有父进程带 -u
        m = re.search(r"--run-name\s+(\S+)", c)
        if m:
            parents.append((m.group(1), pid))

if not parents:
    print("没找到带 -u 的 trainer 父进程")
    raise SystemExit(1)

for run, pid in parents[:2]:
    print("=" * 74)
    print("run = %s   父 pid = %d" % (run, pid))
    print("=" * 74)
    pp = pss(pid)
    print("  父进程 PSS = %6.2f GB   RSS = %6.2f GB" % (pp, rss(pid)))

    kids = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True,
                          text=True).stdout.split()
    tot = 0.0
    per = []
    for k in kids:
        k = int(k)
        if "multiprocess" in cmdline(k) or cmdline(k).count("python") > 0:
            v = pss(k)
            tot += v
            per.append(v)
    print("  worker 数 = %d   合计 PSS = %6.2f GB   单 worker 均值 = %.2f GB"
          % (len(per), tot, (sum(per) / len(per)) if per else 0))
    print("  ⟹ 单 run 合计 ≈ %6.2f GB" % (pp + tot))
    print()

    # 父进程最大的匿名映射（buffer / torch arena / 堆）
    maps = []
    try:
        with open("/proc/%d/smaps" % pid) as f:
            cur = None
            for ln in f:
                if "-" in ln.split()[0] and ln.split()[0].count("-") == 1:
                    cur = ln.strip()
                elif ln.startswith("Pss:") and cur:
                    maps.append((int(ln.split()[1]) / 1e6, cur))
                    cur = None
    except OSError as e:
        print("  读 smaps 失败：%s" % e)
    maps.sort(reverse=True)
    print("  父进程最大的 8 个内存段（PSS GB / 虚拟地址范围 / 权限 / 名字）:")
    for v, seg in maps[:8]:
        parts = seg.split()
        addr = parts[0]
        perms = parts[1] if len(parts) > 1 else ""
        name = parts[-1] if len(parts) > 5 else "[anon]"
        print("    %7.2f GB  %-28s %-5s %s" % (v, addr, perms, name))
    print()
