#!/usr/bin/env python
"""揪出孤儿 worker 并弄清它们属于哪个 run，决定能否安全清理。

为什么必须查清归属：若孤儿属于**已死**的 run，父进程没了，它们永远
不可能再产出任何结果，杀掉纯赚 58 GB；若属于**活着**的 run，杀掉会
直接毁掉正在跑的实验。前提不同，动作相反。

用法：
  python .tmp/orphan_audit.py          # 只看
  python .tmp/orphan_audit.py --kill   # 杀掉已死 run 的孤儿
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
from collections import defaultdict

PROC = "/proc"


def cmdline(pid: str) -> str:
    try:
        with open(f"{PROC}/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def ppid_of(pid: str) -> int:
    try:
        with open(f"{PROC}/{pid}/status") as fh:
            for line in fh:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def pss_kb(pid: str) -> int:
    try:
        with open(f"{PROC}/{pid}/smaps_rollup") as fh:
            for line in fh:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def start_time(pid: str) -> str:
    """进程启动时刻，用来推断它属于哪一批。"""
    try:
        return subprocess.run(
            ["ps", "-o", "lstart=", "-p", pid],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kill", action="store_true")
    args = ap.parse_args()

    pids = [d for d in os.listdir(PROC) if d.isdigit()]
    all_p = {}
    for pid in pids:
        c = cmdline(pid)
        if "python" not in c:
            continue
        all_p[pid] = {"cmd": c, "ppid": ppid_of(pid)}

    # 活着的主训练进程（带 run-name 且不是孤儿）
    live_runs = {}
    for pid, p in all_p.items():
        m = re.search(r"run-name\s+(\S+)", p["cmd"])
        if m and p["ppid"] != 1:
            live_runs[m.group(1)] = pid

    print(f"活着的主训练进程: {live_runs or '(无)'}\n")

    orphans = [(pid, p) for pid, p in all_p.items() if p["ppid"] == 1]
    print(f"孤儿 python 进程: {len(orphans)} 个\n")

    # 按启动时刻聚类，判断批次
    buckets = defaultdict(list)
    for pid, p in orphans:
        buckets[start_time(pid)].append((pid, p))

    print(f"{'启动时刻':<28}{'个数':>6}{'PSS合计':>12}  样本命令")
    print("-" * 100)
    for st, items in sorted(buckets.items()):
        tot = sum(pss_kb(pid) for pid, _ in items)
        sample = items[0][1]["cmd"][:45]
        print(f"{st:<28}{len(items):>6}{tot/1048576:>10.2f}G  {sample}")

    tot_all = sum(pss_kb(pid) for pid, _ in orphans)
    print("-" * 100)
    print(f"{'孤儿 PSS 合计':<28}{len(orphans):>6}{tot_all/1048576:>10.2f}G")

    if args.kill:
        print("\n=== 清理 ===")
        killed = 0
        freed = 0
        for pid, p in orphans:
            tag = re.search(r"run-name\s+(\S+)", p["cmd"])
            name = tag.group(1) if tag else None
            # 只杀「不属于任何活着的 run」的孤儿；
            # 若孤儿的 run-name 与活着的 run 同名，说明是同一个 run 的
            # worker 被 reparent 了（父 shell 死了但 trainer 还在），保留。
            if name and name in live_runs:
                print(f"  保留 pid={pid}（属于活着的 {name}）")
                continue
            try:
                os.kill(int(pid), signal.SIGKILL)
                killed += 1
                freed += pss_kb(pid)
            except OSError as e:
                print(f"  杀 pid={pid} 失败: {e}")
        print(f"\n杀掉 {killed} 个孤儿，释放约 {freed/1048576:.2f}G")


if __name__ == "__main__":
    main()
