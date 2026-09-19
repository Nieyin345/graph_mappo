#!/usr/bin/env python
"""内存看门狗：在 OOM 发生**之前**清掉孤儿 worker，避免连锁杀死训练。

为什么需要独立进程：OOM killer 用的是 SIGKILL，**受害者无法捕获**，所以
trainer 自己不可能在死前清理 worker。唯一可靠的机制是一个旁观者进程，
在内存跌到危险线时抢先回收。

为什么先清孤儿而不是杀训练：孤儿 worker 的父 trainer 已经死了，它们永远
不可能再产出任何结果——回收是纯收益。只有当「清完孤儿仍然不够」时，才
按最晚启动优先（LIFO）杀训练进程，保住最早跑起来的那些。

判据（与 kill_orphans.py 一致）：PPid==1 且 cmdline 含 multiprocess。
父 trainer 活着 → PPid≠1 → 不是孤儿，绝不碰。

用法：
  python .tmp/mem_watchdog.py --threshold 20 --interval 30            # 预演
  python .tmp/mem_watchdog.py --threshold 20 --interval 30 --apply    # 实跑
  setsid nohup python .tmp/mem_watchdog.py --apply > /tmp/watchdog.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import time
from datetime import datetime

PROC = "/proc"


def meminfo() -> dict[str, float]:
    out = {}
    try:
        with open(f"{PROC}/meminfo") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                parts = v.split()
                if parts:
                    out[k] = int(parts[0]) / 1048576.0   # GB
    except OSError:
        pass
    return out


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


def live_trainers() -> dict[int, str]:
    """活着的 trainer 及其 run-name。"""
    out = {}
    for pid in os.listdir(PROC):
        if not pid.isdigit():
            continue
        c = cmdline(pid)
        m = re.search(r"run-name\s+(\S+)", c)
        if m and "train_graph_mappo.py" in c:
            out[int(pid)] = m.group(1)
    return out


def orphan_workers() -> list[tuple[int, int]]:
    """(pid, pss_kb) 的孤儿 worker 列表。"""
    out = []
    for pid in os.listdir(PROC):
        if not pid.isdigit():
            continue
        c = cmdline(pid)
        if "multiprocess" not in c and "spawn_main" not in c:
            continue
        if ppid_of(pid) == 1:
            out.append((int(pid), pss_kb(pid)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=20.0,
                    help="MemAvailable 低于此值(GB)触发回收")
    ap.add_argument("--critical", type=float, default=10.0,
                    help="低于此值(GB)时清完孤儿仍要杀训练")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-hours", type=float, default=20.0)
    args = ap.parse_args()

    t_end = time.time() + args.max_hours * 3600
    print(f"[{datetime.now():%H:%M:%S}] 看门狗启动 阈值={args.threshold}G "
          f"危急={args.critical}G 间隔={args.interval}s "
          f"模式={'实跑' if args.apply else '预演'}", flush=True)

    n_clean = 0
    while time.time() < t_end:
        try:
            mi = meminfo()
            avail = mi.get("MemAvailable", 0.0)
            swap_free = mi.get("SwapFree", 0.0)

            if avail < args.threshold:
                orphs = orphan_workers()
                tot = sum(kb for _, kb in orphs) / 1048576.0
                print(f"[{datetime.now():%H:%M:%S}] ⚠ MemAvailable "
                      f"{avail:.1f}G < {args.threshold}G，"
                      f"孤儿 {len(orphs)} 个 / {tot:.1f}G", flush=True)

                if orphs and args.apply:
                    killed = 0
                    for pid, _ in orphs:
                        try:
                            os.kill(pid, signal.SIGKILL)
                            killed += 1
                        except OSError:
                            pass
                    n_clean += killed
                    time.sleep(2)
                    after = meminfo().get("MemAvailable", 0.0)
                    print(f"[{datetime.now():%H:%M:%S}]   回收 {killed} 个 → "
                          f"MemAvailable {after:.1f}G", flush=True)
                elif orphs:
                    print(f"[{datetime.now():%H:%M:%S}]   (预演，未动手)",
                          flush=True)

                # 清完孤儿仍危急 → 按最晚启动优先杀训练，保住最早的
                if args.apply and meminfo().get("MemAvailable", 0.0) < args.critical:
                    live = live_trainers()
                    if live:
                        pids = sorted(live, reverse=True)   # pid 大 = 后起
                        victim = pids[0]
                        print(f"[{datetime.now():%H:%M:%S}] 🔴 危急，杀最晚的 "
                              f"训练 {live[victim]} (pid={victim})", flush=True)
                        try:
                            os.kill(victim, signal.SIGKILL)
                        except OSError:
                            pass
            elif avail < args.threshold * 1.5:
                print(f"[{datetime.now():%H:%M:%S}] 偏低 MemAvailable "
                      f"{avail:.1f}G swap_free {swap_free:.1f}G", flush=True)

            time.sleep(args.interval)
        except Exception as e:            # 看门狗自己绝不能死
            print(f"[{datetime.now():%H:%M:%S}] 异常 {e!r}", flush=True)
            time.sleep(args.interval)

    print(f"[{datetime.now():%H:%M:%S}] 看门狗退出，累计回收 {n_clean} 个孤儿",
          flush=True)


if __name__ == "__main__":
    main()
