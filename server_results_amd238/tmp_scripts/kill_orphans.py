#!/usr/bin/env python
"""清理孤儿 rollout worker。带安全闸：绝不动还活着的 run 的 worker。

为什么必须带闸：worker 的 cmdline 里没有 run-name（是 "from multiprocess
..."），无法从自身判断归属。唯一可靠的判据是 PPid：
  - 父 trainer 活着 → PPid = trainer_pid ≠ 1 → 不是孤儿，绝不动
  - 父 trainer 被 OOM 杀 → PPid = 1 → 孤儿，永远不可能再产出结果
所以「PPid==1 且 cmdline 含 multiprocess」就是安全且完备的判据。

用法：
  python .tmp/kill_orphans.py           # 只列出将要杀谁
  python .tmp/kill_orphans.py --apply
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import time

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


def live_trainer_pids() -> set[int]:
    """所有活着的 trainer / shell 的 pid，用作保护名单。"""
    keep = set()
    for pid in os.listdir(PROC):
        if not pid.isdigit():
            continue
        c = cmdline(pid)
        # 活着的 trainer（带 run-name 且父进程还在）或包装 shell
        if "train_graph_mappo.py" in c or "run_one_seed.sh" in c:
            keep.add(int(pid))
            keep.add(ppid_of(pid))
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    guard = live_trainer_pids()

    # 保护名单的扩展：活着的 trainer 的所有子孙都不算孤儿（PPid 不会是 1），
    # 所以不需要额外处理；这里只打印出来确认。
    print(f"保护名单（活着的 trainer/shell）：{sorted(guard) or '(无)'}\n")

    targets = []
    for pid in os.listdir(PROC):
        if not pid.isdigit():
            continue
        p = int(pid)
        c = cmdline(pid)
        if "multiprocess" not in c and "spawn_main" not in c:
            continue
        if ppid_of(pid) != 1:          # 父还在 → 不是孤儿
            continue
        if p in guard:
            continue
        targets.append((p, pss_kb(pid), c[:60]))

    # 安全断言：目标里不能出现任何保护名单成员
    bad = [t for t in targets if t[0] in guard]
    if bad:
        print(f"!! 安全闸触发，目标与保护名单冲突：{bad}")
        return

    if not targets:
        print("没有孤儿 worker，无需清理。")
        return

    tot = sum(t[1] for t in targets)
    print(f"待清理孤儿 worker：{len(targets)} 个，PSS 合计 {tot/1048576:.2f}G\n")
    for p, kb, c in sorted(targets, key=lambda t: -t[1])[:10]:
        print(f"  pid={p:<8} {kb/1048576:6.2f}G  {c}")
    if len(targets) > 10:
        print(f"  ...（其余 {len(targets)-10} 个）")

    if not args.apply:
        print(f"\n预演模式。加 --apply 执行清理。")
        return

    print("\n清理中...")
    killed, freed = 0, 0
    for p, kb, _ in targets:
        try:
            os.kill(p, signal.SIGKILL)
            killed += 1
            freed += kb
        except OSError:
            pass
    time.sleep(2)
    still = sum(1 for p, _, _ in targets if os.path.exists(f"{PROC}/{p}"))
    print(f"杀掉 {killed} 个，释放约 {freed/1048576:.2f}G，残留 {still} 个")

    with open(f"{PROC}/meminfo") as fh:
        mi = {l.split(":")[0]: int(l.split(":")[1].split()[0]) for l in fh
              if l.split(":")[1].split()}
    print(f"MemAvailable {mi['MemAvailable']/1048576:.1f}G  "
          f"SwapFree {mi['SwapFree']/1048576:.2f}G")


if __name__ == "__main__":
    main()
