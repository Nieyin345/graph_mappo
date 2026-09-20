"""按进程组停掉指定的臂（不留孤儿 worker）。

## 为什么按进程组

`multiprocessing.spawn` 的 8 个 worker 的 PPID 指向 trainer 父进程。
`kill -TERM -<pgid>` 让整组一起收到 ⟹ 不留孤儿
（记忆 `oom-orphan-workers`：孤儿是 OOM 的**结果**，每个 1.1 GiB 空转）。

## 为什么要先验证

★ 项目红线：`pkill -f` 在 `ssh host '...'` 里会**杀掉自己**（远程 bash -c 的
命令行自身含该模式）。所以本脚本**不 shell out 到 pkill**：
先扫 /proc 定位 PID，验证它确实是 `train_graph_mappo` 的父进程，
再按 pgid 发 TERM。

## ⚠ 后果（必须记下）

被停的臂 `metrics.jsonl` 里会留下 u31..u33 的记录。**日后若再续跑同一臂，
会出现重复 update 号** —— 那是"重启"，**不是**记忆里说的双写损坏。
判健康度时要能区分这两者。
"""
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/opt/qkd/graph_mappo")


def find_parents():
    """{run_name: (pid, ppid, pgid)}"""
    out = {}
    r = subprocess.run("ps -eo pid,ppid,pgid,comm,args",
                       shell=True, capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        parts = ln.split(None, 4)
        if len(parts) < 5:
            continue
        pid, ppid, pgid, comm, args = parts
        if not comm.startswith("python"):
            continue
        if "train_graph_mappo" not in args or "--run-name" not in args:
            continue
        toks = args.split()
        nm = toks[toks.index("--run-name") + 1]
        out[nm] = (int(pid), int(ppid), int(pgid))
    return out


def worker_count(pgid):
    """该进程组里 worker 的个数（验证整组会一起收到 TERM）。"""
    n = 0
    r = subprocess.run("ps -eo pgid,args", shell=True, capture_output=True, text=True)
    for ln in r.stdout.splitlines()[1:]:
        parts = ln.split(None, 1)
        if len(parts) < 2:
            continue
        if parts[0] == str(pgid) and ("multiprocessing" in parts[1]
                                      or "spawn_main" in parts[1]):
            n += 1
    return n


def memavail():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    raise RuntimeError("no MemAvailable")


def main():
    if len(sys.argv) < 2:
        print("用法: reap_arms.py <臂名> [臂名...]")
        return 2
    targets = sys.argv[1:]

    parents = find_parents()
    print(f"当前在跑的父进程 {len(parents)} 条：")
    for nm, (pid, ppid, pgid) in sorted(parents.items()):
        print(f"  {nm:<20} pid={pid} pgid={pgid} workers={worker_count(pgid)}")

    print(f"\n目标（停）：{targets}")
    missing = [t for t in targets if t not in parents]
    if missing:
        print(f"★ 找不到这些臂的父进程：{missing} ⟹ 不猜，直接报错")
        return 2

    before = memavail()
    print(f"\n停止前 MemAvailable = {before:.1f} GiB")

    for t in targets:
        pid, ppid, pgid = parents[t]
        nw = worker_count(pgid)
        print(f"\n--- 停 {t}  (pid={pid}, pgid={pgid}, {nw} 个 worker) ---")
        try:
            os.killpg(pgid, 15)          # SIGTERM 整组
            print(f"    SIGTERM → pgid {pgid}")
        except ProcessLookupError:
            print(f"    ★ 进程组已不存在")
        time.sleep(2)
        # 验证
        still = find_parents()
        if t in still:
            print(f"    ⚠ 2s 后还在，等 5s")
            time.sleep(5)
            still = find_parents()
        if t in still:
            print(f"    ★ 仍在运行 ⟹ 补 SIGKILL")
            os.killpg(pgid, 9)
            time.sleep(2)
            still = find_parents()
        print(f"    {'✓ 已停' if t not in still else '★ 仍未停!'}")

    time.sleep(10)
    after = memavail()
    print(f"\n停止后 MemAvailable = {after:.1f} GiB  （释放 {after - before:+.1f}）")

    # 孤儿检查
    r = subprocess.run("ps -eo pid,ppid,args", shell=True, capture_output=True, text=True)
    orph = [ln for ln in r.stdout.splitlines()
            if ln.split()[1:2] == ["1"] and "multiprocessing" in ln]
    print(f"孤儿 worker（PPid==1 且含 multiprocessing）：{len(orph)} 个"
          f"  {'✓' if not orph else '★ 需清理'}")
    for ln in orph[:5]:
        print(f"    {ln[:110]}")

    remaining = find_parents()
    print(f"\n最终在跑 {len(remaining)} 条：")
    for nm, (pid, ppid, pgid) in sorted(remaining.items()):
        print(f"  {nm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
