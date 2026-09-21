"""诊断：为什么 5 条同时起会 `Too many open files`，而之前的分批不会？

## 已知

- `ulimit -n` = **1024**（软），硬限制 1048576
- 之前 `demandedge` 8 条成功：**分两批**（先 4 条，再 4 条）
- 本次 `layers4` 5 条**同时**起 ⟹ 全挂

## 假说

checkpoint 加载期是 fd 峰值：每条臂
  · 父进程：checkpoint 文件 + job_dir 下每个 worker 一个管道
  · 8 个 spawn worker：各自继承父的 fd **加上**自己的 IPC 管道
  ⟹ N 条臂并发加载时，父+worker 的 fd 总和可能超 1024

## 本探针做什么

**实测**单条臂在 u1 时的 fd 数（父 + workers），据此算并发上限。
不猜，量。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/opt/qkd/graph_mappo")


def fd_counts():
    """{run_name: (父fd, worker fd 合计, worker 数)}"""
    r = subprocess.run("ps -eo pid,ppid,comm,args", shell=True,
                       capture_output=True, text=True)
    parents, workers = {}, []
    for ln in r.stdout.splitlines():
        parts = ln.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, comm, args = parts
        if not comm.startswith("python"):
            continue
        if "train_graph_mappo" in args and "--run-name" in args:
            toks = args.split()
            parents[int(pid)] = toks[toks.index("--run-name") + 1]
        elif "multiprocessing" in args or "spawn_main" in args:
            workers.append((int(pid), int(ppid)))

    def nfd(pid):
        try:
            return len(list(Path(f"/proc/{pid}/fd").iterdir()))
        except Exception:
            return 0

    agg = {nm: [nfd(pid), 0, 0] for pid, nm in parents.items()}
    for wpid, wppid in workers:
        nm = parents.get(wppid)
        if nm in agg:
            agg[nm][1] += nfd(wpid)
            agg[nm][2] += 1
    return agg


def main():
    print("=" * 88)
    print("fd 实测（父 + workers）")
    print("=" * 88)

    # 起一条臂，采 3 个时间点
    LOG = Path("/tmp/fdprobe.log")
    cmd = (f"cd {REPO} && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 setsid nohup "
           f"/opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py "
           f"--configs rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml train_layers4.yaml "
           f"--checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt "
           f"--seed 42 --num-updates 30 --run-name fdprobe_s42 "
           f"> {LOG} 2>&1 < /dev/null &")
    subprocess.run(cmd, shell=True, timeout=25)
    print("\n  起了 1 条探针臂（fdprobe_s42），采 3 个时间点：")
    for t in (10, 40, 90):
        time.sleep(t if t == 10 else t - (10 if t == 40 else 40))
        agg = fd_counts()
        for nm, (p, w, nw) in agg.items():
            if nm.startswith("fdprobe"):
                total = p + w
                print(f"    t≈{t:>3}s  父 {p:>4} + worker {w:>4}（{nw} 个）= {total:>4}")
                print(f"             ⟹ 若 ulimit=1024，可并发 {1024 // max(1,total)} 条"
                      f"（保守）；实测建议留余量 ⟹ {max(1, int(1024 // max(1,total) * 0.7))} 条")
                break
        else:
            print(f"    t≈{t:>3}s  （没找到探针进程）")

    # 停探针
    subprocess.run(
        "ps -eo pid,args | grep fdprobe_s42 | grep -v grep | awk '{print $1}' | "
        "while read p; do kill -TERM -$p 2>/dev/null || kill -TERM $p; done",
        shell=True)
    time.sleep(4)
    print("\n  探针已停。")

    # ulimit 建议
    print("\n  ★ 修复方案（两条，任选，建议都做）：")
    print("     (a) 启动器里 `ulimit -n 65536`（硬限制够，可提）")
    print("     (b) 串行化：每条臂**等前一条到 u1** 再起下一条"
          "（避开 checkpoint 加载期的 fd 峰值重叠）")


if __name__ == "__main__":
    main()
