#!/usr/bin/env python
"""采样每个训练 run 的 PSS 随时间的增长，量化内存泄漏。

背景：看门狗日志显示 MemAvailable 在无孤儿的情况下从 27G 单调掉到 9.9G
（14 分钟 ~17G）。那不是"run 数太多"，是训练循环自己在漏。之前把 OOM
归因于孤儿 worker，孤儿其实是 OOM 被 SIGKILL 的结果，不是原因。

用 PSS 不用 RSS：fork 出来的 rollout worker 共享大量页，RSS 会重复计数。

输出 CSV：时间, run, PID, PSS_GB, 存活轮数

用法：python .tmp/pss_trend.py --minutes 20 --interval 60
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path


def smaps_pss_kb(pid: str) -> int:
    tot = 0
    try:
        with open(f"/proc/{pid}/smaps_rollup", encoding="ascii") as f:
            for line in f:
                if line.startswith("Pss:"):
                    tot += int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return -1
    return tot


def cmdline(pid: str) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def run_name(args: str) -> str:
    m = re.search(r"--run-name\s+(\S+)", args)
    return m.group(1) if m else "(无名)"


def update_count(name: str) -> int:
    p = Path(f"/opt/qkd/graph_mappo/outputs/{name}/metrics.jsonl")
    if not p.exists():
        return -1
    try:
        with p.open(encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return -1


def meminfo():
    info = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k.strip()] = int(v.split()[0]) / 1048576.0
    except OSError:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--out", default="/tmp/pss_trend.csv")
    args = ap.parse_args()

    # 收集所有训练进程（父 + 其所有子进程）
    def snapshot():
        by_run = defaultdict(float)
        det = {}
        out = subprocess.run(["ps", "-eo", "pid,ppid,args"],
                             capture_output=True, text=True).stdout
        parents = {}
        rows = []
        for line in out.splitlines()[1:]:
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            pid, ppid, rest = parts
            rows.append((pid, ppid, rest))
            if "train_graph_mappo.py" in rest and "grep" not in rest:
                parents[pid] = run_name(rest)
        for pid, ppid, rest in rows:
            if pid in parents:
                nm = parents[pid]
            elif ppid in parents:
                nm = parents[ppid]
            else:
                continue
            pss = smaps_pss_kb(pid)
            if pss > 0:
                by_run[nm] += pss / 1048576.0
                det[nm] = det.get(nm, 0) + 1
        return by_run, det, parents

    t0 = time.time()
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_min", "run", "pss_gb", "procs", "update", "memavail_gb",
                    "swap_free_gb"])
        hdr = True
        while (time.time() - t0) / 60 < args.minutes:
            by_run, det, parents = snapshot()
            mi = meminfo()
            tm = (time.time() - t0) / 60
            if hdr:
                print(f"{'t(min)':>7}{'run':<22}{'PSS_GB':>9}{'进程':>6}"
                      f"{'轮':>5}{'MemAvail':>10}", flush=True)
                hdr = False
            for nm in sorted(by_run):
                u = update_count(nm)
                w.writerow([f"{tm:.1f}", nm, f"{by_run[nm]:.2f}", det[nm], u,
                            f"{mi.get('MemAvailable', -1):.2f}",
                            f"{mi.get('SwapFree', -1):.2f}"])
                print(f"{tm:>7.1f}{nm:<22}{by_run[nm]:>9.2f}{det[nm]:>6}{u:>5}"
                      f"{mi.get('MemAvailable', -1):>10.2f}", flush=True)
            f.flush()
            if (time.time() - t0) / 60 < args.minutes:
                time.sleep(args.interval)

    print(f"\nCSV: {args.out}")

    # 泄漏率：对每个 run 做 PSS ~ update 的线性拟合
    print("\n=== 每轮增长（线性拟合 PSS/轮）===")
    data = defaultdict(list)
    with open(args.out, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["update"]) > 0:
                data[r["run"]].append((float(r["update"]), float(r["pss_gb"])))
    for nm, pts in data.items():
        if len(pts) < 3:
            print(f"  {nm:<22} 点太少（{len(pts)}）")
            continue
        # 只保留每轮首个采样，避免同一轮重复点拉平斜率
        seen = {}
        for u, p in pts:
            seen.setdefault(u, p)
        us = sorted(seen)
        xs = [float(u) for u in us]
        ys = [seen[u] for u in us]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
        print(f"  {nm:<22} {slope:+.3f} GB/轮  ({us[0]}→{us[-1]} 轮, "
              f"{ys[0]:.2f}→{ys[-1]:.2f} GB)")


if __name__ == "__main__":
    main()
