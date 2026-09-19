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
        # ★ `-ww`：预防性（见 reap.py 里的实测记录 —— 管道下当前不截断，
        #   但宽度规则随实现/环境变，行为不该依赖默认）。
        out = subprocess.run(["ps", "-ewwo", "pid,ppid,args"],
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
    print_fits(args.out)


def fit_slopes(path: str) -> list[dict]:
    """对每个 run 做 PSS ~ update 的线性拟合。**纯函数，可造反证。**

    ★★ 为什么把它从 `main()` 里**拎出来**（2026-09-21，与 `watch_liveness.py`
      的 `classify()` 同一条教训）：这段判据原先**内联在 main 的打印循环里**，
      于是"它到底在什么条件下拒绝拟合"只能靠人读源码确认 —— 而**判据一旦
      只能靠人读源码来确认，就一定会与它自己的打印分家**。

    返回每条 run 的 dict：`{run, n_raw, n_uniq, slope|None, u0, u1, y0, y1,
    reason|None}`。`slope is None` ⟹ **不拟合**，`reason` 说明为什么。
    """
    data = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["update"]) > 0:
                data[r["run"]].append((float(r["update"]), float(r["pss_gb"])))

    out = []
    for nm, pts in data.items():
        # 只保留每轮首个采样，避免同一轮重复点拉平斜率
        seen = {}
        for u, p in pts:
            seen.setdefault(u, p)
        us = sorted(seen)
        n_uniq = len(us)
        rec = {"run": nm, "n_raw": len(pts), "n_uniq": n_uniq,
               "slope": None, "reason": None, "u0": None, "u1": None,
               "y0": None, "y1": None}
        # ★★ 判据必须查**去重后**的点数（B-14）。旧实现只查 `len(pts) < 3`
        #   （**原始**采集数），而去重后可能只剩 1 点 ⟹ `den = Σ(x−mx)² = 0`
        #   ⟹ `if den else 0.0` 把斜率**打印成 `+0.000 GB/轮`**，区间显示
        #   `10→10 轮` —— **看起来是一次有效的、平坦的测量**。
        #   同族 `silent-lenient-fallback-in-thresholds`：那里的 `.get(df, 2.0)`
        #   制造**假显著**，这里的 `else 0.0` 制造**假平坦**。
        #   ⟹ 分不清「量出来是平的」与「根本没量」，斜率就不该打印。
        if n_uniq < 3:
            rec["reason"] = ("去重后只有 %d 个不同轮号（原始 %d 条采样）"
                             % (n_uniq, len(pts)))
            out.append(rec)
            continue
        xs = [float(u) for u in us]
        ys = [seen[u] for u in us]
        n = n_uniq
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den <= 0:
            # n ≥ 3 且轮号互不相同 ⟹ den > 0 恒成立。留这道断言是为了
            # **让"不可达"显式**：真出现了说明 us 里有重复键（不该发生）。
            rec["reason"] = "自变量无方差（den=%r）—— 不该发生，请查重复轮号" % den
            out.append(rec)
            continue
        rec.update(slope=sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den,
                   u0=us[0], u1=us[-1], y0=ys[0], y1=ys[-1])
        out.append(rec)
    return out


def print_fits(path: str) -> None:
    print("\n=== 每轮增长（线性拟合 PSS/轮）===")
    for r in fit_slopes(path):
        if r["slope"] is None:
            print(f"  {r['run']:<22} ⚠ **不拟合**：{r['reason']}"
                  f" ⟹ 斜率无意义（**不是 0**）")
        else:
            print(f"  {r['run']:<22} {r['slope']:+.3f} GB/轮  "
                  f"({r['u0']}→{r['u1']} 轮, {r['y0']:.2f}→{r['y1']:.2f} GB, "
                  f"n={r['n_uniq']} 点)")


if __name__ == "__main__":
    main()
