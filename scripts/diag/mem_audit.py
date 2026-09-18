#!/usr/bin/env python
"""按 PSS 分组汇总内存，并解释 MemAvailable 为什么低于预期。

**为什么不能看 RSS**：RSS 把共享页在每个进程里重复计数（torch 的 .so、
h5 数据映射都是共享的），对"还能不能再开一个 run"这个问题误导性极强。
本项目实测过这一点，所以一律读 `/proc/<pid>/smaps_rollup` 的 `Pss`。

回答的是这类问题：
  - 125 GB 的机器，4 个 run 怎么吃掉 103 GB？（父进程 ~15 GB × 4 + worker 1.2 GB × 8 × 4）
  - MemAvailable 只剩 20 GB，是 run 正常增长还是有一堆孤儿挂着？
  - 哪个 PID 该 reap？

用法（服务器上）：
  /opt/qkd/venv/bin/python scripts/diag/mem_audit.py
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path


def read_pss_kb(pid: str) -> tuple[float, float] | None:
    """返回 (Pss, 私有页) 单位 GB。私有页高 = 这份内存是它独有的。"""
    try:
        txt = Path(f"/proc/{pid}/smaps_rollup").read_text()
    except (OSError, PermissionError):
        return None
    pss = priv = 0
    for line in txt.splitlines():
        if line.startswith("Pss:"):
            pss = int(line.split()[1])
        elif line.startswith(("Private_Clean:", "Private_Dirty:")):
            priv += int(line.split()[1])
    return pss / 1048576.0, priv / 1048576.0


def read_status(pid: str) -> tuple[str, str]:
    """返回 (PPid, cmdline)。"""
    ppid = "?"
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                ppid = line.split()[1]
                break
    except OSError:
        pass
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ppid, "?"
    return ppid, raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()


def group_of(cl: str) -> str:
    if "train_graph_mappo" in cl:
        seg = cl.split("--run-name")
        return "trainer:" + (seg[1].strip().split()[0] if len(seg) > 1 else "?")
    if "multiprocessing.spa" in cl:
        return "worker"
    if "probe_" in cl or "verify_" in cl or "reap.py" in cl:
        return "tool/probe"
    return "other"


def main():
    rows = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        r = read_pss_kb(pid)
        if r is None:
            continue
        pss, priv = r
        ppid, cl = read_status(pid)
        rows.append((pss, priv, pid, ppid, group_of(cl), cl))

    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[r[4]].append(r)

    print(f"{'组':<26}{'进程数':>6}{'PSS合计GB':>12}{'最大单个GB':>12}")
    print("-" * 58)
    tot = 0.0
    for g in sorted(groups, key=lambda k: -sum(r[0] for r in groups[k])):
        rs = groups[g]
        s = sum(r[0] for r in rs)
        tot += s
        print(f"{g:<26}{len(rs):>6}{s:>12.2f}{max(r[0] for r in rs):>12.2f}")
    print("-" * 58)
    print(f"{'合计':<26}{len(rows):>6}{tot:>12.2f}")

    # trainer 的 worker 应该是 8 的倍数；多出来的就是孤儿。
    # **别靠命令行认领**：worker 的 cmdline 里没有 run-name，无法自证归属
    # （项目里踩过这个坑），所以这里只按 PPid==1 判定，不猜它是谁的。
    n_train = len([1 for r in rows if r[4].startswith("trainer:")])
    n_worker = len(groups.get("worker", []))
    expect = 8 * n_train
    print(f"\nworker {n_worker} 个 / {n_train} 个 trainer × 8 = {expect}"
          + ("（相符）" if n_worker == expect else f"，**多出 {n_worker - expect} 个**"))

    orph = sorted([r for r in rows if r[3] == "1" and r[4] == "worker"],
                  key=lambda r: -r[0])
    print(f"\n=== 孤儿 worker（PPid==1）{len(orph)} 个 ===")
    for pss, _priv, pid, _pp, _g, _cl in orph:
        print(f"  PSS {pss:5.2f}GB  pid {pid}")
    print(f"  小计 {sum(r[0] for r in orph):.2f} GB"
          + ("   → 用 scripts/diag/reap.py --apply 回收" if orph else ""))

    print("\n=== PSS 前八 ===")
    for pss, priv, pid, ppid, g, _cl in sorted(rows, key=lambda r: -r[0])[:8]:
        print(f"  PSS {pss:5.2f}GB  私 {priv:5.2f}GB  pid {pid:<7} ppid {ppid:<7} {g}")

    try:
        mi = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            mi[k] = int(v.split()[0]) / 1048576.0
        print("\n=== /proc/meminfo（GB）===")
        for k in ("MemTotal", "MemFree", "MemAvailable", "Cached", "SReclaimable",
                  "Shmem", "Slab"):
            if k in mi:
                print(f"  {k:<14}{mi[k]:8.2f}")
        # 本项目判据：每个在跑的 run 按 30 GB 算（23.3 起、每轮 +0.21 增长）。
        if "MemAvailable" in mi:
            print(f"\n  按 30 GB/run 预算 → 还能再开 {int(mi['MemAvailable'] // 30)} 个 run")
    except OSError:
        pass


if __name__ == "__main__":
    main()
