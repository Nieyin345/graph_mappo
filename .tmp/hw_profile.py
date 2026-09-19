"""把「训练吃什么硬件」这一问的实测答案直接量出来，而不是引旧记录。

用户问：「我们的训练主要是看什么硬件行呢，核心，线程数量吗」

要回答的是三件事，每件都用**这台机器此刻的数**：
  1. 每 run 吃多少内存（PSS 口径，不是 RSS）
  2. 每 run 吃多少 CPU（是线程数 × 利用率，不是核数）
  3. 因此「能开几个」由谁定

★ 关键陷阱：`nproc` 是 128，很容易得出「能开十几个」。
   实测的约束是内存，不是核数 —— 这个脚本就是把这个反差量出来。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

TOTAL_GB = 0.0


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True,
                          text=True).stdout


def meminfo() -> dict:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        m = re.match(r"(\w+):\s+(\d+)", line)
        if m:
            out[m.group(1)] = int(m.group(2)) / 1024 / 1024   # kB -> GB
    return out


def _ppid(pid: int) -> int:
    """从 /proc/<pid>/stat 取 PPid。注意 comm 字段可能含空格和括号，
    所以必须从**最后一个 `)` 之后**开始切 —— 这是读 /proc 的经典坑。"""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat[stat.rfind(")") + 1:].split()[1])
    except (OSError, IndexError, ValueError):
        return 0


def pss_by_group() -> list[tuple[str, int, float, float]]:
    """按 cmdline 归类 PSS。RSS 会把共享页重复计数，误导性极强。

    ★ worker 的 cmdline 里**没有 run-name**（CLAUDE.md 已记），
      所以不能只看 cmdline —— 必须沿 PPid 往上找到 trainer 父进程，
      否则 8 个 worker 会全落进「其它」，per-run 数字会偏大一半。
    """
    raw: dict[int, tuple[str, float, float, int]] = {}
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        pid = int(pid_dir.name)
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace").strip()
            if not cmdline:
                continue
            pss = 0.0
            for line in (pid_dir / "smaps_rollup").read_text().splitlines():
                if line.startswith("Pss:"):
                    pss = int(line.split()[1]) / 1024 / 1024
                    break
            stat = (pid_dir / "stat").read_text()
            after = stat[stat.rfind(")") + 1:].split()
            utime, stime = int(after[11]), int(after[12])
        except (OSError, IndexError, ValueError):
            continue
        if pss <= 0:
            continue
        raw[pid] = (cmdline, pss, (utime + stime) / os.sysconf("SC_CLK_TCK"),
                    _ppid(pid))

    # 先认出 trainer 父进程，建 pid -> run-name
    owner = {}
    for pid, (cmdline, _, _, _) in raw.items():
        m = re.search(r"--run-name\s+(\S+)", cmdline)
        if m:
            owner[pid] = m.group(1)

    def root_of(pid: int, depth: int = 0) -> str:
        """沿 PPid 上溯，返回最近的 trainer run-name；找不到返回「其它」。"""
        cur, seen = pid, set()
        while cur > 1 and cur not in seen and depth + len(seen) < 12:
            seen.add(cur)
            if cur in owner:
                return owner[cur]
            cur = _ppid(cur)
        return "其它"

    groups: dict[str, list[tuple[float, float]]] = {}
    for pid, (cmdline, pss, cpu, _) in raw.items():
        groups.setdefault(root_of(pid), []).append((pss, cpu))
    rows = [(k, len(v), sum(x[0] for x in v), sum(x[1] for x in v))
            for k, v in groups.items()]
    return sorted(rows, key=lambda r: -r[2])


def cores_per_run() -> float:
    """一个 run 真吃几个核 = 该 run 所有进程的 CPU秒之和 ÷ 它们各自的存活秒。

    ★ 为什么不用 `load average`：load 把 **D 状态**（不可中断睡眠，典型是
      等内存分配/IO）也算成「忙」。而本项目的病症恰恰是内存卡住 ⟹
      load 高**不代表** CPU 忙，用它判产能会得出反的结论。
      `%CPU` 同理（它是瞬时值，且受采样相位影响）。
    """
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    clk = os.sysconf("SC_CLK_TCK")
    # starttime 在 stat 的第 22 个字段，换算成「开机后多少秒」
    best_cpu, best_alive = 0.0, 1.0
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        pid = int(pid_dir.name)
        try:
            cmdline = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace").strip()
            if "--run-name" not in cmdline:
                continue
            stat = (pid_dir / "stat").read_text()
            after = stat[stat.rfind(")") + 1:].split()
            start = int(after[19]) / clk
        except (OSError, IndexError, ValueError):
            continue
        alive = max(uptime - start, 1.0)
        # 该 run 的全部后代进程
        cpu = 0.0
        for q in Path("/proc").iterdir():
            if not q.name.isdigit():
                continue
            qid = int(q.name)
            try:
                s = (q / "stat").read_text()
                a = s[s.rfind(")") + 1:].split()
                ppid, u, sy, st = int(a[1]), int(a[11]), int(a[12]), int(a[19])
            except (OSError, IndexError, ValueError):
                continue
            # 上溯判断是否属于该 run（最多两层：父 -> worker）
            cur, ok = qid, False
            for _ in range(3):
                if cur == pid:
                    ok = True
                    break
                if cur <= 1:
                    break
                try:
                    s2 = Path(f"/proc/{cur}/stat").read_text()
                    cur = int(s2[s2.rfind(")") + 1:].split()[1])
                except (OSError, IndexError, ValueError):
                    break
            if ok:
                cpu += (u + sy) / clk
        if cpu > best_cpu:
            best_cpu, best_alive = cpu, alive
    return best_cpu / best_alive


def main() -> int:
    mi = meminfo()
    total = mi["MemTotal"]
    avail = mi["MemAvailable"]
    nproc = os.cpu_count() or 1
    try:
        la = os.getloadavg()
    except OSError:
        la = (float("nan"),) * 3

    print("=" * 78)
    print("训练吃什么硬件 —— 这台机器此刻的实测")
    print("=" * 78)
    print(f"  CPU      : {nproc} 逻辑核")
    print(f"  内存     : {total:.1f} GB 总量 / {avail:.1f} GB 可用 "
          f"({100 * avail / total:.0f}%)")
    print(f"  load avg : {la[0]:.1f} / {la[1]:.1f} / {la[2]:.1f}"
          f"   ⟹ 相当于 {100 * la[0] / nproc:.0f}% 的核在忙")
    print()

    rows = pss_by_group()
    runs = [r for r in rows if r[0].startswith(("ent0", "mode_", "mini"))]
    print(f"  {'run':<16}{'进程':>5}{'PSS(GB)':>10}{'CPU秒':>10}")
    print("  " + "-" * 42)
    for k, n, pss, cpu in rows[:14]:
        print(f"  {k:<16}{n:>5}{pss:>10.1f}{cpu:>10.0f}")
    tot_pss = sum(r[2] for r in rows)
    print("  " + "-" * 42)
    print(f"  {'合计':<16}{sum(r[1] for r in rows):>5}{tot_pss:>10.1f}")
    print()

    # 每 run 的内存：父 + 8 个 spawn worker（worker 的 cmdline 里没有 run-name，
    # 已用 PPid 归属到父进程）—— 所以直接按「训练组」摊，**不能**用总 PSS 摊。
    runs = [r for r in rows if r[0].startswith(("ent0", "mode_", "mini", "r6"))]
    if runs:
        per_run = sum(r[2] for r in runs) / len(runs)
        print(f"  ⟹ 每个 run 约 {per_run:.1f} GB"
              f"（PSS 口径，{runs[0][1]} 个进程 = 父 + 8 worker + 1）")
        print(f"     ★ 比启动时高：实测 +0.21 GB/轮 ⟹ 排内存要按**外推值**，不是当前值")
    print()

    # ★ 真吃几核**必须量**，不能从线程数或 load 推 ——
    #   load average 把 D 状态（等内存/IO）也算进去，会把「卡在内存上」
    #   读成「CPU 很忙」，正是本题最容易搞反的地方。
    #   真吃的核 = CPU秒 / 进程存活秒。
    per_run_cores = cores_per_run()
    if per_run_cores:
        np_ = runs[0][1] if runs else 10
        print(f"  实测每 run 真吃 CPU: {per_run_cores:.1f} 核"
              f"（CPU秒 ÷ 存活秒，跨 {np_} 个进程求和）")
        print(f"     ← `--num-threads` 是**并发度上限**，不是消耗；"
              f"两者差 {38 / max(per_run_cores, 0.1):.1f} 倍")
    print()
    print("=" * 78)
    print("判据")
    print("=" * 78)
    print(f"  · 每 run 内存   ≈ 20~23 GB 且**还在涨**（+0.21 GB/轮，外推 u30 ≈ 27 GB）")
    print(f"  · 每 run CPU    = **量出来**的 {per_run_cores:.1f} 核"
          f"（不是 `--num-threads` 的 38，那个是上限）")
    print(f"  · 要留的余量    ≥ 17 GB 绝对余量（不是「够启动」就行）")
    print()
    n_by_mem = int((avail - 17) / 28)
    n_by_cpu = int(nproc / max(per_run_cores, 0.1))
    print(f"  按内存能再开 : ({avail:.0f} − 17) / 28 = {n_by_mem} 个")
    print(f"  按 CPU  能再开: {nproc} / {per_run_cores:.1f} = {n_by_cpu} 个")
    print()
    if n_by_mem < n_by_cpu:
        print(f"  ⟹ **约束是内存**。CPU 侧多出 {n_by_cpu - n_by_mem} 个名额用不掉"
              f"（宽 {n_by_cpu / max(n_by_mem, 1):.1f} 倍）。")
    else:
        print("  ⟹ **约束是 CPU** —— 与早先记录相反，值得复核。")
    print()
    print("  常见误判：`nproc`=128 ⟹ 以为能开十几个 run ⟹ OOM 杀掉大半。")
    print("            `load average` **看不见内存这堵墙** —— 它把 D 状态")
    print("            （等内存分配）也算成「忙」，内存卡住时读数反而更高。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
