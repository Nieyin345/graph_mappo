"""按 **PSS** 分组统计内存，回答「现在这 6 个 run 离 OOM 还有多远」。

### 为什么不能用 `free` / RSS

- `MemAvailable` 把可回收的 page cache 也算进来，在跑训练时它会忽高忽低，
  不能直接当"还能起几个 run"的判据（本项目已记：照启动时的数字排 5 个，
  涨到 u15 就顶格）
- **RSS 会把共享页重复计数**：8 个 spawn worker 共享同一份 torch 代码页，
  各算一遍 RSS 会显著高估。本项目实测过「RSS 误导性极强」
- **PSS**（`/proc/<pid>/smaps_rollup` 的 `Pss:`）把共享页按比例摊分，
  是唯一不重复计数的口径

### 输出

1. 按 run-name 分组的 PSS 合计（父进程 + 它的 worker）
2. 系统总体：已用 / 可用 / 缓存
3. **外推**：按 +0.21 GiB/轮 的实测增长率，算到目标轮数的需求

用法（服务器上）：sudo 不需要，同用户进程都能读 smaps_rollup
    /opt/qkd/venv/bin/python /tmp/mem_pss.py [--to-update 30]
"""
from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path

KB = 1024.0
GROWTH_PER_UPDATE = 0.21      # GiB/轮，实测多点线性拟合均值
UPD_RE = re.compile(r"checkpoint_update_(\d+)\.pt$")


def meminfo() -> dict[str, float]:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, _, rest = line.partition(":")
        v = rest.strip().split()
        if v and v[0].isdigit():
            out[k] = float(v[0]) / KB / KB      # kB -> GiB
    return out


def cmdlines() -> dict[int, str]:
    d = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            d[int(p.name)] = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace")
        except (OSError, PermissionError):
            pass
    return d


def ppid_of(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (OSError, PermissionError):
        pass
    return 0


def pss_of(pid: int) -> float:
    """PSS，GiB。读不到（进程已退出/无权限）返回 0。"""
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return float(line.split()[1]) / KB / KB
    except (OSError, PermissionError):
        pass
    return 0.0


def run_name_of(cmd: str) -> str:
    m = re.search(r"--run-name\s+(\S+)", cmd)
    return m.group(1) if m else ""


def latest_update(run: str) -> int:
    d = Path("/opt/qkd/graph_mappo/outputs") / run
    if not d.is_dir():
        return 0
    us = []
    for p in d.glob("checkpoint_update_*.pt"):
        m = UPD_RE.search(p.name)
        if m:
            us.append(int(m.group(1)))
    return max(us) if us else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to-update", type=int, default=30,
                    help="外推到第几轮（默认 30）")
    a = ap.parse_args(argv[1:])

    cmds = cmdlines()
    # run-name 只在父进程命令行里；worker 是自己没有 run-name 的 spawn 子进程，
    # 所以先把每个 pid 归属到"最近的带 run-name 的祖先"
    owner: dict[int, str] = {}
    for pid, cmd in cmds.items():
        rn = run_name_of(cmd)
        if rn:
            owner[pid] = rn
    # 建立 pid -> 祖先 run（最多上溯 4 层，spawn worker 只有 1-2 层）
    def ancestor_run(pid: int) -> str:
        cur = pid
        for _ in range(5):
            if cur in owner:
                return owner[cur]
            nxt = ppid_of(cur)
            if nxt <= 1:
                return ""
            cur = nxt
        return ""

    groups: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    orphans: dict[str, float] = defaultdict(float)
    total_pss = 0.0
    for pid, cmd in cmds.items():
        if "python" not in cmd:
            continue
        g = ancestor_run(pid)
        if not g:
            continue
        v = pss_of(pid)
        if v <= 0:
            continue
        total_pss += v
        groups[g] += v
        counts[g] += 1
        # 孤儿：PPid==1 的 worker —— OOM 的**结果**，不该混进活跃 run 的账
        if ppid_of(pid) == 1 and "multiprocess" in cmd:
            orphans[g] += v

    mi = meminfo()
    print("=" * 74)
    print("按 PSS 分组（不重复计共享页）")
    print("=" * 74)
    extrap_need = 0.0
    for g in sorted(groups, key=lambda k: -groups[k]):
        u = latest_update(g)
        cur = groups[g]
        remain = max(0, a.to_update - u)
        proj = cur + GROWTH_PER_UPDATE * remain
        extrap_need += proj
        print(f"  {g:<22} {cur:6.1f} GiB  ({counts[g]:>2} 进程)  u={u:<3}"
              f" 外推 u{a.to_update} → {proj:5.1f} GiB")
    print(f"  {'合计':<22} {sum(groups.values()):6.1f} GiB")

    if orphans:
        print()
        print("  ⚠ 孤儿 worker（PPid==1，OOM 的**结果**；只清孤儿不解决复发）：")
        for g, v in orphans.items():
            print(f"      {g:<20} {v:5.1f} GiB")

    print()
    print("=" * 74)
    print("系统")
    print("=" * 74)
    print(f"  MemTotal     {mi.get('MemTotal', 0):8.1f} GiB")
    print(f"  MemFree      {mi.get('MemFree', 0):8.1f} GiB")
    print(f"  MemAvailable {mi.get('MemAvailable', 0):8.1f} GiB   ← 含可回收缓存")
    print(f"  训练 PSS 合计 {total_pss:8.1f} GiB")
    if mi.get("MemTotal"):
        other = mi["MemTotal"] - mi.get("MemFree", 0) - total_pss
        print(f"  其它占用      {other:8.1f} GiB   ← 缓存/系统/非训练进程")

    print()
    print("=" * 74)
    print("外推判据")
    print("=" * 74)
    # ★ 两个视角，缺一不可（[[respawn-guard-two-views]]）：
    #     视角 A（增量）：当前 MemFree 装得下**还要长出来的那部分**吗？
    #     视角 B（绝对）：长到目标轮数后，系统还剩多少？必须 >= 17 GiB。
    #
    #   **第一版把这两个搞混了**：拿 MemFree(135.7) 去比外推**总量**(141.4)，
    #   报「已超 5.7 GiB」。但总量里 108.8 是**已经在用**的，真正新增只有
    #   32.6 GiB —— 真实余量 103 GiB，结论完全相反。
    #   与 [[respawn-guard-two-views]] 记的是同一个坑的**镜像**：那次是只算增量
    #   太宽松，这次是拿绝对量当增量、**把够用报成不够用**。
    #   判据落在错误位置，方向不管是松还是紧，都是错的。
    cur_total = sum(groups.values())
    incremental = extrap_need - cur_total
    other = mi.get("MemTotal", 0) - mi.get("MemFree", 0) - cur_total
    free_now = mi.get("MemFree", 0)
    after = mi.get("MemTotal", 0) - other - extrap_need
    MIN_MARGIN = 17.0

    print(f"  在跑 {len(groups)} 个 run：现在 {cur_total:.1f} GiB → u{a.to_update} "
          f"{extrap_need:.1f} GiB（**还要长 {incremental:+.1f} GiB**）")
    print()
    print(f"  视角 A · 增量：MemFree {free_now:.1f} GiB ≥ 新增 {incremental:.1f} GiB ?  "
          f"{'✓' if free_now >= incremental else '✗'}")
    print(f"           长完还剩 {free_now - incremental:.1f} GiB")
    print(f"  视角 B · 绝对：到 u{a.to_update} 时系统余量 {after:.1f} GiB ≥ {MIN_MARGIN} GiB ?  "
          f"{'✓' if after >= MIN_MARGIN else '✗'}")
    print(f"           （其它占用按当前 {other:.1f} GiB 估；MemAvailable "
          f"{mi.get('MemAvailable', 0):.1f} GiB 含可回收缓存，是乐观上界）")
    print()
    if free_now >= incremental and after >= MIN_MARGIN:
        print(f"  ⟹ **两个视角都过**：现有 {len(groups)} 个 run 可以安全跑完。")
        spare = after - MIN_MARGIN
        print(f"     富余 {spare:.1f} GiB（= {spare / 24:.1f} 个 24GiB 的 run）")
    else:
        print("  ⟹ ✗ **至少一个视角不过** —— 该减臂或降轮数，不要再加。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
