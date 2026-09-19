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
    per_run = []                       # [(run, cur, proj, u, remain)]
    for g in sorted(groups, key=lambda k: -groups[k]):
        u = latest_update(g)
        cur = groups[g]
        remain = max(0, a.to_update - u)
        proj = cur + GROWTH_PER_UPDATE * remain
        extrap_need += proj
        per_run.append((g, cur, proj, u, remain))
        print(f"  {g:<22} {cur:6.1f} GiB  ({counts[g]:>2} 进程)  u={u:<3}"
              f" 外推 u{a.to_update} → {proj:5.1f} GiB")
    print(f"  {'合计':<22} {sum(groups.values()):6.1f} GiB")
    if per_run:
        med = sorted(p for _, _, p, _, _ in per_run)[len(per_run) // 2]
        print(f"  （单个 run 外推中位数 {med:.1f} GiB —— **别拿 25 当常数**："
              f"`minibatch 512` ⟹ 29.5、`hist32` ⟹ 52~63，是按配置查表的量）")

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
    # ★★ 两个视角必须**在代数上真的不同**（2026-09-21 修正）。
    #
    #   旧实现的「视角 B」是：
    #       other = MemTotal - MemFree - cur_total
    #       after = MemTotal - other - extrap_need
    #   把 other 代进去：
    #       after = MemTotal - (MemTotal - MemFree - cur_total) - extrap_need
    #             = MemFree + cur_total - extrap_need
    #             = MemFree - incremental            ← ∵ incremental = extrap_need - cur_total
    #   ⟹ **恒等于视角 A 的余量**。两个 ✓/✗ 由构造必然同号，
    #      「缺一不可」的那段注释描述的独立性，代码里**不存在**。
    #     （同族：`cross-check-must-compare-same-population` —— 交叉核对要同人口；
    #       这里是反向的病：两个"独立"视角其实是同一个人。）
    #
    #   但 `MemAvailable` 本来就在手里，所以第二视角**可以**做成真独立：
    #
    #     视角 A · 硬口径：MemFree（不可回收）≥ 还要长的量。**MemFree 是硬约束**。
    #     视角 B · 乐观口径：MemAvailable（含可回收缓存）≥ 还要长的量 + 安全垫。
    #
    #   两者的**差**就是"可回收缓存"这一项 —— 而它恰好是本工具文件头
    #   第一条告诫说的东西（跑训练时忽高忽低，不能当判据）。
    #   ⟹ 让乐观口径去承担"能不能再塞"的判断，硬口径守底线。
    #   ⟹ **两者分歧本身就是信息**：分歧越大，说明缓存越不可靠，
    #      越不该按乐观口径排并发。
    cur_total = sum(groups.values())
    incremental = extrap_need - cur_total
    free_now = mi.get("MemFree", 0)
    avail_now = mi.get("MemAvailable", 0)
    MIN_MARGIN = 17.0
    tot = mi.get("MemTotal", 0)
    # 视角 B 的落点：按乐观口径长完之后，绝对余量还剩多少
    after_opt = avail_now - incremental

    print(f"  在跑 {len(groups)} 个 run：现在 {cur_total:.1f} GiB → u{a.to_update} "
          f"{extrap_need:.1f} GiB（**还要长 {incremental:+.1f} GiB**）")
    print()
    if not groups:
        # ★ 空集不许说「0 个 run 可以安全跑完」—— 与 `watch_liveness`
        #   同一族（`config-enabled-but-term-dead` 的近亲）：**没在算的东西
        #   谈不上"安全"**。空集是第三态：能起，但"能起几个"要按配置算，
        #   不是"现有的能跑完"。
        # ★ 这个分支必须**在打印两视角之前** —— 否则会先输出一行
        #   「还要长 +0.0 GiB」的两视角对比（增量恒 0，两视角必然都 ✓），
        #   再输出「无从外推」，前后自相矛盾：**恒 ✓ 的判据不是判据**。
        print("  ⟹ **没有在跑的 run，无从外推**（第三态：既非通过也非不通过）。")
        print(f"     能起几个 = (MemFree {free_now:.0f} − 垫 {MIN_MARGIN:.0f}) "
              f"÷ 该配置的稳态 PSS。**按配置查表，不是 25**：")
        print("       minibatch 256 → 25｜512 → 29.5｜hist32 → 52~63")
        n_max = max(0, int((free_now - MIN_MARGIN) / 25))
        print(f"     ⟹ 本机此刻按 256 粗算上限 ≈ {n_max} 个。")
        print("     ⚠ 这是**上限**不是建议；起之前先与其它行为者对齐时间点。")
        print("=" * 74)
        return 0
    print(f"  视角 A · 硬（MemFree，不可回收）：{free_now:.1f} GiB ≥ 新增 "
          f"{incremental:.1f} GiB ?  {'✓' if free_now >= incremental else '✗'}")
    print(f"           长完还剩 {free_now - incremental:.1f} GiB")
    print(f"  视角 B · 乐观（MemAvailable，含可回收缓存）：{avail_now:.1f} GiB ≥ 新增 "
          f"{incremental:.1f} GiB + 垫 {MIN_MARGIN:.0f} ?  "
          f"{'✓' if after_opt >= MIN_MARGIN else '✗'}")
    print(f"           长完还剩 {after_opt:.1f} GiB")
    print(f"           ── 两口径的差 = 可回收缓存 {avail_now - free_now:.1f} GiB"
          f"（占 MemTotal 的 {100 * (avail_now - free_now) / tot if tot else 0:.1f}%）")
    print()
    if free_now < incremental:
        print("  ⟹ ✗ **硬口径不过**：MemFree 装不下还要长出来的量。"
              "靠回收缓存可能侥幸撑住，但那是赌 —— 该减臂或降轮数。")
    elif after_opt < MIN_MARGIN:
        print(f"  ⟹ ⚠ 硬口径过、**乐观口径不过**：绝对余量 < {MIN_MARGIN:.0f} GiB。"
              "按 [[respawn-guard-two-views]]：绝对余量才是该拒绝的那一票。")
    else:
        spare = after_opt - MIN_MARGIN
        print(f"  ⟹ **两个口径都过**：现有 {len(groups)} 个 run 可以安全跑完。")
        print(f"     乐观口径富余 {spare:.1f} GiB；硬口径富余 "
              f"{free_now - incremental - MIN_MARGIN:.1f} GiB。")
        print(f"     ⚠ 想再塞一个，按**硬口径的富余**算 —— 且单个 run 的稳态"
              f"要**按配置查表**，不是 25。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
