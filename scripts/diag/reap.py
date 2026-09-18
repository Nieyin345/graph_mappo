#!/usr/bin/env python
"""安全回收：孤儿 worker + 我遗留的探针父进程。

为什么需要它（看门狗救不了这两类）：
  1. OOM 杀掉训练父进程后，8 个 multiprocessing.spawn 的 rollout worker
     会被 reparent 到 init（PPid=1）并**永不退出**，每个约 1.1 GB。
  2. **探针脚本自己**（.tmp/probe_*.py）跑完不退出，父进程可以留 5 GB 以上。
     实测 verify_loss_refactor.py 留了 5.17 GB、probe_loss_breakdown.py 留了 4.16 GB。
     看门狗只匹配 cmdline 含 "multiprocess"，抓不到这一类。

判据必须严：只杀 PPid==1（真孤儿）或明确在白名单里的 .tmp 探针，且**永不**
碰任何有存活训练父进程的子进程。

用法：
  python .tmp/reap.py                # 只报告
  python .tmp/reap.py --apply        # 真杀
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time


def proc_table():
    out = subprocess.run(["ps", "-eo", "pid,ppid,etime,args"],
                         capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, etime, args = parts
        if not pid.isdigit():
            continue
        rows.append({"pid": int(pid), "ppid": int(ppid),
                     "etime": etime, "args": args})
    return rows


def pss_gb(pid: int) -> float:
    try:
        tot = 0
        with open(f"/proc/{pid}/smaps_rollup", encoding="ascii") as f:
            for line in f:
                if line.startswith("Pss:"):
                    tot += int(line.split()[1])
        return tot / 1048576.0
    except OSError:
        return 0.0


PROBE_RE = re.compile(r"\.tmp/(probe_|verify_|bench_|analyze_|ab_|mine_|collect_|"
                      r"pss_trend|mem_audit|orphan_audit|thread_audit|seed_audit|"
                      r"diff_runs|show_val|val_)")


def parse_etime(s: str) -> float:
    """把 ps 的 etime（[[DD-]HH:]MM:SS）换成分钟。"""
    days = 0.0
    if "-" in s:
        d, _, s = s.partition("-")
        days = float(d)
    parts = [float(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return days * 1440 + parts[0] * 60 + parts[1] + parts[2] / 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真杀（默认只报告）")
    ap.add_argument("--min-gb", type=float, default=0.2,
                    help="只处理 PSS 超过此值的进程。**默认 0.2，不是 0.5**："
                         "本机孤儿 worker 实测每个 0.29–0.30 GB，旧的 0.5 默认值"
                         "会静默漏掉 14/16 个（漏 5.19 GB）还报'无可回收'。"
                         "阈值按旧数量级设定即静默失效——与 mem_watchdog 同类错误。")
    ap.add_argument("--min-age-min", type=float, default=20.0,
                    help="只处理存活超过此分钟数的探针（保护正在跑的探针）")
    ap.add_argument("--protect", type=int, nargs="*", default=[],
                    help="永不触碰的 PID（正在跑的自有探针）")
    args = ap.parse_args()

    rows = proc_table()
    trainers = {r["pid"] for r in rows if "train_graph_mappo.py" in r["args"]}
    live_children = {r["pid"] for r in rows if r["ppid"] in trainers}
    protect = set(args.protect)
    # 连带保护被保护进程的子进程
    for r in rows:
        if r["ppid"] in protect:
            protect.add(r["pid"])

    targets = []
    skipped_gb: list[tuple[float, int, str]] = []      # 被 --min-gb 筛掉的
    skipped_age: list[tuple[float, int, str]] = []     # 被 --min-age-min 筛掉的探针
    for r in rows:
        pid, ppid, a = r["pid"], r["ppid"], r["args"]
        if pid in live_children or pid in trainers or pid in protect:
            continue                      # 绝不碰在跑的训练
        if "python" not in a:
            continue
        is_orphan_worker = (ppid == 1 and "multiprocessing.spawn" in a)
        # 探针父进程卡在 do_wait（等永不退出的 worker）时 PPid 不是 1 ——
        # 实测 verify_loss_refactor.py 卡了 24 分钟、probe_loss_breakdown.py
        # 卡了 79 分钟，各占 5.17 / 4.16 GB。它们是**我自己**的一次性脚本，
        # 匹配 PROBE_RE 即可杀，不需要 ppid 条件；但要加存活时长门槛，
        # 否则会误杀当前正在产出结果的探针。
        looks_like_probe = bool(PROBE_RE.search(a))
        is_probe = (looks_like_probe
                    and parse_etime(r["etime"]) >= args.min_age_min)
        if not (is_orphan_worker or is_probe):
            if looks_like_probe:
                skipped_age.append((pss_gb(pid), pid, r["etime"]))
            continue
        g = pss_gb(pid)
        if g < args.min_gb:
            # **必须记下来**：阈值按旧数量级设定时会静默吃掉真目标。
            # 不报"筛掉几个"的工具，跨量级变化时只会安静地少报。
            skipped_gb.append((g, pid, r["etime"]))
            continue
        targets.append((g, pid, "孤儿worker" if is_orphan_worker else "遗留探针",
                        r["etime"], a))

    if skipped_gb:
        # 只报计数与合计，不逐条打印（阈值调错时这里会变成一条长清单，
        # 那一行本身就是"阈值与该机当前量级不符"的信号）。
        print(f"\n[阈值筛掉] --min-gb={args.min_gb} 跳过了 {len(skipped_gb)} 个孤儿 worker，"
              f"合计 {sum(s[0] for s in skipped_gb):.2f} GB"
              f"（最大单个 {max(s[0] for s in skipped_gb):.2f} GB）"
              f"\n            若这个数不可忽略，说明阈值与本机当前量级不符，"
              f"用 --min-gb 调小重跑。")
    if skipped_age:
        print(f"\n[时长筛掉] --min-age-min={args.min_age_min} 跳过 {len(skipped_age)} 个疑似探针"
              f"（最大 {max(s[0] for s in skipped_age):.2f} GB）——"
              f"若其中有卡死的，调小该阈值或加 --protect 后重跑。")

    if not targets:
        print(f"无可回收目标（训练父进程 {len(trainers)} 个，"
              f"在跑子进程 {len(live_children)} 个）")
        return

    targets.sort(reverse=True)
    total = sum(t[0] for t in targets)
    print(f"{'PSS_GB':>8}  {'PID':>7}  {'存活':>9}  {'类型':<10} 命令")
    print("-" * 100)
    for g, pid, kind, et, a in targets:
        print(f"{g:>8.2f}  {pid:>7}  {et:>9}  {kind:<10} {a[:52]}")
    print(f"\n合计可回收 {total:.2f} GB / {len(targets)} 个进程")

    if not args.apply:
        print("\n（只报告。加 --apply 真杀）")
        return

    for g, pid, kind, et, a in targets:
        try:
            os.kill(pid, 15)
        except OSError as e:
            print(f"  kill {pid} 失败: {e}")
    time.sleep(3)
    still = [t for t in targets if os.path.exists(f"/proc/{t[1]}")]
    for g, pid, kind, et, a in still:
        try:
            os.kill(pid, 9)
        except OSError:
            pass
    time.sleep(1)
    freed = sum(t[0] for t in targets if not os.path.exists(f"/proc/{t[1]}"))
    mi = {}
    with open("/proc/meminfo", encoding="ascii") as f:
        for line in f:
            k, _, v = line.partition(":")
            mi[k.strip()] = int(v.split()[0]) / 1048576.0
    print(f"已回收 {freed:.2f} GB；MemAvailable 现在 "
          f"{mi.get('MemAvailable', -1):.1f} GB")


if __name__ == "__main__":
    main()
