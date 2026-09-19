"""把 outputs/scale_* 与 outputs/thr_* 里的实测数据汇成一张吞吐表。

为什么要单独写：这些数字是决定"实验该怎么排"的依据，必须能一眼复核。
口径用 metrics.jsonl 里的 elapsed_s（稳态每轮成本），不用墙钟 —— 墙钟含启动
和 H5 加载，会把薄进程那一侧算便宜。

两个容易错的地方，这里都记着：
  * 目录名里的数字是**每进程线程数**，不是进程数（`thr_t2_1` 是"每进程 2 线程、
    第 1 个进程"）。进程数等于同组的 run 数。
  * `scale_t4`（单进程 4 线程）和 `scale_p4_*`（4 进程 × 4 线程）线程数相同但
    不是同一组，分组键必须把它们分开。

用法（节点上）：
    cd /opt/qkd/graph_mappo && /opt/qkd/venv/bin/python /tmp/summarize_throughput.py
"""

from __future__ import annotations

import glob
import json
import os
import re

MAIN = "/opt/qkd/graph_mappo"


def read_run(path: str) -> tuple[float, float, float] | None:
    """返回 (每轮秒, rollout 秒, update 秒)；没有训练记录则 None。"""
    tot = ro = up = 0.0
    n = 0
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "update" not in d:
            continue
        tot += d.get("elapsed_s", 0.0)
        ro += d.get("rollout_s", 0.0)
        up += d.get("update_s", 0.0)
        n += 1
    return None if not n else (tot / n, ro / n, up / n)


def group_key(name: str) -> tuple[str, int] | None:
    """run 名 -> (组名, 每进程线程数)。组名相同的 run 是同一批并发实验。"""
    m = re.fullmatch(r"scale_t(\d+)", name)          # 单进程，N 线程
    if m:
        return f"单进程 {m.group(1)} 线程", int(m.group(1))
    m = re.fullmatch(r"scale_p(\d+)_\d+", name)      # N 进程，每进程 4 线程
    if m:
        return f"{m.group(1)} 进程 × 4 线程", 4
    m = re.fullmatch(r"thr_t(\d+)_\d+", name)        # 进程数见组内 run 数
    if m:
        return f"每进程 {m.group(1)} 线程", int(m.group(1))
    m = re.fullmatch(r"thr_u\d+_\d+", name)          # 每进程 1 线程
    if m:
        return "每进程 1 线程", 1
    return None


def main() -> None:
    groups: dict[str, tuple[int, list[tuple[float, float, float]]]] = {}
    for d in sorted(glob.glob(f"{MAIN}/outputs/*/")):
        name = os.path.basename(d.rstrip("/"))
        parsed = group_key(name)
        if parsed is None:
            continue
        key, threads = parsed
        r = read_run(os.path.join(d, "metrics.jsonl"))
        if r is not None:
            groups.setdefault(key, (threads, []))[1].append(r)

    rows = []
    for key, (threads, per) in groups.items():
        n_proc = len(per)       # 进程数 = 同组 run 数
        mean = sum(p[0] for p in per) / len(per)
        rows.append((
            key, n_proc, threads, mean,
            sum(p[1] for p in per) / len(per),
            sum(p[2] for p in per) / len(per),
            n_proc / mean,      # 吞吐：全部进程合计每秒完成的训练轮数
        ))
    rows.sort(key=lambda r: r[6])

    print(f"{'组':<22}{'进程':>5}{'线程/个':>8}{'总线程':>7}"
          f"{'每轮秒':>9}{'rollout':>9}{'update':>8}{'吞吐(轮/s)':>12}")
    print("-" * 80)
    for key, n_proc, threads, mean, ro, up, thru in rows:
        print(f"{key:<22}{n_proc:>5}{threads:>8}{n_proc * threads:>7}"
              f"{mean:>9.1f}{ro:>9.1f}{up:>8.1f}{thru:>12.3f}")

    print()
    if rows:
        best = max(rows, key=lambda r: r[6])
        print(f"最高吞吐：{best[0]}（{best[1]} 进程）—— {best[6]:.3f} 轮/s")
        single = [r for r in rows if r[1] == 1]
        if single:
            s = min(single, key=lambda r: r[3])
            print(f"单进程最快：{s[0]} —— 每轮 {s[3]:.1f}s，吞吐 {s[6]:.3f} 轮/s")
            print(f"并行加速比：{best[6] / s[6]:.1f}×")
    print("SUMMARIZE_DONE")


if __name__ == "__main__":
    main()
