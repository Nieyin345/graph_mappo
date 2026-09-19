#!/usr/bin/env python
"""核实各 run 的线程数，判断配对对照是否可比。

为什么关键：本项目的线程数会**确定性**改变训练结果（记忆库实测差 0.018），
而我要测的 entropy 效应是 0.04 量级——若对照组与实验组线程数不同，
效应会被线程差污染一半。OMP_NUM_THREADS 是起进程时的环境变量，
不会写进 yaml，所以只能从 resolved_config 的 runtime 段 + 启动日志反推。

用法：python /tmp/thread_audit.py r8_base_s42 ent01_s42 ...
"""
from __future__ import annotations

import glob
import os
import sys

import yaml

R = "/opt/qkd/graph_mappo/outputs"


def main():
    names = sys.argv[1:]
    if not names:
        names = [os.path.basename(d) for d in sorted(glob.glob(R + "/*"))
                 if os.path.isdir(d)]

    print(f"{'run':<18}{'runtime.num_threads':>22}{'torch_num_threads':>20}"
          f"{'workers':>9}{'ent':>8}{'gamma':>8}")
    print("-" * 90)
    for n in names:
        p = os.path.join(R, n, "resolved_config.yaml")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            c = yaml.safe_load(fh) or {}
        rt = c.get("runtime") or {}
        tr = c.get("train") or {}
        ppo = tr.get("ppo") or {}
        print(f"{n:<18}{str(rt.get('num_threads')):>22}"
              f"{str(rt.get('torch_num_threads')):>20}"
              f"{str(tr.get('n_rollout_workers')):>9}"
              f"{str(ppo.get('entropy_coef')):>8}"
              f"{str(tr.get('gamma')):>8}")

    # 从启动日志里找 OMP_NUM_THREADS 痕迹
    print("\n=== 启动日志里的线程线索 ===")
    import re
    for n in names:
        for cand in (f"/tmp/{n}.log",):
            if not os.path.exists(cand):
                continue
            with open(cand, encoding="utf-8", errors="replace") as fh:
                txt = fh.read(4000)
            m = re.findall(r"(?:OMP_NUM_THREADS|threads?)\s*[=:]\s*(\d+)", txt)
            if m:
                print(f"  {n}: {m[:4]}")


if __name__ == "__main__":
    main()
