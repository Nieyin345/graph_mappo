#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一行式状态快照：轮数 + 内存 + 控制脚本 + 判读文件。只读，不碰任何 run。"""
import json
import os
import subprocess

R = "/opt/qkd/graph_mappo/outputs"
RUNS = ["ent001_s42", "ent001_s43", "ent001_s44", "mini512_s44",
        "hist32v3_s42", "hist32_s43", "hist32_s44"]
VERDICTS = ["/tmp/wave5_verdict.txt", "/tmp/ent001_verdict.txt",
            "/tmp/hist_verdict.txt"]


def upd(run):
    p = os.path.join(R, run, "metrics.jsonl")
    if not os.path.exists(p):
        return None
    n = 0
    for ln in open(p, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            try:
                if json.loads(ln).get("update"):
                    n += 1
            except ValueError:
                pass
    return n


def avail():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / 1e6
    return 0.0


print("时间 %s   可用 %.1f GB" % (
    subprocess.run(["date", "-u", "+%FT%TZ"], capture_output=True,
                   text=True).stdout.strip(), avail()))
print("--- 轮数 ---")
for r in RUNS:
    n = upd(r)
    print("  %-14s %s" % (r, "u%d" % n if n is not None else "（无）"))

print("--- 控制脚本 ---")
out = subprocess.run(["pgrep", "-af", "chain.py"], capture_output=True, text=True).stdout
lines = [l for l in out.splitlines() if "pgrep" not in l]
print("  " + ("\n  ".join(l[:70] for l in lines) if lines else "（无）"))

print("--- 判读文件 ---")
for v in VERDICTS:
    if os.path.exists(v):
        with open(v, encoding="utf-8") as f:
            head = f.readline().strip()
        print("  ★ %s  (%s)" % (v, head))
    else:
        print("  - %s 未有" % v)
