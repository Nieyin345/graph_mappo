# -*- coding: utf-8 -*-
"""逐个训练 run 量 PSS（父 + spawn worker），按 run 分组。
判据：PSS 才是真占用（RSS 会把共享页重复计数，误导性极强）。"""
import os, re, sys
from collections import defaultdict

def read_pss(pid):
    try:
        with open("/proc/%d/smaps_rollup" % pid, encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("Pss:"):
                    return int(ln.split()[1]) / 1024.0 / 1024.0   # GB
    except OSError:
        pass
    return 0.0

def cmdline(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            return f.read().decode("utf-8", "replace").replace("\0", " ")
    except OSError:
        return ""

def ppid(pid):
    try:
        with open("/proc/%d/stat" % pid, encoding="utf-8") as f:
            s = f.read()
        return int(s[s.rindex(")") + 2:].split()[1])
    except Exception:
        return 0

pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
# 找 trainer 父进程
trainers = {}
for p in pids:
    cl = cmdline(p)
    m = re.search(r"--run-name\s+(\S+)", cl)
    if m and "train_graph_mappo" in cl:
        trainers[p] = m.group(1)
# 归属：每个进程向上找 trainer 祖先
owner = {}
for p in pids:
    cur, seen = p, set()
    while cur > 1 and cur not in seen:
        seen.add(cur)
        if cur in trainers:
            owner[p] = cur
            break
        cur = ppid(cur)

per = defaultdict(float); nproc = defaultdict(int)
for p in pids:
    t = owner.get(p)
    if t is None:
        continue
    per[trainers[t]] += read_pss(p)
    nproc[trainers[t]] += 1

tot = sum(per.values())
print("=== 每 run PSS（父 + 全部后代）===")
for r in sorted(per, key=lambda k: -per[k]):
    print("  %-18s %6.2f GB  (%d 进程)" % (r, per[r], nproc[r]))
print("\n在跑 run 数 = %d" % len(per))
if per:
    print("合计 = %.1f GB   均值 = %.2f GB/run" % (tot, tot / len(per)))
    print("外推满并发（按当前均值）：9 run → %.1f GB" % (tot / len(per) * 9))
