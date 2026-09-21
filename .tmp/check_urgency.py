"""检查 urgency 臂进度：哪些跑完了 u30、哪些还在跑、哪些死了。"""
import glob, json, os, subprocess, sys
from pathlib import Path

REPO = Path("/opt/qkd/graph_mappo")
ARMS = [("%s_s%d" % (a, s)) for a in ("u0ctl", "u1half", "u2bal", "u3two", "u4quad")
        for s in (42, 43, 44, 45, 46)]

# 哪些在跑（问进程表，不问账本）
out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
running = set()
for line in out.splitlines():
    if "train_graph_mappo" in line and "--run-name" in line:
        parts = line.split()
        if "--run-name" in parts:
            running.add(parts[parts.index("--run-name") + 1])

done, running_l, dead, nodir = [], [], [], []
for arm in ARMS:
    d = REPO / "outputs" / arm
    mj = d / "metrics.jsonl"
    if not d.is_dir():
        nodir.append(arm); continue
    last = 0
    if mj.exists():
        for line in open(mj, encoding="utf-8"):
            line = line.strip()
            if line and '"update"' in line:
                try:
                    last = max(last, int(json.loads(line).get("update", 0)))
                except Exception:
                    pass
    if last >= 30:
        done.append((arm, last))
    elif arm in running:
        running_l.append((arm, last))
    else:
        dead.append((arm, last))

print("已完成 u30 : %d  %s" % (len(done), [a for a, _ in done]))
print("在跑       : %d  %s" % (len(running_l), ["%s(u%d)" % t for t in running_l]))
print("未启动     : %d  %s" % (len(nodir), nodir))
if dead:
    print("★ 疑似死掉 : %d  %s" % (len(dead), ["%s(u%d)" % t for t in dead]))
print()
print("在跑进程数 = %d" % len(running))
