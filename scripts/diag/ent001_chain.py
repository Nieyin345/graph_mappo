#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ent001 链：等 3 臂到 u30 → 跑判读 → 写 /tmp/ent001_verdict.txt。

### 为什么需要它
`parallel_chain.py`（并行启动器）**只负责启动，不写判读**。而本地那个等
`/tmp/ent001_verdict.txt` 的接力器会**永远等下去** —— 因为节点上从来没有
产生这个文件的脚本。这正是记忆 `wait-handles-must-have-timeout` 记的
「坏了与还在跑从外部无法区分」。

### 纪律
- **超时不是可选项**：`WAIT_MAX` 到了就写 timeout 标记并退出，不无限等。
- **三态**：done / timeout 写进同一个文件的开头，便于本地判读是哪种。
- 不杀任何在跑的 run；只读 `metrics.jsonl`。
"""
import json
import os
import subprocess
import sys
import time

ROOT = "/opt/qkd/graph_mappo"
PY = "/opt/qkd/venv/bin/python"
VERDICT = "/tmp/ent001_verdict.txt"
LOG = "/tmp/ent001_chain.log"
SEEDS = (42, 43, 44)
UPDATES = 30
WAIT_MAX = 8 * 3600


def log(m):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def last_update(run):
    p = os.path.join(ROOT, "outputs", run, "metrics.jsonl")
    n = 0
    try:
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                if json.loads(ln).get("update"):
                    n += 1
            except ValueError:
                pass
    except OSError:
        return -1
    return n


def write(state, body):
    with open(VERDICT, "w", encoding="utf-8") as f:
        f.write("STATE=%s\n" % state)
        f.write(body)


def main():
    log("=" * 72)
    log("ent001 判读链启动（等 3 臂到 u%d）" % UPDATES)

    todo = ["ent001_s%d" % s for s in SEEDS]
    missing = [r for r in todo if last_update(r) < 0]
    if missing:
        log("!! 缺臂：%s —— 不启动等待" % ", ".join(missing))
        write("failed", "缺臂：%s\n" % ", ".join(missing))
        return

    t0 = time.time()
    while time.time() - t0 < WAIT_MAX:
        n = sum(1 for r in todo if last_update(r) >= UPDATES)
        if n == len(todo):
            log("✓ 3 臂全部到 u%d（%.0f 分钟）" % (UPDATES, (time.time() - t0) / 60))
            break
        if int(time.time() - t0) % 900 < 130:
            log("  进度 %d/%d  (%s)" % (n, len(todo),
                 " ".join("%s=u%d" % (r, last_update(r)) for r in todo)))
        time.sleep(150)
    else:
        log("!! 等待超时（%.0f 分钟）—— 按现有数据判读" % (WAIT_MAX / 60))

    log("跑判读")
    r = subprocess.run([PY, "/tmp/ent001_verdict.py"], capture_output=True, text=True)
    txt = r.stdout or ""
    for ln in txt.splitlines():
        log("  " + ln)
    if r.returncode != 0:
        log("!! 判读脚本返回 %d，stderr 尾：\n%s" % (r.returncode, (r.stderr or "")[-800:]))
        write("failed", txt + "\n\nstderr:\n" + (r.stderr or "")[-2000:])
        return

    n_done = sum(1 for r_ in todo if last_update(r_) >= UPDATES)
    write("done" if n_done == len(todo) else "timeout", txt)
    log("判读写入 %s（state=%s）" % (VERDICT, "done" if n_done == len(todo) else "timeout"))


if __name__ == "__main__":
    main()
