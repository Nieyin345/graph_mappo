"""看住 v2 那 5 条臂，全部到终态就写完成标记。

## 判据（★ 覆盖所有终态，不只覆盖成功）

`training-runs-can-deadlock-frozen`：**别把慢当死，也别把死当慢**。
`activity-check-must-not-use-cumulative-stats`：`ps` 的 %CPU 是开机至今均值，
冻死的进程仍显示高 CPU ⟹ 只有 **CPU tick 增量**可信。

所以每条臂的终态有三种，**都要报**：

  done     轮数达到 --num-updates
  dead     进程没了、但轮数没到（★ 这是要吵的那个）
  running  进程在且轮数没到（继续等）

「进程在但 tick 不涨」额外单独判：连续 3 次采样 tick 增量为 0 就报 frozen
（★ 不是判死，是报出来让人看 —— 慢和死要能区分）。

## 为什么分离运行

ssh 会在 2 小时里掉几次（实测过 `Connection reset`）。看门脚本跑在**服务器
自己的** setsid 会话里，本地只轮询 `/tmp/watch_v2.done` ⟹ ssh 掉线不丢看门。
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
DONE_FLAG = Path("/tmp/watch_v2.done")
ARMS = [f"v2_bottleneck_s{s}" for s in (42, 43, 44, 45, 46)]
TARGET_UPDATES = 30
POLL = 60
PATTERN = "train_graph_mappo"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def update_count(arm):
    """已完成的 update 轮数。

    ★ 数 `"update"` 出现的行（`reading-a-number-is-not-reading-the-metric`：
    `grep -c .` 会把 eval 行也算成一轮，给出偏大的轮数）。
    """
    f = OUT / arm / "metrics.jsonl"
    if not f.exists():
        return 0
    n = 0
    for line in f.read_text(errors="replace").splitlines():
        if '"update"' in line:
            n += 1
    return n


def proc_state(arm):
    """返回 (pid, utime+stime 总 tick)。用 comm 锚定真 python，别用 pgrep -f。"""
    out = sh(f"ps -eo pid,comm,args | awk '$2 ~ /^python/ && /{PATTERN}/'")
    for line in out.splitlines():
        if arm in line:
            pid = line.split()[0]
            st = Path(f"/proc/{pid}/stat")
            if st.exists():
                parts = st.read_text().split()
                # utime=14, stime=15 (1-indexed) -> 13,14 (0-indexed)
                return pid, int(parts[13]) + int(parts[14])
            return pid, -1
    return None, 0


def main():
    print(f"[watch_v2] 开始看 {len(ARMS)} 条臂，目标 {TARGET_UPDATES} 轮", flush=True)
    prev_tick = {}
    zero_streak = {}
    reported = {}

    while True:
        states = {}
        for arm in ARMS:
            n = update_count(arm)
            pid, tick = proc_state(arm)
            if n >= TARGET_UPDATES:
                st = "done"
            elif pid is None:
                st = "dead"
            else:
                st = "running"

            # 冻结检测：只对 running 做
            note = ""
            if st == "running" and tick >= 0:
                if arm in prev_tick:
                    if tick == prev_tick[arm]:
                        zero_streak[arm] = zero_streak.get(arm, 0) + 1
                    else:
                        zero_streak[arm] = 0
                    if zero_streak.get(arm, 0) >= 3:
                        note = f" ★★ CPU tick 连续 {zero_streak[arm]} 次不涨（疑似冻结，不是慢）"
                prev_tick[arm] = tick

            states[arm] = st
            msg = f"{arm}  u={n}/{TARGET_UPDATES}  {st}{note}"
            # 状态串或轮数变了才打印（不然每 60s 刷一堆一样的行）
            key = (st, n)
            if reported.get(arm) != key or note:
                print(f"[watch_v2] {time.strftime('%H:%M:%S')}  {msg}", flush=True)
                reported[arm] = key

        if all(s in ("done", "dead") for s in states.values()):
            done = [a for a, s in states.items() if s == "done"]
            dead = [a for a, s in states.items() if s == "dead"]
            print(f"[watch_v2] 全部到终态。done={done} dead={dead}", flush=True)
            DONE_FLAG.write_text(json.dumps(
                {"done": done, "dead": dead,
                 "updates": {a: update_count(a) for a in ARMS}}, ensure_ascii=False))
            return 0
        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main())
