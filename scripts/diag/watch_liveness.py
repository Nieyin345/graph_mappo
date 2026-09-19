#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""活性看门狗：一次问清「哪些 run 真在算、哪些冻住了、哪些链在等一个死东西」。

### 为什么需要它（2026-09-20 的两次真实损失）
1. **两个 run 静默冻结近 2 小时**才被发现。默认的看活方式**全都看不见**：
   `ps` 的 STATUS `S` 与 BLAS 空闲线程无法区分；`ps` 的 %CPU 是**终生均值**
   （冻结进程照样显示 428%）。**只有 CPU tick 增量能判活。**
2. **两条启动链各自卡在一个已经死掉的臂上**，等一个**永远不会长大的文件**：
   `hist32_chain` 等 `hist32v3_s42` u13→u30（进程早没了）、
   `gae90_n5_chain2` 等 `gae90_n5_s45` u5→u30（进程早没了）。
   链的判据是 `last_update(run) >= 30` —— 它问的是**文件**，不是**进程**。
   文件不会自己长，所以链会一直转，最后**按残缺数据判读或直接超时**
   ⟹ 两个预注册测试一起报废。

**共同点：判据盯的是「记录」而不是「世界」。**（同族：
`chain-must-ask-proc-not-its-own-ledger`、`gate-must-print-its-inputs`）

### 它做什么
只读，不杀任何东西。对每个在跑的训练 run 报：
    轮数 / metrics 年龄 / CPU tick 增量（40 s 采样）/ PSS
再对每个在跑的链报：**它正在等哪些 run、那些 run 是否还活着**。

### 判活为什么要 40 s
`Δticks == 0` 在一个采样点上**不一定是死**（可能只是在等锁/IO/一次长 BLAS 调用）。
40 s 是实测校准出来的：冻结者 0 tick，健康者 ~29000~32000 tick，
两者差三个数量级，没有中间地带。**阈值类工具上线前必须用实测校准**
（本项目有一条被错误阈值精确杀掉对照臂的教训）。

★ 2026-09-20 更正：上面这句「没有中间地带」是**实测那一次**的读数，
**不是普遍保证**。真实的冻结签名是「卡在某个 syscall 里永不返回」，
而同样 0 tick 也可能是**一次很长的 `madvise`/锁等待**，会自己恢复。
⟹ 本脚本因此把**单次** 0 增量定为「**疑似**」，并**再采一次**确认；
只有**两次连续** 0 增量才报「冻结」。
（子代理指出本条存在假阳性模式；不采信它的具体机制，但"单点不定死"
  这个方向是对的，且与本文件 docstring 原有的告诫一致——原先代码没照做。）

### 用法
    python scripts/diag/watch_liveness.py              # 只报（默认）
    python scripts/diag/watch_liveness.py --sample 20  # 改采样秒数
退出码：0=全健康；1=有冻结的 run；2=有**在跑的**链在等已经死掉的臂；3=两者都有

★ 假警防护（实测踩过，务必保留）：
  · 链**不在跑**时，它的日志是**历史**，不能拿来判「永远等不到」——
    第一版把正常跑完 u30 的 ent001 三条臂报成「进程已死」（假警）。
  · 同一个链的 `.log` 与 `.out` 是同一份流的两种落盘，要**去重**。
  **假警比漏报更坏**：它留下「检查过了」的假信心。
"""
import argparse
import glob
import json
import os
import re
import sys
import time

ROOT = os.environ.get("QKD_ROOT", "/opt/qkd/graph_mappo")

# 每轮耗时的实测值（秒），用来定「多久没动才算异常」。
# ⚠ 这些是**实测**不是猜的：hist 系一轮 rollout 103 + update 194 ≈ 298 s；
#   基线一轮 rollout 51 + update 102 ≈ 153 s。取 3 倍留余量。
ROUND_S = {"hist": 300.0, "default": 150.0}
STALE_FACTOR = 3.0


def _cmdline(pid):
    """★ `/proc/<pid>/cmdline` 用 NUL 分隔 argv，必须换成空格再上正则，
    否则 `--run-name\\s+(\\S+)` **永远匹配不上** ⟹ 处处数到 0 条
    （`\\s` 不匹配 `\\x00`；这条坑让内存门**恒通过**过，见
    `gate-must-print-its-inputs`）。"""
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            return f.read().decode("utf-8", "replace").replace("\0", " ")
    except OSError:
        return ""


def _ppid(pid):
    try:
        s = open("/proc/%d/stat" % pid, encoding="utf-8").read()
        return int(s[s.rindex(")") + 2:].split()[1])
    except Exception:
        return 0


def _pss(pid):
    try:
        for ln in open("/proc/%d/smaps_rollup" % pid):
            if ln.startswith("Pss:"):
                return int(ln.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def _ticks(pid):
    """utime+stime。**增量**才是活性判据。"""
    try:
        s = open("/proc/%d/stat" % pid, encoding="utf-8").read()
        f = s[s.rindex(")") + 2:].split()
        return int(f[11]) + int(f[12])
    except Exception:
        return None


def rounds_and_age(run):
    """(轮数, metrics.jsonl 年龄秒)。轮数 = 带 `update` 键的**行数**
    （与链的 `last_update()` 同一口径 —— 必须一致，否则我这边的判读
     和链的等待条件会对不上）。"""
    p = os.path.join(ROOT, "outputs", run, "metrics.jsonl")
    n, age = -1, None
    try:
        age = time.time() - os.path.getmtime(p)
    except OSError:
        return -1, None
    n = 0
    try:
        with open(p, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    if json.loads(ln).get("update"):
                        n += 1
                except ValueError:
                    pass
    except OSError:
        pass
    return n, age


def scan():
    """[{run,pid,pss,rounds,age}] + {run: [chain scripts waiting on it]}"""
    pids = [int(x) for x in os.listdir("/proc") if x.isdigit()]
    trainers, children = {}, {}
    for p in pids:
        c = _cmdline(p)
        if "train_graph_mappo" in c:
            m = re.search(r"--run-name\s+(\S+)", c)
            if m:
                trainers[p] = m.group(1)
        pp = _ppid(p)
        if pp:
            children.setdefault(pp, []).append(p)

    out = []
    for pid, name in trainers.items():
        seen, stack, tree = set(), [pid], []
        while stack:
            q = stack.pop()
            if q in seen:
                continue
            seen.add(q)
            tree.append(q)
            stack.extend(children.get(q, []))
        n, age = rounds_and_age(name)
        out.append({"run": name, "pid": pid,
                    "pss": sum(_pss(q) for q in tree),
                    "rounds": n, "age": age, "t0": _ticks(pid)})
    return out, trainers


def chain_waits():
    """从链自己的日志里读出「它在等哪些 run」，并判断链**是否还活着**。

    链的日志格式（两条链都打印进度行）：
        gae90_n5: `进度 3/5 ｜ gae90_n5_s45=u5 gae90_s42=u30 ...`
        hist32  : `进度 0/3 (hist32v3_s42=u13 hist32_s43=u14 ...)`
    解析出 (run -> uN)，后面再判这些 run 是否还活着。

    ★★ 2026-09-20 实测的假警（必须记住）：
    我第一版只看**最后一条进度行**，于是 `ent001_chain` 被报成
    「等 ent001_s42（停在 u27）⟹ 永远等不到」。**全是假的**：
    那三条臂**正常跑完了 u30**（`checkpoint_final.pt` 都在），
    链也**早就写出 verdict 并退出了**（`ent001_chain.log` 最后一行是
    `判读写入 /tmp/ent001_verdict.txt（state=done）`）。
    我把**陈旧日志**当成了**当前状态**。

    ⟹ 判据必须两层：
      (a) 链**进程**还在不在（进程表）——不在 ⟹ 它已经结束了，日志是历史
      (b) 链若在跑，它的**日志 mtime** 必须够新，否则它可能卡住不出声
    **假警比漏报更坏**（它留下"检查过了"的假信心），所以宁可不报。
    """
    res = {}
    logs = glob.glob("/tmp/*_chain*.log") + glob.glob("/tmp/*_chain*.out")
    # 哪些链脚本此刻真的在跑
    live_scripts = {}
    for d in glob.glob("/proc/[0-9]*"):
        c = _cmdline(int(d.rsplit("/", 1)[1]))
        if "_chain" not in c:
            continue
        m = re.search(r"/tmp/([A-Za-z0-9_.]+\.py)", c)
        if m:
            live_scripts[m.group(1)] = int(d.rsplit("/", 1)[1])

    for logf in sorted(set(logs)):
        base = os.path.basename(logf)
        stem = base[:-4] if base.endswith(".log") else base
        # ★ 同一个链的 `.log` 与 `.out` 是同一条流的两种落盘，**去重**
        #   （不去重会把同一条链报两遍，看起来像两条都卡住了）
        if stem.endswith(".out") and os.path.exists("/tmp/%s.log" % stem[:-4]):
            continue
        script = stem + ".py"
        running_pid = live_scripts.get(script)
        try:
            lines = open(logf, encoding="utf-8", errors="replace").readlines()
            age = time.time() - os.path.getmtime(logf)
        except OSError:
            continue
        # 链自己的终态标记（写完就退出了）
        finished = any(("判读写入" in ln) or ("判读已写入" in ln) or
                       ("state=done" in ln) or ("STATE=done" in ln)
                       for ln in lines[-40:])
        prog = None
        for ln in reversed(lines):
            if "进度" in ln:
                prog = ln
                break
        pairs = re.findall(r"([A-Za-z0-9_]+)=(u-?\d+)", prog) if prog else []
        res[base] = {
            "script": script,
            "pid": running_pid,
            "alive": running_pid is not None,
            "finished": finished,
            "log_age": age,
            "waits": {r: int(u[1:]) for r, u in pairs},
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=float, default=40.0,
                    help="CPU tick 采样秒数（默认 40，实测校准值）")
    ap.add_argument("--target", type=int, default=30,
                    help="链等待的目标轮数（默认 30，与预注册一致）")
    a = ap.parse_args()

    print("=" * 88)
    print("活性看门狗 ｜ 采样 %.0f s ｜ root=%s" % (a.sample, ROOT))
    print("=" * 88)

    runs, trainers = scan()
    if not runs:
        print("**没有任何 train_graph_mappo 进程在跑**")
    else:
        print("\n[1] 在跑的 run（PSS 按进程树求和；Δticks 是 %.0f s 增量）" % a.sample)
        print("    %-18s %-8s %8s %6s %9s %10s" %
              ("run", "pid", "PSS(G)", "轮数", "metrics年龄", "Δticks"))
        # 两次采样算增量
        for r in runs:
            r["t1"] = _ticks(r["pid"])
        time.sleep(a.sample)
        suspects = []
        rows = []
        for r in runs:
            t2 = _ticks(r["pid"])
            d = None if (t2 is None or r["t1"] is None) else t2 - r["t1"]
            rows.append((r, d))
            if d == 0:
                suspects.append(r)

        # ★ 单次 0 增量**不定死**：再采一次，两次都 0 才报冻结。
        #   理由见文件头「判活为什么要 40 s」的 2026-09-20 更正。
        reconfirm = {}
        if suspects:
            for r in suspects:
                r["t2"] = _ticks(r["pid"])
            time.sleep(a.sample)
            for r in suspects:
                t3 = _ticks(r["pid"])
                reconfirm[r["run"]] = None if (t3 is None or r["t2"] is None) \
                    else t3 - r["t2"]

        frozen = []
        for r, d in rows:
            n, age = rounds_and_age(r["run"])
            r["rounds"], r["age"], r["delta"] = n, age, d
            base = ROUND_S["hist"] if "hist" in r["run"] else ROUND_S["default"]
            stale = age is not None and age > base * STALE_FACTOR
            flag = ""
            if d is not None and d == 0:
                d2 = reconfirm.get(r["run"])
                if d2 == 0:
                    flag = "  ✗**冻结**（两次采样均 0 tick）"
                    frozen.append(r["run"])
                elif d2 is None:
                    flag = "  ⚠疑似冻结（二次采样进程消失）"
                    frozen.append(r["run"])
                else:
                    flag = ("  ·疑似停顿（首次 0，二次 %d tick ⟹ 已恢复，"
                            "不报冻结）" % d2)
            elif stale:
                flag = "  ⚠metrics 停滞 >%.0fs" % (base * STALE_FACTOR)
            print("    %-18s %-8d %8.1f %6d %9s %10s%s"
                  % (r["run"], r["pid"], r["pss"], n,
                     ("%.0fs" % age) if age is not None else "--",
                     d if d is not None else "进程消失", flag))
        tot = sum(r["pss"] for r in runs)
        print("    " + "-" * 80)
        print("    合计 PSS %.1f GB ｜ 在跑 %d 条" % (tot, len(runs)))

    # [2] 链在等谁、那些东西还活着吗
    waits = chain_waits()
    blocked = []
    print("\n[2] 启动链在等什么（**这是本次事故的核心：链等的是文件不是进程**）")
    if not waits:
        print("    没有解析到链日志")
    live = {r["run"] for r in runs}
    for logf, info in sorted(waits.items()):
        tag = ("·在跑 pid=%s" % info["pid"]) if info["alive"] else (
            "✓已结束" if info["finished"] else "·已不在（历史陈迹）")
        print("    --- %s  [%s] ---" % (logf, tag))
        # ★★ 两条必须分开（都在 ent001 / parallel_chain 上实测踩过）：
        #   (a) 链**不在跑** ⟹ 它不可能"永远等"（已经退出了）。
        #       日志是**历史**。不判「等不到」——否则正常跑完的三条臂
        #       被报成「进程已死」（假警，比漏报更坏）。
        #   (b) 链**在跑** 才判「它等的臂是不是死了」。
        if not info["alive"]:
            if info["finished"]:
                print("        （链已收工，以下为历史进度）")
            else:
                print("        （链已不在且无终态标记——看它当时的进度即可，"
                      "它不会再阻塞任何东西）")
            for run, u in sorted(info["waits"].items()):
                print("        %-18s u%-4d" % (run, u))
            continue

        # --- 只有**在跑的链**才进入「它在等的东西还活着吗」---
        for run, u in sorted(info["waits"].items()):
            done = u >= a.target
            alive = run in live
            mark = "✓已到" if done else ("·在跑" if alive else "✗✗**进程已死**")
            print("        %-18s u%-4d %s" % (run, u, mark))
            if not done and not alive:
                blocked.append((logf, run, u))
        if info["log_age"] > 1800:
            print("        ⚠ 链的日志已 %.0f 分钟没更新，它可能自己卡住了"
                  % (info["log_age"] / 60))
        if any(b[0] == logf for b in blocked):
            print("        ⟹ !! 这条链**永远等不到**：它等的是文件，文件不会自己长。")
            print("           处置：把死掉的臂从它自己的 checkpoint **续跑**到 u%d。" % a.target)

    # [3] 结论
    print("\n" + "=" * 88)
    rc = 0
    if runs and any(r.get("delta") == 0 for r in runs):
        rc |= 1
    if blocked:
        rc |= 2
    if rc == 0:
        print("结论：全部健康（%d 条 run 在算，%d 条链没有在等死物）" % (len(runs), len(waits)))
    else:
        if rc & 1:
            print("!! 有 run **冻结**：%s"
                  % ", ".join(r["run"] for r in runs if r.get("delta") == 0))
        if rc & 2:
            print("!! 有链在等**已经死掉的臂**：")
            for logf, run, u in blocked:
                print("     %s 等 %s（停在 u%d）" % (logf, run, u))
    print("=" * 88)
    return rc


if __name__ == "__main__":
    sys.exit(main())
