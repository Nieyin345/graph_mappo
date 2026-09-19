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

退出码：`rc` 是**位掩码**，另加两个**整值**的第三态：
    0 = 全健康
    1 = 有 run 冻结（或**进程在采样窗内消失**）
    2 = 有**在跑的**链在等已经死掉的臂
    3 = 上两者都有（1|2）
    ★ 3 同时也是「**没有可判读的对象**」的专用码 —— 见下。

★ 退出码 3 的**二义**是刻意保留的（2026-09-21）：`1|2` 与「0 条 run」
在数值上撞车，但两者的**结论行不同**（一个说"有冻结/有链卡住"，
一个说"无法判定"），而 `0 条 run` 在实操里只会出现在**停训期间**，
不会与"既冻结又有链卡住"同时发生。真要消除二义得改成 4，但那会让
"位掩码"这条更值钱的约定失效。**先记下这个取舍，别当成没想到。**

★ 第三态为什么必要：原实现在 0 条 run 时走 `if runs and any(...)` 的短路，
⟹ `rc = 0` ⟹ 打印「结论：全部健康（**0 条 run 在算**）」。**那是自相矛盾**：
没在算的东西不可能"健康"。而在停训期间这恰恰是**最常见**的状态，
于是一个本该喊"我什么都没看到"的工具，天天报"一切正常"。

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
    """(轮号, metrics.jsonl 年龄秒)。**轮号 = 最后一个 `"update": N` 的值。**

    ★★ 本函数原先数的是「带 `update` 键的**行数**」，注释还写着「与链的
    `last_update()` 同一口径」—— 而两件事**都错了**：

      1. `metrics.jsonl` **每轮不止一行**：`eval_validation` 行也占一行，且它
         **没有 `update` 键**（`eval_interval=5` ⟹ 30 轮是 36 行）。数行数会
         系统性多报 `轮数 // eval_interval`。实测：`ent01_rerun_s43` 报 34、
         真值 30。
      2. 链侧 `hist32_chain.last_update()` 当时**也**在数行数，所以「同一口径」
         这句在当时成立，但**两边一起错**。链侧已改为按末个 `"update"` 读。

    这里返回的轮号**直接驱动 `u >= target` 的完成判据**（`:332`），所以偏快
    ⟹ 在本工具里报「✓ 已到 u30」，而链还在等 6 轮。

    这是本项目在「拿位置/计数冒充身份」上的**第四次**复发（前三次：
    按行号当轮号、按累计行数、`launch_g2.updates_done` 数行数）。
    记忆 `eval-update-number-not-from-position`。
    """
    p = os.path.join(ROOT, "outputs", run, "metrics.jsonl")
    n, age = -1, None
    try:
        age = time.time() - os.path.getmtime(p)
    except OSError:
        return -1, None
    n, last = -1, None          # last = 轮号；n 保留为「文件是否读得出」的信号
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or '"update"' not in ln:
                    continue
                try:
                    u = json.loads(ln)["update"]     # 没有该键的（eval 行）会 KeyError
                except (ValueError, KeyError, TypeError):
                    continue                          # 半行 / eval 行：跳过，不猜
                try:
                    last = int(u)
                except (TypeError, ValueError):
                    continue
    except OSError:
        pass
    return (last if last is not None else (-1 if not os.path.exists(p) else 0)), age


def classify(delta, delta2=None, stale_after=None):
    """把「一次采样 + 一次复采」映射成 `(kind, flag)`。**纯函数，可造反证。**

    `kind ∈ {None, "frozen", "vanished"}`；`flag` 是给人看的后缀。

    ★★ 为什么把它从打印循环里**拎出来**（2026-09-21）：
    这条判据原先是**内联在打印循环里的一条 if/elif 链**，而 `rc` 又在
    结论段**另算一遍**（`any(r.get("delta") == 0)`，用的是**首次**采样）。
    于是「明细行说了什么」与「退出码怎么定」**各自漂移**，实测就漂了：
    明细打印「疑似停顿（二次已恢复，**不报冻结**）」，结论却判有冻结、exit 1。

    **判据一旦只能靠人读源码来确认，就一定会与它自己的打印分家。**
    拎成纯函数之后，「判据」这件事有了**唯一的实现**（同族记忆：
    `cross-check-must-compare-same-population` —— 口径只能有一个）。

    参数语义（顺序要紧 —— `delta is None` 优先于 `delta == 0`）：
      delta is None                ⟹ 进程在采样窗内消失（最严重，原实现漏报）
      delta == 0 且 delta2 is None ⟹ 进程在**复采**窗口消失
      delta == 0 且 delta2 == 0    ⟹ 冻结（两次连续 0 tick）
      delta == 0 且 delta2  > 0    ⟹ 疑似停顿但已恢复 ⟹ **不报**
      delta  >  0 且 stale_after   ⟹ 在算，但 metrics 不落盘（另一种故障）
    """
    if delta is None:
        return "vanished", "  ✗**进程消失**（采样窗内）"
    if delta == 0:
        if delta2 == 0:
            return "frozen", "  ✗**冻结**（两次采样均 0 tick）"
        if delta2 is None:
            return "vanished", "  ✗**进程在二次采样窗口消失**"
        return None, ("  ·疑似停顿（首次 0，二次 %d tick ⟹ 已恢复，"
                      "不报冻结）" % delta2)
    if stale_after is not None:
        return None, "  ⚠metrics 停滞 >%.0fs（进程在算，但结果没落盘）" % stale_after
    return None, ""


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
        vanished = []
        for r, d in rows:
            # ★ 用 scan() 早先量到的轮号/年龄；**不要**在这里再读一次文件
            #   （原实现每行调一次 `rounds_and_age`，与 scan() 那次读的
            #   可能不是同一个瞬间 —— 又一次"同一个量两个实现"）。
            n, age = r["rounds"], r["age"]
            base = ROUND_S["hist"] if "hist" in r["run"] else ROUND_S["default"]
            stale_after = base * STALE_FACTOR if (age or 0) > base * STALE_FACTOR \
                else None
            kind, flag = classify(d, reconfirm.get(r["run"]), stale_after)
            r["kind"], r["flag"] = kind, flag
            if kind == "frozen":
                frozen.append(r["run"])
            elif kind == "vanished":
                vanished.append(r["run"])
            r["frozen"] = kind == "frozen"
            r["vanished"] = kind == "vanished"
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
    # ★★ 判据一律以 `frozen` / `vanished` 为准，**不用 `delta == 0`**。
    #   原实现用 `any(r.get("delta") == 0 ...)` 判冻结，而 `delta` 是**首次**
    #   采样的结果 ⟹ 与明细行自相矛盾：明细写「疑似停顿（二次已恢复，不报冻结）」
    #   而结论行写「有 run 冻结」、exit 1。**打印修了、判据没修**
    #   （记忆 `later-sections-retract-earlier-ones`：别信"已修"，要求证代码）。
    dead = [r["run"] for r in runs if r.get("frozen") or r.get("vanished")]
    if dead:
        rc |= 1
    if blocked:
        rc |= 2
    if not runs:
        # ★ 「一条 run 都没有」不许说"健康" —— 那是自相矛盾（原实现会打印
        #   「结论：全部健康（0 条 run 在算）」并 exit 0）。也不能报故障：
        #   可能只是链还没放臂。所以是**第三态**：明确说"没有可判的对象"。
        print("结论：**没有可判读的对象**（0 条 run 在跑）。"
              "这既不是健康也不是故障 —— 无法判定。")
        if blocked:
            print("  但已有 %d 条链在等（见上），它们等的臂**当前不在跑**。" % len(waits))
        print("=" * 88)
        return 3
    if rc == 0:
        print("结论：全部健康（%d 条 run 在算，%d 条链没有在等死物）" % (len(runs), len(waits)))
    else:
        if rc & 1:
            fz = [r["run"] for r in runs if r.get("frozen")]
            vz = [r["run"] for r in runs if r.get("vanished")]
            if fz:
                print("!! 有 run **冻结**（两次采样均 0 CPU tick）：%s" % ", ".join(fz))
            if vz:
                print("!! 有 run **进程消失**（采样窗内退出）：%s" % ", ".join(vz))
        if rc & 2:
            print("!! 有链在等**已经死掉的臂**：")
            for logf, run, u in blocked:
                print("     %s 等 %s（停在 u%d）" % (logf, run, u))
    print("=" * 88)
    return rc


if __name__ == "__main__":
    sys.exit(main())
