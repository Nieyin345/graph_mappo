#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""并行控制脚本：**不杀任何在跑的 run**，复用 wave5 退场腾出的槽位，
把 ent001_s{42,43,44} 与 hist_s{42,43,44} 六条**尽早**放出去。

### 为什么要另写（而不是等原链）
节点 uptime 11:55（05:18 UTC 启动），约剩 12h。原链的门是坏的，
`STEADY=25.0` 被当**整 run** 用（实际拿到的是**父进程**的 ~17G，因为
spawn worker 的 cmdline 里既无 run-name 也无脚本名）⟹ 待涨被算成 ~51G
⟹ 要等 wave5 **全部**退场、且在跑数 ≤4 才肯起。两条链串行 = **10h，零余量**。

而实测：6 个并发 run 只吃 150 GB / 48 线程（128 核），
**并发不吃速度也不改结果**（记忆 `node-parallelism-profile`、`thread-count-changes-training`：
改结果的是**线程数**，已固定在 8）⟹ **应当并行，不该串行**。

### 门（三量，按配置查表）
    可用 − Σ(稳态ᵢ − 当前PSSᵢ) − 稳态新run ≥ 17

稳态按 **`minibatch_size` 查表**（本日实测：256→25.0、512→29.5）。
量 PSS **按 PPid 向上归组**，不按 cmdline 抓父进程。

### 互斥
同一时刻只允许**一个控制脚本**在决策——两个脚本各自看门会同时通过
（记忆 `respawn-guard-two-views`）。用 `/tmp/launcher.lock` 的 `flock` 串行化。
"""
import fcntl
import json
import os
import re
import subprocess
import sys
import time

ROOT = "/opt/qkd/graph_mappo"
PY = "/opt/qkd/venv/bin/python"
CKPT = "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
BASE = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml"]
UPDATES = 30
THREADS = "8"
STEADY = {256: 25.0, 512: 29.5}   # 实测（2026-09-20）
FLOOR = 17.0
MAX_RUNS = 9

# (臂名, 配置文件名, 种子)
JOBS = [("ent001", "train_ent001.yaml", 42), ("ent001", "train_ent001.yaml", 43),
        ("ent001", "train_ent001.yaml", 44),
        ("hist", "train_hist.yaml", 42), ("hist", "train_hist.yaml", 43),
        ("hist", "train_hist.yaml", 44)]
LOG = "/tmp/parallel_chain.log"


def log(m):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def avail():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / 1e6
    return 0.0


def pss(pid):
    try:
        for ln in open("/proc/%d/smaps_rollup" % pid):
            if ln.startswith("Pss:"):
                return int(ln.split()[1]) / 1e6
    except Exception:
        pass
    return 0.0


def ppid(pid):
    try:
        s = open("/proc/%d/stat" % pid, encoding="utf-8").read()
        return int(s[s.rindex(")") + 2:].split()[1])
    except Exception:
        return 0


def cmdline(pid):
    try:
        # ★★ 必须把 NUL 换成空格：`/proc/<pid>/cmdline` 用 `\0` 分隔 argv，
        #    而 `\s` **不匹配** `\0` ⟹ 漏了这行则 `--run-name\s+(\S+)` 永不匹配
        #    ⟹ `live_runs()` 恒空 ⟹ **内存门恒通过**、无条件启动。
        #    实测（2026-09-20）：6 条 run 在跑，日志却打印「在跑 0」，
        #    于是它把 ent001×3 与 hist×3 **一起铺出去**，直接导致 hist_s42 被 OOM 杀掉。
        #    `scripts/diag/pss_per_run.py` 有这一行，所以它一直是对的——
        #    本函数是从它移植过来的，**移植时漏了**。
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            return f.read().decode("utf-8", "replace").replace("\0", " ")
    except OSError:
        return ""


def live_runs():
    """[(run_name, minibatch, 整run PSS)] —— 按 PPid 向上归组。"""
    pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    trainers = {}
    for p in pids:
        cl = cmdline(p)
        if "train_graph_mappo" not in cl:
            continue
        m = re.search(r"--run-name\s+(\S+)", cl)
        trainers[p] = m.group(1) if m else "?"
    owner = {}
    for p in pids:
        cur, seen = p, set()
        while cur > 1 and cur not in seen:
            seen.add(cur)
            if cur in trainers:
                owner[p] = cur
                break
            cur = ppid(cur)
    tot = {}
    for p in pids:
        t = owner.get(p)
        if t is not None:
            tot[t] = tot.get(t, 0.0) + pss(p)
    out = []
    for t, name in trainers.items():
        mb = 256
        try:
            for ln in open(os.path.join(ROOT, "outputs", name, "resolved_config.yaml"),
                           encoding="utf-8"):
                m = re.match(r"\s*minibatch_size:\s*(\d+)", ln)
                if m:
                    mb = int(m.group(1))
                    break
        except OSError:
            pass
        out.append((name, mb, tot.get(t, 0.0)))
    return out


def done(name):
    m = os.path.join(ROOT, "outputs", name, "metrics.jsonl")
    if not os.path.exists(m):
        return False
    last = 0
    try:
        for ln in open(m, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                o = json.loads(ln)
                if o.get("update"):
                    last = o["update"]
    except Exception:
        return False
    return last >= UPDATES


def launch(arm, cfg, seed):
    run = "%s_s%d" % (arm, seed)
    cmd = ([PY, "scripts/train/train_graph_mappo.py", "--configs"] + BASE + [cfg] +
           ["--checkpoint", CKPT, "--seed", str(seed),
            "--num-updates", str(UPDATES), "--run-name", run])
    env = dict(os.environ, OMP_NUM_THREADS=THREADS, PYTHONUNBUFFERED="1")
    f = open("/tmp/%s.out" % run, "ab")
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    log("  ★ 起 %s (pid=%d)" % (run, p.pid))
    return p


# ============================ 主流程 ============================
def main():
    lock = open("/tmp/launcher.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("另一个控制脚本持有锁，退出")
        return
    log("=" * 70)
    log("并行链启动（本脚本**不杀任何在跑的 run**）")

    # 前置条件**按臂分别判定**：
    #   · ent001 是**纯配置改动**（只改 entropy_coef）⟹ 不依赖任何代码改动，可立刻起
    #   · hist 需要 edge_h 修复（否则崩）+ 加载器（否则丢暖启动）
    # ⚠ `hist_ok` 必须在**循环里每次重算**：代码要等 wave5 退场后才好同步，
    #   若只在启动时算一次，同步完 hist 也永远不会被起（本项目踩过同类哑火）。
    tr = open(os.path.join(ROOT, "qkd_rl/rl/algos/mappo_trainer.py"), encoding="utf-8").read()
    gm = open(os.path.join(ROOT, "qkd_rl/rl/models/graph_mappo.py"), encoding="utf-8").read()
    if not os.path.exists(os.path.join(ROOT, CKPT)):
        log("!! BC 起点不存在，什么都不起")
        return
    log("ent001 就绪（纯配置）。hist 前置条件当前=%s"
        % ("满足" if ("_upgrade_state_dict_for_model" in tr
                      and "edge_h 只在 h_rows 非空时拼接" in gm) else "**未满足**"))

    launched = set()
    t0 = time.time()
    hist_started = False
    while len(launched) < len(JOBS) and time.time() - t0 < 10 * 3600:
        # 每次重算：代码同步后 hist 自动放行
        try:
            tr = open(os.path.join(ROOT, "qkd_rl/rl/algos/mappo_trainer.py"),
                      encoding="utf-8").read()
            gm = open(os.path.join(ROOT, "qkd_rl/rl/models/graph_mappo.py"),
                      encoding="utf-8").read()
        except OSError:
            pass
        hist_ok = ("_upgrade_state_dict_for_model" in tr
                   and "edge_h 只在 h_rows 非空时拼接" in gm)
        runs = live_runs()
        n = len(runs)
        a = avail()
        grow = sum(max(0.0, STEADY.get(mb, 25.0) - cur) for _, mb, cur in runs)
        margin = a - grow - STEADY[256]
        if n < MAX_RUNS and margin >= FLOOR:
            for arm, cfg, seed in JOBS:
                if (arm, seed) in launched:
                    continue
                if arm == "hist" and not hist_ok:
                    continue          # hist 代码没同步好，跳过（继续起 ent001）
                launched.add((arm, seed))
                if arm == "hist" and not hist_started:
                    hist_started = True
                    log("★ 开始起 hist（说明代码已同步）")
                launch(arm, cfg, seed)
                time.sleep(15)
                break
        else:
            if int(time.time() - t0) % 600 < 140:
                log("  门未过：可用 %.0f 待涨 %.1f 余量 %.1f（在跑 %d）"
                    % (a, grow, margin, n))
        time.sleep(60)

    log("已起 %d/%d   （hist 前置条件满足=%s）" % (len(launched), len(JOBS), hist_ok))

    # 等扫尾
    todo = ["%s_s%d" % (arm, seed) for arm, _, seed in JOBS]
    while time.time() - t0 < 12 * 3600:
        nd = sum(1 for r in todo if done(r))
        if nd == len(todo):
            log("✓ 6 臂全部完成")
            break
        time.sleep(150)
    log("控制脚本退出（未完成的臂仍在服务器上跑，结果照常落在 outputs/）")


if __name__ == "__main__":
    main()
