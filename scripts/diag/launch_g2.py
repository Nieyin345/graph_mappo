#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开波器 g2 —— gae90 配对 A/B + 跑到 u50 的延长。

## 为什么存在这个文件（而不是继续用 /tmp/launch_wave263.py）

`launch_wave263.py` 是**在 `.tmp/` 里手搓、从未进版本库**的（记忆
`plan-is-a-claim-about-the-world-recheck-at-launch` 记过这个毛病）。它的**内存门
是对的**，是被事故换来的（NUL 分隔、进程树求和、待涨量、打印输入、断言非空）。
本文件把那套门**原样搬进版本库**并泛化到多族臂，不再每次重写。

## ★★ 相对 wave263 的唯一实质改动：**逐条放，不是整批等**

wave263 的 driver 是**一条一条**放的：每放一条就 `break` 回去**重读内存**再决策。
而我的第一版 g2 链是"整批等 142G"——**这是错的模式**，实测代价：

  - T+108 时 driver 退出，此刻 MemAvailable ≈ 118G
  - 整批门（5×25+17 = 142G）**过不去** ⟹ 要等到 T+123 另一条 hist 跑完
  - 逐条门（25+17 = 42G）在 T+108 **立刻能放 4 条**

⟹ 逐条放**不需要更大的内存**，只是不会把"一次放得下几条"这种事算错。
   `pending_growth()`（待涨量）是逐条门能成立的关键：新起的臂 PSS 只有 ~9G，
   没有待涨量就会以为还有富余，一路放到 OOM。

## 三条服从（不是我想选的，是被在跑的波定死的）

1. **臂名**：对照是 `ent01_rerun_s*`（第一代名），不是 `_v2` —— 必须与 wave263
   的对照同族才能合并成 n=5。已核对 `CTRL_CFGS` 与本文件 `CTRL_CFGS` 逐字相同。
2. **线程数 8**：`launch_wave263.py` 的 `THREADS = "8"`。线程数**确定性**改变
   训练结果（差 0.018，记忆 `thread-count-changes-training`）⟹ 必须同为 8。
3. **同一时刻只有一个启动器**：phase A 先**等 wave263 的 driver 进程退出**
   （无歧义、无竞态，记忆 `plan-is-a-claim-...`），再接棒。**不抢、不杀它**。

## 预注册（跑之前写死，防事后编故事）

**待检验的唯一机制**：λ=0.95 → 0.90（更少依赖 bootstrap）。
旧节点上：窗口 u25/u30、n=5、Δ=+0.0146、t=3.61、df=4、临界 2.776 ⟹ **过线**；
**但换窗口就翻转**（单轮 u5 是 −0.0244、t=−3.23；全程 u5–u30 只有 +0.0005）。

⟹ 真问题不是"有没有效应"，而是：**那个正号是稳定平台，还是窗口挑出来的涨落？**

判据（phase B/C 才用得上，现在写死）：
  - u35–u50 仍给正号 ⟹ **不是「多训就涨」** ⟹ 支持「稳定平台」
  - u35–u50 正号消失 ⟹ 就是窗口效应 ⟹ **gae90 这条线关闭**
  - 观测单位 = **训练种子**（n=5，df=4）；p 与临界值一律**现算**，不手抄

用法（服务器上）：
    setsid nohup /opt/qkd/venv/bin/python -u /tmp/launch_g2.py A \\
        > /tmp/g2_A.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

ROOT = "/opt/qkd/graph_mappo"
PY = "/opt/qkd/venv/bin/python"
BC_CKPT = "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"

CTRL_CFGS = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml"]
GAE_CFGS = CTRL_CFGS + ["train_gae90.yaml"]
HIST_CFGS = CTRL_CFGS + ["train_hist.yaml"]

SEEDS = ["42", "43", "44", "45", "46"]
NEW_CTRL_SEEDS = ["45", "46"]        # wave263 只放了 42/43/44 的对照
THREADS = "8"                        # ★ 必须与 launch_wave263.py 一致

UPDATES = 30                         # 从头训
UPDATES_EXT = 20                     # 续跑 u30 → u50

# 内存模型（实测）
PSS_BASE = 25.0
PSS_HIST = 63.0
FLOOR = 17.0
MAX_HIST = 2                         # hist 系同时最多 2 条（实测约束，不是偏好）

# 等 wave263 的 driver 退出（phase A 用）。**字符类**避开自匹配。
DRIVER_RE = re.compile(r"launch_wave263\.py")


def log(m: str) -> None:
    print("[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), m),
          flush=True)


# ---------------------------------------------------------------- 内存读数
def cmdline(pid: int) -> str:
    """★ `.replace("\\0", " ")` 是**必须**的：`/proc/<pid>/cmdline` 用 NUL 分隔
    argv，而 `\\s` **不匹配** `\\x00` ⟹ 正则永远匹配不上 ⟹ 门**恒通过**、
    无条件启动、直到 OOM 才现形（记忆 `gate-must-print-its-inputs`）。"""
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            return f.read().decode("utf-8", "replace").replace("\0", " ")
    except OSError:
        return ""


def ppid(pid: int) -> int:
    try:
        for ln in open("/proc/%d/status" % pid, encoding="utf-8", errors="replace"):
            if ln.startswith("PPid:"):
                return int(ln.split()[1])
    except OSError:
        pass
    return 0


def pss(pid: int) -> float:
    """该进程的 PSS（GB）。含私有匿名页 + 共享页的公平分摊；**不是 RSS**
    （RSS 会把 8 个 spawn worker 共享的 torch 代码页重复计数）。"""
    tot = 0
    try:
        with open("/proc/%d/smaps_rollup" % pid, encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("Pss:"):
                    tot += int(ln.split()[1]) * 1024
    except OSError:
        return 0.0
    return tot / 1e9


def _pids() -> list[int]:
    return [int(p) for p in os.listdir("/proc") if p.isdigit()]


def live_runs() -> list[tuple[str, str, float]]:
    """[(run_name, kind, 整 run PSS GB)] —— **按进程树**求和，不只抓父进程。
    spawn worker 的 cmdline 里**没有** run-name，只看父进程会少算约 25%。"""
    trainers: dict[int, str] = {}
    for p in _pids():
        cl = cmdline(p)
        if "train_graph_mappo" not in cl:
            continue
        m = re.search(r"--run-name\s+(\S+)", cl)
        if m:
            trainers[p] = m.group(1)
    owner: dict[int, int] = {}
    for p in _pids():
        cur, seen = p, set()
        while cur > 1 and cur not in seen:
            seen.add(cur)
            if cur in trainers:
                owner[p] = cur
                break
            cur = ppid(cur)
    tot: dict[int, float] = {}
    for p in _pids():
        t = owner.get(p)
        if t is not None:
            tot[t] = tot.get(t, 0.0) + pss(p)
    return [(name, kind_of(name), tot.get(t, 0.0)) for t, name in trainers.items()]


def kind_of(name: str) -> str:
    return "hist" if "hist" in name else "base"


def avail_gb() -> float:
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def updates_done(name: str) -> int:
    """已完成的 update 数 —— metrics.jsonl 每轮一行。
    ⚠ 续跑臂的 outputs 是**新目录**（`_u30to50`），行数从 0 起，所以这里量的
    是"本段跑了多少轮"，用于待涨量估计是安全的（前 15 轮一律按未到稳态算）。"""
    p = os.path.join(ROOT, "outputs", name, "metrics.jsonl")
    try:
        with open(p, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def steady_of(kind: str) -> float:
    return PSS_HIST if kind == "hist" else PSS_BASE


def pending_growth(runs) -> float:
    """在跑的 run **还没涨到的量** —— 这一项是隐形的，不算就等于用启动时的
    小读数去排满载（记忆 `mem-gate-must-count-warmup-growth`）。

    ★ 第一版有个 `if updates_done(name) >= 15: continue` 的截断（理由：拟合是
    +0.21 GB/轮，到 u15 已接近稳态）。**这个截断是错的**：实测 hist 臂在 u24
    时仍然只有 57–59G（模型值 63G）⟹ 截断会**丢掉这 10G 的待涨量**，
    把它从门的输入里抹掉。改成对**所有**在跑的臂取 `max(0, steady − cur)`，
    不设截断 —— 简单、且严格更保守。"""
    pend = 0.0
    for name, kind, cur in runs:
        tgt = steady_of(kind)
        if cur < tgt:
            pend += (tgt - cur)
    return pend


def memtotal_gb() -> float:
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemTotal:"):
                return int(ln.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def memfree_gb() -> float:
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemFree:"):
                return int(ln.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def two_views(runs, kind_new: str) -> tuple[float, float, float, float]:
    """**两个视角，缺一不可**（记忆 `respawn-guard-two-views`）：
    只算增量会在稳态总和已经超过 `MemTotal` 时**放行**；只算绝对又会把
    「够用」报成「不够用」。返回 (增量余量, 绝对余量, 待涨量, 其它占用)。

      · 视角 A（增量）：可用 − Σ待涨 − 本臂稳态 ≥ FLOOR
        问的是"现在起得起吗"
      · 视角 B（绝对）：MemTotal − 其它占用 − Σ**全部**稳态（含本臂）≥ FLOOR
        问的是"涨到稳态后系统还剩多少" ⟸ **这道墙 `load average` 和
        `%CPU` 都看不见**，正是「照启动时的 23GB 排 5 个、涨到 u15 就顶格」的病根
    """
    avail = avail_gb()
    pend = pending_growth(runs)
    inc = avail - pend - steady_of(kind_new)
    other = memtotal_gb() - memfree_gb() - sum(p for _n, _k, p in runs)
    total_steady = sum(steady_of(k) for _n, k, _p in runs) + steady_of(kind_new)
    absolute = memtotal_gb() - other - total_steady
    return inc, absolute, pend, other


# ---------------------------------------------------------------- 臂表
def arms_for(phase: str) -> list[tuple[str, list[str], str, int, str | None]]:
    """[(run_name, cfgs, kind, num_updates, resume_ckpt_or_None)] —— **按优先级排序**。

    顺序即优先级（逐条放，前面的先占内存）：
      hist32_s44 先 —— 它完成 wave263（Task #1），且它是唯一被 MAX_HIST 卡的
      然后 gae90 ×5（Task #3 的预注册检验）
      最后补对照 s45/s46（把对照族凑到 n=5，配对分析要用）
    """
    out = []
    if phase == "A":
        out.append(("hist32_s44", HIST_CFGS, "hist", UPDATES, None))
    if phase in ("A", "B"):
        for s in SEEDS:
            if phase == "B":
                parent = "gae90_s%s" % s
                ck = "outputs/%s/checkpoint_update_%06d.pt" % (parent, UPDATES)
                out.append(("%s_u30to50" % parent, GAE_CFGS, "base", UPDATES_EXT, ck))
            else:
                out.append(("gae90_s%s" % s, GAE_CFGS, "base", UPDATES, None))
    if phase in ("A", "C"):
        for s in (SEEDS if phase == "C" else NEW_CTRL_SEEDS):
            if phase == "C":
                parent = "ent01_rerun_s%s" % s
                ck = "outputs/%s/checkpoint_update_%06d.pt" % (parent, UPDATES)
                out.append(("%s_u30to50" % parent, CTRL_CFGS, "base", UPDATES_EXT, ck))
            else:
                out.append(("ent01_rerun_s%s" % s, CTRL_CFGS, "base", UPDATES, None))
    return out


# ---------------------------------------------------------------- 预检
def preflight(arms) -> bool:
    """**起任何一条之前，全员预检**（记忆 `failed-launch-must-be-loud`）：
    缺一个就**一条都不起**。否则失败会被静默成"跳过"，整队空等到超时。"""
    log("== 预检（全员；任一项缺失则整波不启动）==")
    bad = []
    if not os.path.exists(os.path.join(ROOT, BC_CKPT)):
        bad.append("BC 起点缺失: %s" % BC_CKPT)
    for cfg in sorted({c for _n, cfgs, _k, _u, _r in arms for c in cfgs}):
        if "/" in cfg:
            bad.append("--configs 要**裸文件名**，收到 %r（build_config 会自己拼 configs/）" % cfg)
        elif not os.path.exists(os.path.join(ROOT, "configs", cfg)):
            bad.append("配置缺失: configs/%s" % cfg)
    for name, _cfgs, _kind, _u, resume in arms:
        if resume and not os.path.exists(os.path.join(ROOT, resume)):
            bad.append("父臂检查点缺失（续跑 %s 要用）: %s" % (name, resume))
        d = os.path.join(ROOT, "outputs", name)
        if os.path.isdir(d):
            log("  · %s 的 outputs/ 已存在 ⟹ 将跳过（视为已完成/在跑）" % name)
    if bad:
        log("  ✗ 预检**失败**，一条都不启动：")
        for b in bad:
            log("      - %s" % b)
        return False
    log("  ✓ 预检通过（checkpoint、配置、续跑父臂全部在位）")
    return True


def alive(name: str) -> bool:
    pat = re.compile(r"--run-name\s+%s(\s|$)" % re.escape(name))
    for p in _pids():
        if pat.search(cmdline(p)):
            return True
    return False


def launch(name, cfgs, kind, nupd, resume) -> int:
    ck = resume or BC_CKPT
    cmd = ([PY, "-u", "scripts/train/train_graph_mappo.py", "--configs"] + cfgs
           + ["--checkpoint", ck,
              "--seed", name.split("_s")[-1].split("_")[0],
              "--num-updates", str(nupd), "--run-name", name])
    env = dict(os.environ, OMP_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS)
    logf = open("/tmp/%s.log" % name, "ab")
    pr = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf,
                          stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                          start_new_session=True)
    log("  已起 %s（%s，稳态 %.0f GB，%d 轮，起点 %s）"
        % (name, kind, steady_of(kind), nupd, os.path.basename(ck)))
    return pr.pid


# ---------------------------------------------------------------- 等 driver
def wait_driver() -> bool:
    """等 wave263 的 driver **进程退出**。无歧义、无竞态（记忆
    `plan-is-a-claim-about-the-world-recheck-at-launch`）。
    ⚠ 不能用 `pkill -f` —— 在 `ssh host '...'` 里会杀掉自己。"""
    def running() -> list[int]:
        return [p for p in _pids() if DRIVER_RE.search(cmdline(p))]

    if not running():
        log("wave263 的 driver 已不在（可能早已退出）")
        return True
    log("等 wave263 的 driver 退出（它卡在 hist 门 2<2，还要放 hist32_s44）…")
    for _ in range(720):                      # 最多等 6h
        if not running():
            log("driver 已退出 ✓ —— 本脚本现在是唯一启动器")
            return True
        time.sleep(30)
    log("⚠ 等 driver 超时（6h），它还在跑 —— **不启动**（宁可漏跑，不冒险 OOM）")
    return False


# ---------------------------------------------------------------- 主循环
def main(argv: list[str]) -> int:
    phase = (argv[1] if len(argv) > 1 else "").upper()
    if phase not in ("A", "B", "C"):
        log("!! 用法: %s A|B|C" % argv[0])
        return 2
    go = "/tmp/g2_%s.go" % phase
    if os.path.exists(go):
        log("已有标记 %s，跳过" % go)
        return 0

    arms = arms_for(phase)
    log("=" * 70)
    log("开波 g2 phase %s ｜ 目标 %d 条臂 ｜ 内存模型 base %.0f / hist %.0f ｜ "
        "FLOOR %.0f ｜ MAX_HIST %d ｜ %s 线程"
        % (phase, len(arms), PSS_BASE, PSS_HIST, FLOOR, MAX_HIST, THREADS))
    if not preflight(arms):
        return 1

    if phase == "A" and not wait_driver():
        return 1

    # ★★ 待起清单必须**在等完 driver 之后**才算 —— 不是之前。
    #   实测教训：`hist32_s44` 在启动时还没被 driver 放出来（所以它在 todo 里），
    #   等 driver 退出时，driver **已经把它起了**。若拿等之前的快照去起，就会
    #   **两份进程写同一个 outputs/** ⟹ 读数作废（记忆
    #   `chain-must-ask-proc-not-its-own-ledger`：去重问世界，不问账本；
    #   `plan-is-a-claim-about-the-world-recheck-at-launch`：计划是对世界的断言，
    #   启动那一刻必须重读）。
    todo = [a for a in arms
            if not (os.path.isdir(os.path.join(ROOT, "outputs", a[0]))
                    or alive(a[0]))]
    for a in arms:
        if a not in todo:
            log("  跳过 %s（已完成或在跑）" % a[0])
    if not todo:
        log("无待起臂。")
        open(go, "w").write("noop\n")
        return 0

    # ---------- 逐条放：**每放一条就重读内存**（照抄 wave263 的 driver）----------
    started: list[tuple[str, int]] = []
    while todo:
        runs = live_runs()
        n_hist = sum(1 for _n, k, _p in runs if k == "hist")

        # ★ 断言：有训练进程却数出 0 条 ⟹ 门的输入是坏的 ⟹ 立即停。
        #   这正是「恒真的门不报错」要防的事（记忆 gate-must-print-its-inputs）。
        n_train = sum(1 for p in _pids() if "train_graph_mappo" in cmdline(p))
        if n_train and not runs:
            log("✗ 检测到 %d 个训练进程但 live_runs() 返回空 ⟹ 门的输入已坏，停止。"
                % n_train)
            return 1

        progressed = False
        for i, (name, cfgs, kind, nupd, resume) in enumerate(list(todo)):
            avail = avail_gb()
            pend = pending_growth(runs)
            inc, absolute, _p, other = two_views(runs, kind)
            hist_ok = (kind != "hist") or (n_hist < MAX_HIST)
            ok = (inc >= FLOOR) and (absolute >= FLOOR) and hist_ok

            log("-" * 70)
            log("在跑 %d 条：%s" % (len(runs),
                "、".join("%s(%.1fG,u=%d)" % (n, p, updates_done(n))
                          for n, _k, p in runs) or "无"))
            log("  MemTotal %.1f ｜ MemFree %.1f ｜ MemAvailable %.1f ｜ 其它占用 %.1f"
                % (memtotal_gb(), memfree_gb(), avail, other))
            log("  待涨量 %.1f GB（在跑的还没涨到稳态的部分，**隐形**）" % pend)
            log("  候选 %s（%s，%d 轮，稳态 %.0fG）：" % (name, kind, nupd, steady_of(kind)))
            log("    视角A·增量：可用 %.1f − 待涨 %.1f − 本臂 %.0f = %.1f ≥ %.0f ? %s"
                % (avail, pend, steady_of(kind), inc, FLOOR, inc >= FLOOR))
            log("    视角B·绝对：MemTotal − 其它 − Σ**全部**稳态 = %.1f ≥ %.0f ? %s"
                % (absolute, FLOOR, absolute >= FLOOR))
            log("    hist 门：%d < %d ? %s" % (n_hist, MAX_HIST, hist_ok))
            log("    ⟹ **%s**" % ("放行" if ok else "等"))
            if not ok:
                continue
            pid = launch(name, cfgs, kind, nupd, resume)
            time.sleep(20)
            if alive(name):
                log("    ✓ %s 存活 (pid %d)" % (name, pid))
            else:
                log("    ✗ %s 即死。日志尾部：" % name)
                try:
                    with open("/tmp/%s.log" % name, encoding="utf-8",
                              errors="replace") as f:
                        for ln in f.read().splitlines()[-12:]:
                            log("        | %s" % ln)
                except OSError:
                    pass
                log("    ⟹ 停止（不重试；失败必须吵，记忆 failed-launch-must-be-loud）")
                return 1
            started.append((name, pid))
            todo.pop(i)
            progressed = True
            break               # 起一条就重新读数（内存窗口会变）

        if not progressed:
            time.sleep(60)

    log("=" * 70)
    log("全部 %d 条已起：%s" % (len(started), "、".join(n for n, _p in started)))

    # ---------- 启动后验证：**进程在 ≠ 跑对了** ----------
    time.sleep(150)
    log("=== 启动后验证（resolved_config.yaml 里的生效值）===")
    bad = 0
    for name, _pid in started:
        rc = os.path.join(ROOT, "outputs", name, "resolved_config.yaml")
        if not os.path.exists(rc):
            log("  ?? %s: 还没有 resolved_config.yaml（进程可能已死）" % name)
            bad = 1
            continue
        got = subprocess.run([PY, "/tmp/rc_get.py", rc,
                              "train.gae_lambda", "train.ppo.entropy_coef"],
                             capture_output=True, text=True).stdout.strip()
        log("  %s: %s" % (name, got.replace("\n", " ")))
        want = "0.9" if name.startswith("gae90") else None
        if want and ("train.gae_lambda=%s" % want) not in got:
            log("       ✗ 期望 train.gae_lambda=%s —— **结果不可用**" % want)
            bad = 1
        if not name.startswith("gae90") and "train.gae_lambda=0.95" not in got:
            log("       ✗ 期望 train.gae_lambda=0.95 —— **结果不可用**")
            bad = 1
        if "train.ppo.entropy_coef=0.01" not in got:
            log("       ✗ 期望 train.ppo.entropy_coef=0.01 —— **结果不可用**")
            bad = 1
    if bad:
        log("!! 有臂异常 —— **不写标记**，修好后可重跑")
        return 1
    log("启动后验证全部通过 ✓")

    # ---------- 等跑完 ----------
    TMO = 10 * 3600
    log("等本波臂跑完（最长 %dh）…" % (TMO // 3600))
    waited = 0
    while True:
        n_alive = sum(1 for n, _p in started if alive(n))
        if n_alive == 0:
            log("✓ 全部臂已退出（等了 %ds）" % waited)
            break
        if waited >= TMO:
            log("✗ 超时：仍有 %d 条存活 —— **不写标记**" % n_alive)
            return 1
        if waited % 600 == 0:
            log("  还在跑 %d 条（已等 %d 分钟）" % (n_alive, waited // 60))
        time.sleep(60)
        waited += 60

    open(go, "w").write("done %d arms\n" % len(started))
    log("=" * 70)
    log("=== phase %s 结束（标记 %s）===" % (phase, go))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
