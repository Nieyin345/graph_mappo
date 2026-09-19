#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开波器 g2 —— gae90 配对 A/B + 跑到 u50 的延长。

## 为什么存在这个文件（而不是继续用 /tmp/launch_wave263.py）

`launch_wave263.py` 是**在 `.tmp/` 里手搓、从未进版本库**的（记忆
`plan-is-a-claim-about-the-world-recheck-at-launch` 记过这个毛病）。它的**内存门
是对的**，是被事故换来的（NUL 分隔、进程树求和、待涨量、打印输入、断言非空）。
本文件把那套门**原样搬进版本库**并泛化到多族臂，不再每次重写。

## ★★ 两处相对 wave263 的实质改动

wave263 的 driver 是**一条一条**放的：每放一条就 `break` 回去**重读内存**再决策。
而我的第一版 g2 链是"整批等 142G"——**这是错的模式**，实测代价：

  - T+108 时 driver 退出，此刻 MemAvailable ≈ 118G
  - 整批门（5×25+17 = 142G）**过不去** ⟹ 要等到 T+123 另一条 hist 跑完
  - 逐条门（25+17 = 42G）在 T+108 **立刻能放**（实测 2 条；第 3 条余量转负）
    ⟹ 不是"能放更多"，是"不用白等 15 分钟才放第一条"

⟹ 逐条放**不需要更大的内存**，只是不会把"一次放得下几条"这种事算错。
   `pending_growth()`（待涨量）是逐条门能成立的关键：新起的臂 PSS 只有 ~9G，
   没有待涨量就会以为还有富余，一路放到 OOM。

**改动 2：纠正一个**静默偏松**的单位错。** 原版（含 wave263）把 `/proc/meminfo`
当十进制 GB 读（÷1e6），而 `PSS_BASE/PSS_HIST` 是从 `mem_pss.py` 按 **GiB**
（÷1024÷1024）量出来的 ⟹ 同一台机器两个 `MemTotal`（251.2 GiB vs 263.4 GB），
稳态被系统性低估 1.8（base）~4.6（hist）GiB/run。现场 5 条 + 1 新臂时累计
低估约 11 GiB，而 `FLOOR` 只有 17 ⟹ **安全边际被吃掉 63%**。详见下面常量处的
注释；自检见 `scripts/diag/check_mem_units.py`。

## 三条服从（不是我想选的，是被在跑的波定死的）

1. **臂名**：对照是 `ent01_rerun_s*`（第一代名），不是 `_v2` —— 必须与 wave263
   的对照同族才能合并成 n=5。已核对 `CTRL_CFGS` 与本文件 `CTRL_CFGS` 逐字相同。
2. **线程数 8**：`launch_wave263.py` 的 `THREADS = "8"`。线程数**确定性**改变
   训练结果（差 0.018，记忆 `thread-count-changes-training`）⟹ 必须同为 8。
3. **同一时刻只有一个启动器** ⟹ 与 driver **划界**：本波只拿它不碰的臂。
   ★ 第一版写的是"**等 driver 进程退出**"，实测**被推翻**：driver 卡在
   hist 门 `2<2`，要 ~108 分钟才放下 `hist32_s44` 再退出，而本波的门**当时
   就过得了** ⟹ 白等 108 分钟、机器闲置 40–70 GiB，违反"把电脑性能吃满"。
   ⟹ 改成划界（集合不相交，可直接查验），并把 `hist32_s44` 让给 driver。

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

import json
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

# ★★ 单位：**全程 GiB**（2^30 / 2^20），不是十进制 GB。
#    实测教训：`/proc/meminfo` 的 "kB" 其实是 **KiB**。`mem_pss.py` 用
#    ÷1024÷1024（GiB）读数，而 `PSS_BASE/PSS_HIST` 正是从它量出来的；
#    本文件第一版却用 ÷1e6/÷1e9（十进制 GB）⟹ 同一台机器两个 MemTotal
#    （251.2 GiB vs 263.4 GB），**稳态被系统性低估 1.8~4.6 GB/run**。
#    现场 5 条稳态 + 1 条新臂时累计低估约 11 GB，而 FLOOR 只有 17 GB
#    ⟹ 安全边际被吃掉 63%（记忆 `silent-lenient-fallback-in-thresholds`、
#    `thresholds-and-transcribed-numbers` 同族：单位错是**静默偏松**）。
#    自检见 `scripts/diag/check_mem_units.py`。
KiB_PER_GiB = 1024.0 * 1024.0
B_PER_GiB = 1024.0 ** 3

# 内存模型（实测，单位 GiB）
PSS_BASE = 25.0
PSS_HIST = 63.0
FLOOR = 17.0
MAX_HIST = 2                         # hist 系同时最多 2 条（实测约束，不是偏好）

# 等 wave263 的 driver 退出（phase A 用）。**字符类**避开自匹配。
DRIVER_RE = re.compile(r"launch_wave263\.py")


# ★★ driver 的内存**预留**（phase A 用）。
#
#   问题：driver 与 g2 是**两个各自独立读内存、独立决策**的进程，对**同一个
#   内存池**做判断。g2 的绝对视角 `memtotal − other − Σ稳态` **完全不知道
#   driver 即将起的那一条**，会把机器填到它自己的上限；driver 那边同时判定
#   "有余量" 就叠一条 63 GiB 的 hist ⟹ 超发。
#   （两者都是"每起一条就 break 回去重读内存"，所以竞态窗口只有**同一个
#     轮询周期**，超发上限约一条臂压在 FLOOR 上 —— 不是"必然 OOM"，但真的会过线。）
#
#   修法：把 driver **还没起的那些臂**当成"已经花掉了"计入 Σ稳态。
#   这样两个决策者看到的是**同一个内存池**，超发在结构上不可能。
#
#   ⚠ **待办不能从"候选"反推** —— `候选` 行是**累积**的，会把已经起过的臂
#     再数进来（那个坑本项目踩过：记忆 `launch-g2` 的前身把它叫
#     `driver_pending()` 并因此撒过谎）。待办 = **静态表 − 日志里"已起"的**。
#
#   静态表是 driver 自己的 todo（可用 `launch_wave263.py --help` 及其实测
#   回放核对）。**这是写死的清单**：若 driver 的臂表改了而这里没改，预留会
#   失准 —— 所以下面的 `driver_reservation()` 把"未知臂"**响亮报错**，
#   而不是当成 0（当成 0 就是静默偏松，等于没修）。
DRIVER_TODO = ["ent01_rerun_s42", "ent01_rerun_s43", "ent01_rerun_s44",
               "hist32_s42", "hist32_s43", "hist32_s44"]
DRIVER_LAUNCHED_RE = re.compile(r"已起\s+(\S+?)（")


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
    """该进程的 PSS（**GiB**）。含私有匿名页 + 共享页的公平分摊；**不是 RSS**
    （RSS 会把 8 个 spawn worker 共享的 torch 代码页重复计数）。"""
    tot = 0
    try:
        with open("/proc/%d/smaps_rollup" % pid, encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("Pss:"):
                    tot += int(ln.split()[1]) * 1024
    except OSError:
        return 0.0
    return tot / B_PER_GiB      # tot 是 bytes


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


def _u_str(u: int) -> str:
    """轮号的显示口径 —— `-1` 是"读不出"，**不许显示成数字**。

    记忆 `gate-must-print-its-inputs` 的同族：读数读不出时必须**看得出来**。
    若把 −1 直接 `%d` 打出去，现场看到的是 `u=-1`，与"跑到第 −1 轮"一样费解；
    而若退化成 0，就与"还没起"同形 —— 那是**静默偏松**，会掩盖故障。
    """
    if u < 0:
        return "u=??（metrics.jsonl 读不出）"
    return "u=%d" % u


def avail_gib() -> float:
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / KiB_PER_GiB
    except OSError:
        pass
    return 0.0


def updates_done(name: str) -> int:
    """已完成的 update 数 —— 数 `metrics.jsonl` 里带 `"update"` 键的**行数**。

    ★ 为什么不能直接 `sum(1 for _ in f)`（旧写法，2026-09-21 改）：
    这个文件里**每轮不止一行** —— `eval_validation` 行也占一行，而它
    **没有 `update` 键**（`metrics.jsonl` 实测：30 轮 + `eval_interval=5`
    ⟹ 36 行）。旧写法把 eval 行也算成了"跑过一轮" ⟹ **系统性多算**
    `轮数 // eval_interval` 轮。实测：`gae90_s42` 真实 u20、被报成 u22；
    `ent01_rerun_s43` 真实 u30、被报成 u34；`hist32_s43` 真实 u15、被报成 u18。

    ⚠ 这个读数**只进日志**（下面 500 行附近的现场行），**不喂内存门** ——
    `pending_growth()` 用的是**实测 PSS**（`live_runs()`），不是轮数。
    所以它偏了**不危险，但会骗人**：`u=` 是盯现场时唯一的进度眼睛，
    显示 22 而实际 20，会让人以为跑得比实际快（也会掩盖"冻结"这类故障
    ——那种情况下行数会停住，但停在哪一轮是错的）。

    正确口径 = **最后一个出现的 `"update": N`**，与探针脚本
    `.tmp/status_now.py` / `scripts/diag/hist32_chain.py` 对齐。
    解析失败一律返回 **−1**（"读不出"），**不许退化成 0** ——
    0 与"还没跑"同形，是静默偏松（记忆 `silent-lenient-fallback-in-thresholds`）。
    """
    p = os.path.join(ROOT, "outputs", name, "metrics.jsonl")
    if not os.path.exists(p):
        return 0                      # 文件不存在 ⟹ 确实一轮没跑，0 是真的
    last, n = -1, 0
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if '"update"' not in line:
                    continue
                try:
                    last = int(json.loads(line)["update"])
                except (ValueError, KeyError, TypeError):
                    continue          # 半行/坏行：跳过，不猜
                n += 1
    except OSError:
        return -1
    return last if last >= 0 else (0 if n == 0 else -1)


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


def memtotal_gib() -> float:
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemTotal:"):
                return int(ln.split()[1]) / KiB_PER_GiB
    except OSError:
        pass
    return 0.0


def memfree_gib() -> float:
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemFree:"):
                return int(ln.split()[1]) / KiB_PER_GiB
    except OSError:
        pass
    return 0.0


def two_views(runs, kind_new: str, phase: str = "") -> tuple[float, float, float, float]:
    """**两个视角，缺一不可**（记忆 `respawn-guard-two-views`）：
    只算增量会在稳态总和已经超过 `MemTotal` 时**放行**；只算绝对又会把
    「够用」报成「不够用」。返回 (增量余量, 绝对余量, 待涨量, 其它占用)。

      · 视角 A（增量）：可用 − Σ待涨 − 本臂稳态 − **driver 预留** ≥ FLOOR
        问的是"现在起得起吗"
      · 视角 B（绝对）：MemTotal − 其它占用 − Σ**全部**稳态（含本臂与预留）≥ FLOOR
        问的是"涨到稳态后系统还剩多少" ⟸ **这道墙 `load average` 和
        `%CPU` 都看不见**，正是「照启动时的 23GB 排 5 个、涨到 u15 就顶格」的病根

    ★ `phase` 只为一件事存在：phase A 与 wave263 的 driver **并行**，必须把
      driver 还没起的那几条**预留**进来，否则两个决策者各按各的账本放行 ⟹ 超发。
      phase B/C 时 driver 早已退出，预留自然为空（`driver_reserve` 返回 []）。

      ⚠ 预留对**两个视角都减**：那些臂**一次都还没起**（不在 `runs` 里），
        所以它们的"待涨量"就是它的**全额稳态** —— 对视角 A 而言减全额才是对的。
        已经起过的臂由 `live_runs()` 看见、已在 `runs` 里，**不重复计**。
    """
    reserve = driver_reserve(runs) if phase.upper() == "A" else []
    rsv = sum(steady_of(k) for _n, k in reserve)
    avail = avail_gib()
    pend = pending_growth(runs)
    inc = avail - pend - steady_of(kind_new) - rsv
    other = memtotal_gib() - memfree_gib() - sum(p for _n, _k, p in runs)
    total_steady = (sum(steady_of(k) for _n, k, _p in runs)
                    + steady_of(kind_new) + rsv)
    absolute = memtotal_gib() - other - total_steady
    return inc, absolute, pend, other


# ---------------------------------------------------------------- 臂表
def arms_for(phase: str) -> list[tuple[str, list[str], str, int, str | None]]:
    """[(run_name, cfgs, kind, num_updates, resume_ckpt_or_None)] —— **按优先级排序**。

    顺序即优先级（逐条放，前面的先占内存）：
      gae90 ×5        —— Task #3 的预注册检验（本波的主目的）
      补对照 s45/s46  —— 把对照族凑到 n=5，配对分析要用
      （续跑族在 phase B/C）

    ★ **`hist32_s44` 不在本表里** —— 它归 wave263 的 driver 放。
      理由：driver 的 todo 只剩它一条，本波对它的命令与 driver **逐字相同**
      （`verify_takeover_equiv.py` 已验证 argv 16 token 全等、线程数全等），
      所以"本波接手"与"driver 自己放"产出**同一条臂**，没有收益；
      而**同时接管会抢**（两边各查各的 `alive()`，之间有窗口 ⟹ 可能双开
      写同一 `outputs/` ⟹ 读数作废）。
      ⟹ 划界比抢或等都好：driver 继续管它那一条，本波管它不碰的 7 条。

    ★ 为什么**不等 driver 退出**：实测 driver 卡在 hist 门 `2<2`，要 ~108 分钟
      才会放下 s44 再退出；而本波的门**现在就能过**。等它 = 让 gae90 白等
      108 分钟、机器闲置 40–70 GB ⟹ 违反用户"把电脑性能吃满"的指示。
    """
    out = []
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
    log("  已起 %s（%s，稳态 %.0f GiB，%d 轮，起点 %s）"
        % (name, kind, steady_of(kind), nupd, os.path.basename(ck)))
    return pr.pid


# ---------------------------------------------------------------- 与 driver 划界
DRIVER_LOG = "/tmp/wave263_driver.log"
_DRIVER_STATIC = {                      # launch_wave263.py 的 ARMS（已从服务器抄回核对）
    "ent01_rerun_s42", "ent01_rerun_s43", "ent01_rerun_s44",
    "hist32_s42", "hist32_s43", "hist32_s44",
}


def driver_arms() -> set[str] | None:
    """wave263 driver 管的**全部**臂名。

    ★ 第一版写成"从日志里读它**还没放**的臂"（正则 `候选 (\\S+?)（`），
    **那是错的**：driver 日志是**累积**的，每一轮把 6 个臂全打一遍 ⟹ 这个正则
    返回的是它的**全部 ARMS**，不是"待放"。名字与行为不符，而且随着它放臂，
    返回值**不变** ⟹ 用它做划界会在错误的集合上做对的事（本例恰好安全，
    因为它本来就不是本波的臂 —— 但那是**巧合**，不是设计）。

    ⟹ 改成直接返回它的全部 ARMS（日志里的候选 ∪ 静态表），并把真正的判据
      换成**集合不相交**：只要两边臂名不交，就不可能双开同一条臂；
       这比"猜它还要放什么"是**更强且更可查**的不变量。
    返回 None 表示日志读不到 ⟹ 调用方**必须停下**（不许猜）。
    """
    try:
        with open(DRIVER_LOG, encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except OSError:
        return None
    return _DRIVER_STATIC | set(re.findall(r"候选\s+(\S+?)（", txt))


def driver_reserve(runs) -> list[tuple[str, str]]:
    """driver **还没起**的臂 → `[(名字, kind)]`。它们要占的内存必须**现在**就从
    g2 的账本里减掉，否则两个决策者各按各的账本放行 ⟹ 超发（见 `two_views`）。

    待办 = 静态表 `DRIVER_TODO` − 日志里 `已起` 的 − 已经在跑的。
    **不是**从"候选"反推（那是**累积**的，会重复计入已起的臂）。

    ★ 未知臂**响亮报错**，不当成 0。当成 0 是**静默偏松** —— 等于没修，
      而且看起来像修好了（记忆 `silent-lenient-fallback-in-thresholds`）。
    """
    try:
        with open(DRIVER_LOG, encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except OSError:
        raise RuntimeError(
            "读不到 %s ⟹ 不知道 driver 还要起什么 ⟹ 无法安全计算预留量。"
            "（不许当成 0：那正是静默偏松）" % DRIVER_LOG)

    started = set(DRIVER_LAUNCHED_RE.findall(txt))
    unknown = started - set(DRIVER_TODO)
    if unknown:
        raise RuntimeError(
            "日志里出现了静态表 `DRIVER_TODO` 之外的臂 %s ⟹ driver 的臂表改了而"
            "本文件的预留表没跟上 ⟹ 预留量算不准 ⟹ **不启动**。"
            "请同步 DRIVER_TODO（源头见 launch_wave263.py）。" % sorted(unknown))

    live = {n for n, _k, _p in runs}
    return [(n, kind_of(n)) for n in DRIVER_TODO
            if n not in started and n not in live]


def driver_alive() -> list[int]:
    """⚠ 用**字符类**避开自匹配（`pkill -f` 在 `ssh host '...'` 里会杀掉自己的教训）。"""
    return [p for p in _pids() if DRIVER_RE.search(cmdline(p))]


# ---------------------------------------------------------------- 主循环
def verify_arm(name: str) -> bool:
    """核对一条臂的**生效配置**（不是启动参数、也不是我写下的意图）。

    ★ **进程在 ≠ 跑对了**：配置链是 `load_default_config → 各 yaml 深合并 →
    env_full 强制叠加 → profiles`，后面的覆盖前面的，**文档比代码旧**。所以
    只能读它自己写出来的 `resolved_config.yaml`（记忆：以 resolved_config 为准）。

    判据：
      · `gae90_*` ⟹ `train.gae_lambda=0.9`
      · 其它      ⟹ `train.gae_lambda=0.95`
      · 全体      ⟹ `train.ppo.entropy_coef=0.01`
    （A/B 的**唯一**差异必须是 λ 本身；若 entropy_coef 也漂了，就不是 A/B 了。）
    """
    rc = os.path.join(ROOT, "outputs", name, "resolved_config.yaml")
    if not os.path.exists(rc):
        log("  ?? %s: 还没有 resolved_config.yaml（进程可能已死）" % name)
        return False
    got = subprocess.run([PY, "/tmp/rc_get.py", rc,
                          "train.gae_lambda", "train.ppo.entropy_coef"],
                         capture_output=True, text=True).stdout.strip()
    log("  %s: %s" % (name, got.replace("\n", " ")))
    ok = True
    want = "0.9" if name.startswith("gae90") else "0.95"
    if ("train.gae_lambda=%s" % want) not in got:
        log("       ✗ 期望 train.gae_lambda=%s —— **结果不可用**" % want)
        ok = False
    if "train.ppo.entropy_coef=0.01" not in got:
        log("       ✗ 期望 train.ppo.entropy_coef=0.01 —— **结果不可用**")
        ok = False
    return ok


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

    # ---------- 与 wave263 的 driver 划界（**不等它退出**，见 arms_for 的说明）----
    if phase == "A":
        d_alive = driver_alive()
        d_arms = driver_arms()
        if d_arms is None:
            log("✗ 读不到 driver 日志 %s ⟹ 无法知道它管哪些臂 ⟹ **不启动**。"
                "（不许猜：猜错就可能与它抢同一条臂，双开写同一 outputs）" % DRIVER_LOG)
            return 1
        mine = [a for a in arms if a[0] not in d_arms]
        theirs = [a for a in arms if a[0] in d_arms]
        # ★★ 断言两个集合**不相交** —— 这才是有边界效应的不变量。它可直接查验，
        #    不依赖"我猜它对不对"。相交就停：那种情况下的"让"或"抢"都会双开。
        overlap = {a[0] for a in arms} & d_arms
        if overlap:
            log("✗ 本波的臂与 driver 的臂**相交**：%s ⟹ **不启动**（会双开同一条臂）"
                % "、".join(sorted(overlap)))
            return 1
        if d_alive:
            log("wave263 的 driver 仍在跑（pid %s）⟹ **划界不抢**：它管 %s；本波管 %s"
                % (",".join(map(str, d_alive)),
                   "、".join(sorted(d_arms)), "、".join(a[0] for a in mine) or "(无)"))
            arms = mine
            if not arms:
                log("本波无臂可起（全归 driver）—— 写标记退出")
                open(go, "w").write("noop-all-driver\n")
                return 0
        else:
            log("driver 已不在 ⟹ 本波是唯一启动器（它那 %d 条归它，本波不碰）"
                % len(d_arms))
        if theirs:
            log("  （本波按设计不含：%s —— 归 driver）"
                % "、".join(a[0] for a in theirs))

    # ★★ 待起清单必须**在划界之后**才算 —— 不是之前。
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
            avail = avail_gib()
            pend = pending_growth(runs)
            inc, absolute, _p, other = two_views(runs, kind, phase)
            reserve = driver_reserve(runs) if phase == "A" else []
            rsv = sum(steady_of(k) for _n, k in reserve)
            hist_ok = (kind != "hist") or (n_hist < MAX_HIST)
            ok = (inc >= FLOOR) and (absolute >= FLOOR) and hist_ok

            log("-" * 70)
            log("在跑 %d 条：%s" % (len(runs),
                "、".join("%s(%.1fG,%s)" % (n, p, _u_str(updates_done(n)))
                          for n, _k, p in runs) or "无"))
            log("  MemTotal %.1f ｜ MemFree %.1f ｜ MemAvailable %.1f ｜ 其它占用 %.1f"
                "   （单位一律 **GiB**）"
                % (memtotal_gib(), memfree_gib(), avail, other))
            log("  待涨量 %.1f GiB（在跑的还没涨到稳态的部分，**隐形**）" % pend)
            if phase == "A":
                log("  ★ driver 预留 %.1f GiB：%s"
                    % (rsv, "、".join("%s(%s)" % (n, k) for n, k in reserve)
                       or "（driver 已无待起臂 ⟹ 预留为空）"))
            log("  候选 %s（%s，%d 轮，稳态 %.0fGiB）：" % (name, kind, nupd, steady_of(kind)))
            log("    视角A·增量：可用 %.1f − 待涨 %.1f − 本臂 %.0f − 预留 %.1f = %.1f ≥ %.0f ? %s"
                % (avail, pend, steady_of(kind), rsv, inc, FLOOR, inc >= FLOOR))
            log("    视角B·绝对：MemTotal − 其它 − Σ**全部**稳态（含预留 %.1f）= %.1f ≥ %.0f ? %s"
                % (rsv, absolute, FLOOR, absolute >= FLOOR))
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
            # ★★ **每条起完立刻验配置**，不等所有臂放完。
            #   实测教训：放臂会跨越数小时（内存门一条条等），而第一版的验证
            #   在全部放完之后才做 ⟹ 第一条如果配错，会**白等几小时**才发现，
            #   期间还白占 25GiB。`failed-launch-must-be-loud` 要的是**尽早吵**，
            #   不是"吵得完整"。
            time.sleep(130)          # 等它写出 resolved_config.yaml
            if not verify_arm(name):
                log("    ⟹ 本臂生效配置不对 ⟹ 停止（不写标记）")
                return 1
            break               # 起一条就重新读数（内存窗口会变）

        if not progressed:
            time.sleep(60)

    log("=" * 70)
    log("全部 %d 条已起：%s" % (len(started), "、".join(n for n, _p in started)))

    # ---------- 收尾扫一遍（每条起时已各验过一次，这里只做汇总）----------
    log("=== 收尾复核（全部臂的生效配置）===")
    bad = 0
    for name, _pid in started:
        if not verify_arm(name):
            bad += 1
    if bad:
        log("!! 有臂异常 —— **不写标记**，修好后可重跑")
        return 1
    log("收尾复核全部通过 ✓")

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
