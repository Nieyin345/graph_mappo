#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hist32 链（自包含）：等内存 → 起 s43/s44 → 等三臂到 u30 → 判读 → 写 verdict。

### 与 `hist_chain.py` 的区别（三条）
1. **自带判读**，不依赖另一个文件（避免两文件版本漂移；`hist_chain.py` 依赖
   `/tmp/hist_verdict.py`，而那个文件曾是**只在本地**的 ⟹ 哑火）。
2. **不重启已在跑的臂**：s42 已经以 `hist32v3_s42` 在跑（同配置、u1 已验证
   逐位等于对照），本脚本**只补 s43/s44**，绝不杀它。
3. **门按 hist 自己的 PSS 算**（实测 52.5 GB，不是基线的 23 GB）。

### 实测成本（2026-09-20，seq_len=32）
    hist32v3:  rollout 103.4 + update 194.2 = **297.7 s/轮**, PSS **52.5 GB**
    ent001  :  rollout  57.1 + update 129.0 = **186.1 s/轮**, PSS  23.1 GB
    ⟹ **1.60× 慢、2.28× 重**。这两项必须与 Δ 一起权衡。

### u1 守卫（已通过，本脚本复核并记录）
    hist32v3_s42 = ent01_rerun_s42 = ent001_s42 = **0.8725097090**
    （u1 发生在第一次梯度步之前 ⟹ 只取决于权重+种子+代码+硬件；
      加载器把新增 64 列置零 ⟹ 前向逐位等价 ⟹ 守卫必须通过。）
"""
import json
import math
import os
import re
import statistics as st
import subprocess
import sys
import time

# ★ 允许用环境变量指向别处的 outputs/ —— 让**同一个判读实现**既能在服务器上
#   跑（实时、要 /proc），也能在本地跑（抓回来的快照、只读文件）。
#
#   为什么不是"在本地另写一份判读"：本项目反复踩的坑就是**每个新地方重新推导**
#   （`updates_done` 数行数、`last_update` 数行数、`avail` 单位错 —— 三次都是
#   同一个口径在新文件里被重新发明）。判读口径必须**只有一个实现**。
#
#   ⚠ 本地跑时 `live_runs()` / `avail()` 这些读 `/proc` 的函数**没有意义**
#     （本地没有那些 run 在跑）；本地只用 `verdict()` 那一半（纯读文件）。
ROOT = os.environ.get("QKD_ROOT", "/opt/qkd/graph_mappo")
PY = "/opt/qkd/venv/bin/python"
CKPT = "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
BASE = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml", "train_hist.yaml"]
UPDATES = 30
THREADS = "8"
PSS_HIST = 52.5          # 实测
FLOOR = 17.0
MAX_RUNS = 9
# ★★ hist 的实时 PSS 会**超过**稳态 52.5（实测爬到 60），所以三条 hist 并发时
#    7 run 就能到 252 GB ⟹ 必 OOM。实测（2026-09-20T18:0x）：
#    hist32v3_s42=52.3 / hist32_s43=41.2(还在爬) / hist32_s44=31.6(还在爬)
#    + mini512_s44=27.8 + ent001×3=70.7 ⟹ 223.5 GB、只剩 29.7 GB，
#    而两条 hist 还要涨 +32 GB ⟹ 会到 255.7 > 233。
#    ⟹ **hist 系同时最多 2 条**（2×高估的 61 + 其余），其余并行额留给基线臂。
MAX_HIST = 2


def n_hist_live(runs):
    return sum(1 for nm, _, _ in runs if "hist" in nm)

# ★★ ``run`` 的约定**一律是输出目录名**（含 ``_s<种子>``）。别再混两种写法：
#    上一轮就是因为 ``read()`` 当基名、``u1_of()`` 当全名，判读**恒空**却看起来
#    像「还没跑完」。见 ``_metrics_path()`` 的说明。
#
#    原值 ["hist32v3_s42", "hist32_s43", "hist32_s44"] **混了两种约定**，且 s42 原设计
#    是**从 outputs/hist32v3_s42/checkpoint_update_000010.pt 续跑 20 轮**，
#    与 s43/s44（从 BC 新起 30 轮）不对称；那个 u10 checkpoint 已随旧节点 clnode316
#    丢失 ⟹ 现在三条统一为**从 BC 新起 30 轮**，更干净，但改变了原设计意图。
#
# ★ 现在**由种子直接算出目录名**，不再靠 ARM_RUNS[SEEDS.index(s)] 索引对齐 ——
#   两处对齐就多一处能分叉的地方。CTRL 同理。
SEEDS = (42, 43, 44)
ARM_STEM = "hist32"
CTRL = "ent01_rerun"


def arm_run(seed):
    return "%s_s%d" % (ARM_STEM, seed)


def ctrl_run(seed):
    return "%s_s%d" % (CTRL, seed)


# TODO 已由 .tmp/launch_wave263.py 接管（它用内存门统一排队，不在这里重复起臂）
TODO = []
# ★★ 临界值一律从**唯一来源**取（同目录 stats_crit.py），不在这里手写常数。
#
#   2026-09-20 修复。原写法是：
#       crit = CRIT3 if df == 2 else (2.776 if df == 4 else 2.145)
#   它只覆盖 df=2 与 df=4，**其余一律落到 2.145**——那是 **df=14** 的值。
#
#   对本脚本这不是假想路径，而是**主路径**：SEEDS=(42,43,44)，而 s44 长期被
#   MAX_HIST=2 挡着（两条 hist 并发就够把 7 run 推到 252 GB）⟹ 判读时最常见的
#   就是 **n=2（df=1）**，其正确临界值是 **12.706**。回退给的 2.145 **宽了 6 倍**
#   ⟹ |t| 落在 2.145~12.706 之间就会**假显著**。
#   详见记忆 `silent-lenient-fallback-in-thresholds`（ent001_verdict 同款，
#   那次 t=−0.69 恰好离两条线都远，是运气不是设计）。
#
#   stats_crit.t_crit 对未知 df **抛 KeyError，绝不回退**。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_crit import t_crit  # noqa: E402
EXPERT = 0.6979220689
T_LO, T_HI = -0.035, 0.035

LOG = "/tmp/hist32_chain.log"
VERDICT = "/tmp/hist_verdict.txt"


def log(m):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def avail():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) * 1024.0 / (1024.0 ** 3)
    return 0.0


def pss(pid):
    try:
        for ln in open("/proc/%d/smaps_rollup" % pid):
            if ln.startswith("Pss:"):
                return int(ln.split()[1]) * 1024.0 / (1024.0 ** 3)
    except Exception:
        pass
    return 0.0


def ppid(pid):
    try:
        s = open("/proc/%d/stat", encoding="utf-8").read()
        return int(s[s.rindex(")") + 2:].split()[1])
    except Exception:
        return 0


def cmdline(pid):
    try:
        # ★★ 必须把 NUL 换成空格（`/proc/<pid>/cmdline` 用 `\0` 分隔 argv，
        #    而 `\s` **不匹配** `\0`）。漏了这一行 ⟹ `--run-name\s+(\S+)`
        #    永远匹配不上 ⟹ `live_runs()` **恒为空** ⟹ 内存门**恒通过**
        #    ⟹ 门形同不存在，无条件启动。
        #    实测（2026-09-20）：6 条 run 在跑，日志却打印「在跑 0」。
        #    `scripts/diag/pss_per_run.py` 里有这一行，所以它一直是对的；
        #    本函数是从它移植来的，**移植时漏了**。
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            return f.read().decode("utf-8", "replace").replace("\0", " ")
    except OSError:
        return ""


def live_runs():
    """[(run, minibatch, 整run PSS)]——按 PPid 向上归组。"""
    pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    trainers = {}
    for p in pids:
        cl = cmdline(p)
        if "train_graph_mappo" not in cl:
            continue
        m = re.search(r"--run-name\s+(\S+)", cl)
        if m:
            trainers[p] = m.group(1)
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
                    mb = int(m.group(1)); break
        except OSError:
            pass
        out.append((name, mb, tot.get(t, 0.0)))
    return out


def steady_of(name, mb):
    return PSS_HIST if "hist" in name else (29.5 if mb == 512 else 25.0)


def live_run_names():
    """此刻**真在做**的 run 名集合（走 /proc，与门同一个数据源）。

    ★ 2026-09-20 实测的 bug：本链的 `launched` 是**局部 set**，
    重启后从空开始 ⟹ 会把**已经在跑**的臂再起一遍。
    实际发生：`hist32_s43` 被起了第二份（pid 176975），
    与已在跑、已有 16 轮的那份（pid 141126）**写同一个 outputs 目录**
    （`metrics.jsonl` 被两个进程交错追加、`checkpoint_*.pt` 互相覆盖）
    ⟹ **这一臂的读数作废**。
    ⟹ 判据必须问**进程表**，不能只信自己的账本
    （同族：[[renaming-running-script-changes-nothing]] 里"内存里的文本"那类错误——
      **自己的状态 ≠ 世界的状态**）。
    """
    names = set()
    # ★★ 没有 /proc 时**必须响亮报错**，不能安静返回空集。
    #    空集在调用方读作「没有 run 在跑」⟹ `launch()` 会把**已经在跑**的臂
    #    再起一遍，两份进程写同一个 `outputs/` ⟹ **该臂读数作废**
    #    （本项目实测过：`hist32_s43` 双开、`metrics.jsonl` 被交错追加）。
    #    这与 `updates_done()` 区分 `-1`（读不出）/ `0`（真的没跑）是**同一条原则**：
    #    **「读不出」不许长得像「没有」**。
    #    （旧写法用 `glob.glob("/proc/[0-9]*")`，在无 /proc 的平台上**安静返回空**，
    #      恰好是最危险的那种失败。）
    if not os.path.isdir("/proc"):
        raise RuntimeError(
            "没有 /proc ⟹ 无法判断哪些 run 在跑。**不能返回空集**："
            "调用方会把「空集」读成「没有 run 在跑」而重复起臂。"
            "本函数只在 Linux 训练节点上有意义。")
    # ★★ 2026-09-21：这行原本写 `glob.glob("/proc/[0-9]*")`，而本文件**从未
    #    `import glob`** ⟹ 一调用就 `NameError`。它没被发现，是因为本文件的
    #    `TODO = []`（已被 `.tmp/launch_wave263.py` 接管）⟹ 阶段 1 的循环
    #    `while len(launched) < len(TODO)` **一次都不进** ⟹ `launch()` 从不执行
    #    ⟹ `live_run_names()` **是一条从未跑过的代码路径**
    #    （记忆 `never-run-code-path-hides-bugs`）。
    #    改成 `os.listdir("/proc")` 走**同一数据源**，与上面的 `live_runs()`
    #    逐字一致 —— 两个函数读世界的方式不同，本身就是分叉点。
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % d, "rb") as f:
                c = f.read().decode("utf-8", "replace").replace("\x00", " ")
        except OSError:
            continue
        if "train_graph_mappo.py" not in c:
            continue
        m = re.search(r"--run-name\s+(\S+)", c)
        if m:
            names.add(m.group(1))
    return names


def launch(seed):
    run = "hist32_s%d" % seed
    # ★ 双保险：先问**进程表**。已在跑就直接记账，不重复起。
    if run in live_run_names():
        log("  %s **已在跑** ⟹ 不重复启动（只记账）" % run)
        return None
    cmd = ([PY, "scripts/train/train_graph_mappo.py", "--configs"] + BASE +
           ["--checkpoint", CKPT, "--seed", str(seed),
            "--num-updates", str(UPDATES), "--run-name", run])
    env = dict(os.environ, OMP_NUM_THREADS=THREADS, PYTHONUNBUFFERED="1")
    f = open("/tmp/%s.out" % run, "ab")
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    log("  ★ 起 %s (pid=%d)" % (run, p.pid))
    return p


def last_update(run):
    """该臂**已完成到第几轮** —— 按最后一个 `"update": N` 读，**不是数行数**。

    ★★ 2026-09-21 修：原实现是 `if json.loads(ln).get("update"): n += 1`，
    即**数带 update 键的行数**。这在两种情形下都错：

      (a) **续跑臂**：`update` 计数器**续着编**，不归零
          （`mappo_trainer.py` 的 `target_updates = self.update_count + num_updates`）。
          从 u5 崩、续跑 20 轮 ⟹ 文件里是 `6..25`，行数 **20**，
          而真实进度是 **25** ⟹ 少算 5 轮。
      (b) **中途缺 eval**：行数只数训练行，这一项恰好不受影响；
          但一旦将来把 eval 行也计进来就会多算（`launch_g2.updates_done()`
          就踩了这个，见下）。

    正确口径 = **最后一个 `"update": N` 的值**，与 `read()`（本文件）、
    `launch_g2.updates_done()`、`.tmp/status_now.py` 四处**统一**。
    记忆 `eval-update-number-not-from-position`：本项目在这上面已经错了三次
    （按行号 / 按累计行数 / 按行数），每次都在**新地方重新推导**而不是照抄已有的。

    本函数被阶段 2 的"等三臂到 u30"用（本节点三臂都是**从头跑**，
    所以按行数与按轮号在本例恰好相同 —— 这就是它一直没暴露的原因：
    **只有续跑或 eval 计数变化时才分叉**）。返回 −1 = 读不出。
    """
    p = _metrics_path(run)
    if not os.path.exists(p):
        return -1
    last = -1
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or '"update"' not in ln:
                    continue
                try:
                    last = int(json.loads(ln)["update"])
                except (ValueError, KeyError, TypeError):
                    continue          # 半行/坏行：跳过，不猜
    except OSError:
        return -1
    return last


# ============================ 判读 ============================
def _metrics_path(run):
    """``run`` 一律是**输出目录名**（如 ``hist32_s42`` / ``ent01_rerun_s42``）。

    ★★ 本文件出过一个**静默假阴性**：``read()`` 与 ``u1_of()`` 对 ``run`` 的约定
    **相反** —— ``read()`` 当基名再拼 ``_s<种子>``，``u1_of()`` 当全名。
    于是 ``read(ARM_RUNS[i], s)`` 去读 ``outputs/hist32_s42_s42/metrics.jsonl``
    （**永不存在**），恒返回 ``[]`` ⟹ §1 主判据、§2 逐轮 Δ、§3 跨种子 SD、
    §5 吞吐代价**全部不输出**，只打印「!! 数据不足，无法判读」。

    **而 §0 的 u1 守卫走 ``u1_of()``，是通的** ⟹ 输出看起来像「守卫过了、
    只是数据还没到」，不像 bug。上一轮「修正」只改了 ``ARM_RUNS`` 的写法
    （把 ``hist32v3_s42`` 换成 ``hist32_s42``），**没有统一两条路径的约定**
    ⟹ 修了一半，坏的那半继续静默（见记忆 ``later-sections-retract-earlier-ones``：
    别信文档说「已修」，要对代码求证）。

    ⟹ 现在**只有这一个函数**拼路径，``read()`` 与 ``u1_of()`` 都调它。
    ``verdict()`` §0 会打印实际探测的路径，路径写错**当场可见**。
    """
    return os.path.join(ROOT, "outputs", run, "metrics.jsonl")


def read(run):
    """读一条臂的 ``metrics.jsonl``。``run`` = **输出目录名**（含 ``_s<种子>``）。"""
    p = _metrics_path(run)
    out = []
    if not os.path.exists(p):
        return out
    last = None
    for ln in open(p, encoding="utf-8", errors="replace"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
        except ValueError:
            continue          # 半行（进程正写到一半）/ 坏行：跳过，不猜
        if "update" in o:
            last = o["update"]
        ev = o.get("eval_validation")
        if isinstance(ev, dict) and ev.get("per_seed_success") and last is not None:
            out.append((last, [float(x) for x in ev["per_seed_success"]],
                        o.get("rollout_s"), o.get("update_s")))
    return out


def plat_ps(rows):
    d = {u: p for u, p, _, _ in rows}
    if 25 not in d or 30 not in d or len(d[25]) != len(d[30]):
        return None
    return [st.mean([d[25][i], d[30][i]]) for i in range(len(d[25]))]


def u1_of(run):
    p = _metrics_path(run)
    try:
        for ln in open(p, encoding="utf-8", errors="replace"):
            ln = ln.strip()
            if ln:
                try:
                    return json.loads(ln).get("mean_success_rate")
                except ValueError:
                    continue      # 半行：继续找第一条能解析的
    except OSError:
        pass
    return None


def verdict():
    L = []
    A = L.append
    A("=" * 74)
    A("hist32 判读：时序编码器（只开节点通道，seq_len=32）  对照 = %s（本节点、8 线程、同 BC）" % CTRL)
    A("=" * 74)

    A("\n### 0. ★ u1 守卫（同 BC + 同种子 + 同节点 ⟹ 必须逐位相同）")
    A("    hist 是**结构改动**（新增 64 列输入），加载器把它们**置零** ⟹ 前向逐位等价，")
    A("    所以初始策略必须与对照完全相同。不同 ⟹ 加载器没做到承诺，下面全部不可信。")
    # ★ 打印实际探测的路径 —— 路径写错必须**当场可见**。上一轮 read() 拼错目录名，
    #   判读恒空而 §0 守卫照过，看起来像「还没跑完」，静默了很久才发现。
    for s in SEEDS:
        A("   [路径] %s" % _metrics_path(arm_run(s)))
    guard_ok = True
    for s in SEEDS:
        a = u1_of(arm_run(s))
        b = u1_of(ctrl_run(s))
        if a is None or b is None:
            A("   s%d  hist=%s  ctrl=%s（数据缺）" % (s, a, b)); guard_ok = False; continue
        d = a - b
        ok = abs(d) < 1e-9
        guard_ok = guard_ok and ok
        A("   s%d  hist=%.12f  ctrl=%.12f  Δ=%.2e  %s"
          % (s, a, b, d, "✓ 逐位相同" if ok else "✗ **不相同**"))
    A("   ⟹ %s" % ("暖启动确实保住了，配对干净" if guard_ok
                    else "**加载器没做到承诺的等价，下面结果不可信**"))

    A("\n### 1. 主判据：平台 u25/u30，逐请求种子配对")
    avail_s = []
    for s in SEEDS:
        if read(arm_run(s)) and read(ctrl_run(s)):
            avail_s.append(s)
    A("   可用训练种子：%s" % avail_s)
    if not avail_s:
        A("   !! **一个训练种子都没有** —— 两侧数据都没命中 u25/u30 平台窗口。")
        A("      （不是「测不出」，是「没测到」：三臂需至少一条 hist 到 u30。）")
        return "\n".join(L)
    # ★★ 2026-09-21：`< 2` 太宽 —— n=1 时 df=0，`t_crit(0)` 抛 KeyError
    #    先于下面那句 n=1 的诚实话术，于是脚本**崩在解释"为什么不能判读"的路上**。
    #    改成拒绝 n<2，把 n=1 的说明真正送出去。
    if len(avail_s) < 2:
        A("   ⚠ **只有 1 个训练种子（df=0）—— 本框架下没有合法判据**（t 无定义，"
          "且 n=1 的分辨率约 0.035，与效应量同量级）。")
        A("      正确做法是补训练种子，不是拿它下结论。")
        return "\n".join(L)

    per_seed, nc = [], None
    for s in avail_s:
        Aps, Cps = plat_ps(read(arm_run(s))), plat_ps(read(ctrl_run(s)))
        if Aps is None or Cps is None or len(Aps) != len(Cps):
            A("   s%d：平台窗口不全" % s); continue
        nc = len(Aps)
        per_seed.append(st.mean([Aps[i] - Cps[i] for i in range(nc)]))
    if not per_seed:
        A("   !! 没有任何种子两侧都命中平台窗口")
        return "\n".join(L)

    m = st.mean(per_seed)
    sd = st.stdev(per_seed) if len(per_seed) > 1 else float("nan")
    se = sd / math.sqrt(len(per_seed)) if len(per_seed) > 1 else float("nan")
    t = m / se if se and se == se and se != 0 else float("nan")
    df = len(per_seed) - 1
    # 未知 df 由 t_crit 抛 KeyError —— 宁可炸，不要静默换成宽松值。
    crit = t_crit(df)
    A("   n=%d 训练种子 × %d 请求种子   df=%d  临界值=%.3f" % (len(per_seed), nc, df, crit))
    A("   Δ = %+.4f   SD(训练种子) = %.4f   SE = %.4f   t = %+.2f  %s"
      % (m, sd, se, t, "★ 过线" if abs(t) > crit else "未过线"))
    # ★ t 无定义（df=0，单个训练种子）时不许报「过线/未过线」：nan 比较恒 False
    #   ⟹ 会静默读成「未过线」，那是把「没有合法判据」伪装成「测过但没测出」。
    if df == 0 or t != t:
        A("   ⚠ **n=1（df=0），t 无定义 —— 本框架下没有合法判据**，"
          "上面的过线/未过线**不成立**，必须补训练种子。")
    A("   逐训练种子 Δ: %s" % " ".join("%+.4f" % x for x in per_seed))
    if m >= T_HI:
        v = "★ 时序信息**有用**（Δ ≥ +0.035）⟹ 值得加深（多通道 / 接边）"
    elif m <= T_LO:
        v = "★ 时序编码器**有害**（Δ ≤ −0.035）"
    else:
        v = "**测不出**（|Δ| < 0.035）—— n=3 分辨率下的正确读法，不是「无用」"
    A("   ⟹ 判据：%s" % v)

    A("\n### 2. 逐轮配对 Δ")
    evs = [1, 5, 10, 15, 20, 25, 30]
    A("   %-12s" % "臂" + "".join("%9s" % ("u%d" % u) for u in evs))
    row = "   %-12s" % "hist32"
    for u in evs:
        dd = []
        for s in avail_s:
            Aa = {x[0]: x[1] for x in read(arm_run(s))}
            Cc = {x[0]: x[1] for x in read(ctrl_run(s))}
            if u in Aa and u in Cc and len(Aa[u]) == len(Cc[u]):
                dd.append(st.mean([Aa[u][i] - Cc[u][i] for i in range(len(Aa[u]))]))
        row += "%+9.4f" % st.mean(dd) if dd else "%9s" % "--"
    A(row)

    A("\n### 3. 逐轮跨种子 SD")
    A("   %-12s" % "臂" + "".join("%9s" % ("u%d" % u) for u in evs))
    for nm in ("ctrl", "hist32"):
        row = "   %-12s" % nm
        for u in evs:
            vs = []
            for s in avail_s:
                run = ctrl_run(s) if nm == "ctrl" else arm_run(s)
                for (uu, p, _, _) in read(run):
                    if uu == u:
                        vs.append(st.mean(p))
            row += "%9.4f" % st.stdev(vs) if len(vs) > 1 else "%9s" % "--"
        A(row)

    A("\n### 4. 绝对水平 vs 专家锚 %.4f（⚠ 臂 vs 专家有节点偏置不抵消）" % EXPERT)
    for nm, get in (("ctrl", lambda s: read(ctrl_run(s))),
                    ("hist32", lambda s: read(arm_run(s)))):
        vals = []
        for s in avail_s:
            r = get(s)
            # ★ 按**轮号**取平台（u25/u30），不按位置取末两点 ——
            #   续跑臂（轮号 31…50）或中途缺一轮 eval 的臂，末两点会变成 u45/u50，
            #   「平台」就不是平台（记忆 window-aggregation-must-align-both-sides）。
            d = {x[0]: x[1] for x in r}
            if 25 in d and 30 in d and len(d[25]) == len(d[30]):
                vals.append(st.mean([st.mean([d[25][i], d[30][i]])
                                     for i in range(len(d[25]))]))
        if vals:
            A("   %-10s 平台=%.4f  Δ(臂−专家)=%+.4f   逐种子 %s"
              % (nm, st.mean(vals), st.mean(vals) - EXPERT,
                 " ".join("%.4f" % v for v in vals)))

    A("\n### 5. ★ 吞吐代价（决定「值不值得继续投入」的一半）")
    A("   %-12s %10s %10s %10s" % ("臂", "rollout_s", "update_s", "合计"))
    tot = {}
    for nm, get in (("ctrl", lambda s: read(ctrl_run(s))),
                    ("hist32", lambda s: read(arm_run(s)))):
        rs, us = [], []
        for s in avail_s:
            rows = [x for x in get(s) if x[2] and x[3]]
            if rows:
                rs.append(st.mean([x[2] for x in rows]))
                us.append(st.mean([x[3] for x in rows]))
        if rs:
            tot[nm] = (st.mean(rs), st.mean(us))
            A("   %-12s %10.1f %10.1f %10.1f" % (nm, tot[nm][0], tot[nm][1],
                                                 tot[nm][0] + tot[nm][1]))
    if "ctrl" in tot and "hist32" in tot:
        A("   ⟹ 倍数：rollout %.2fx  update %.2fx  整轮 %.2fx"
          % (tot["hist32"][0] / tot["ctrl"][0], tot["hist32"][1] / tot["ctrl"][1],
             (tot["hist32"][0] + tot["hist32"][1]) / (tot["ctrl"][0] + tot["ctrl"][1])))
        A("   ⚠ 还慢 1.6x、还重 2.3x ⟹ 即使 Δ 为正，也要按「每单位算力的收益」权衡")
    A("")
    A("=" * 74)
    return "\n".join(L)


def main():
    log("=" * 72)
    log("hist32 链启动（seq_len=32；只补 s43/s44，不动在跑的 hist32v3_s42）")

    # ---- 预检：代码与配置都在位 ----
    tr = open(os.path.join(ROOT, "qkd_rl/rl/algos/mappo_trainer.py"), encoding="utf-8").read()
    if "_upgrade_optimizer_state_for_model" not in tr:
        log("!! 节点 trainer **不含优化器升级** ⟹ 会崩在第一个 optimizer.step()")
        open(VERDICT, "w", encoding="utf-8").write(
            "STATE=failed\n缺 _upgrade_optimizer_state_for_model\n")
        return
    cfg = open(os.path.join(ROOT, "configs/train_hist.yaml"), encoding="utf-8").read()
    if not re.search(r"^\s*seq_len:\s*32", cfg, re.M):
        log("!! train_hist.yaml 里没有 seq_len: 32")
    if not os.path.exists(os.path.join(ROOT, CKPT)):
        log("!! BC 起点不存在")
        open(VERDICT, "w", encoding="utf-8").write("STATE=failed\nBC 起点不存在\n")
        return
    log("预检通过：优化器升级 ✓ / seq_len=32 ✓ / BC 起点 ✓")

    # ---- 阶段 1：按内存门起 s43/s44 ----
    launched = set()
    t0 = time.time()
    while len(launched) < len(TODO) and time.time() - t0 < 4 * 3600:
        runs = live_runs()
        n = len(runs)
        a = avail()
        grow = sum(max(0.0, steady_of(nm, mb) - cur) for nm, mb, cur in runs)
        margin = a - grow - PSS_HIST
        if n < MAX_RUNS and n_hist_live(runs) < MAX_HIST and margin >= FLOOR:
            for _, seed in TODO:
                if seed in launched:
                    continue
                launched.add(seed)
                launch(seed)
                time.sleep(15)
                break
        else:
            if int(time.time() - t0) % 600 < 140:
                log("  门未过：可用 %.0f 待涨 %.1f 余量 %.1f（在跑 %d，其中 hist %d/%d）"
                    % (a, grow, margin, n, n_hist_live(runs), MAX_HIST))
        time.sleep(60)
    log("已起 %d/%d" % (len(launched), len(TODO)))

    # ---- 阶段 2：等三臂到 u30 ----
    t0 = time.time()
    while time.time() - t0 < 6 * 3600:
        n = sum(1 for s in SEEDS if last_update(arm_run(s)) >= UPDATES)
        if n == len(SEEDS):
            log("✓ 三臂全部到 u%d（%.0f 分钟）" % (UPDATES, (time.time() - t0) / 60))
            break
        if int(time.time() - t0) % 900 < 130:
            log("  进度 %d/%d (%s)" % (n, len(SEEDS),
                 " ".join("%s=u%d" % (arm_run(s), last_update(arm_run(s)))
                          for s in SEEDS)))
        time.sleep(150)
    else:
        log("!! 等待超时，按现有数据判读")

    txt = verdict()
    for ln in txt.splitlines():
        log("  " + ln)
    n_done = sum(1 for s in SEEDS if last_update(arm_run(s)) >= UPDATES)
    with open(VERDICT, "w", encoding="utf-8") as f:
        f.write("STATE=%s\n" % ("done" if n_done == len(SEEDS) else "timeout"))
        f.write(txt)
    log("判读写入 %s" % VERDICT)


if __name__ == "__main__":
    main()
