#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gae90 补种子到 n=5：起 ent01_n5_s45/46 + gae90_n5_s45/46，等 u30，内联判读。

### 为什么要做
wave5 判读（docs/训练诊断记录.md「第五波判读」节）：
    gae90 (λ=0.90)  Δ=+0.0167  SD=0.0121  t=+2.40  未过线
n=3 ⟹ df=2 ⟹ 临界值 **4.303**。+0.0167 恰好卡在测不出的量级上，
但它的**形状**不像噪声：逐轮配对 Δ 单调上升且全程同号（−0.0066 → +0.0211）。

本项目有一个**完全同形**的先例（记忆 `entropy-coef-0-01-lead`）：
entropy_coef 也是 n=3 时 p=0.074 未达显著 → 补种子到 n=5 → p=0.0074 过线。

⚠ 但那条先例的教训有**两半**，第二半必须一起记住：
  **越线靠的是功效不是效应量。** n=5 显著 ≠ +0.0167 是真效应，只等于「比 0 大」。
  所以判读**必须**同时报效应量、SD 与逐轮曲线。
⚠ 而且那次**五个种子整批在旧节点**，Δ 与硬件偏置同量级，去偏置后归零。
  ⟹ 本波**四个 run 全部在本节点现跑**，判读前做 u1 指纹自检。

### 对照为什么也要新跑
n=5 要求每个训练种子**两侧都有读数**，现有对照只到 s44 ⟹ s45/s46 必须补。
不能拿 outputs/ 里现成的同配置异种子臂代替：
  `ent03_*` / `ent001_*` / `ep2_*` 都改过别的旋钮（不是同配置），
  `ent01_t8_*` 是跨节点搬来的（u1 差 0.0147505，不可配对）。

### 门：本脚本的质量点全在这里
1. **必须打印自己的输入**（记忆 `gate-must-print-its-inputs`）。
   恒真的门**不报错**，会一路藏到 OOM 才现形。
2. **必须算待涨量**（记忆 `mem-gate-must-count-warmup-growth`）。
   `free` 只报当前读数；刚起的 run 从 ~9G 爬到稳态 25G，**待涨量是隐形的**。
3. **必须与独立读数交叉验证**：在跑 run 数用 `/proc` 数一遍，
   再用**文件系统**（outputs/ 里 metrics.jsonl 的 mtime）独立数一遍。
   不一致就**拒绝启动并吵**。恒真的门与恒假的门症状相同，
   但前者不留痕 —— 只有两个独立读数互相矛盾时才暴露。
4. **不杀任何东西**，只等。
5. **启动后必须验证**（记忆 `failed-launch-must-be-loud`）：进程秒死时
   「失败了」与「还在跑」从外部无法区分。
"""
import glob
import json
import math
import os
import re
import statistics as st
import subprocess
import sys
import time

ROOT = "/opt/qkd/graph_mappo"
PY = "/opt/qkd/venv/bin/python"
CKPT = "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
BASE = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml"]

QUEUE = [("ent01_n5_s45", "train_ent01.yaml", 45),
         ("gae90_n5_s45", "train_gae90.yaml", 45),
         ("ent01_n5_s46", "train_ent01.yaml", 46),
         ("gae90_n5_s46", "train_gae90.yaml", 46)]

UPDATES = 30
STEADY = 25.0      # GB/run 稳态（本配置实测 23.8~25.0）
FLOOR = 17.0       # GB 绝对余量下限
THREADS = "8"
LOG = "/tmp/gae90_n5.log"
VERDICT = "/tmp/gae90_n5_verdict.txt"


def need_for(n_left):
    """还差 `n_left` 条没起时，放行**下一条**需要的可用量。

    ★ 必须随「已起臂数」递减，不能写成 `len(QUEUE) * STEADY + FLOOR` 这样的常量。
    起完第一条后，那条臂会从 ~4G 爬到稳态 25G，而**这段已经计入 `grow`**
    （本波起的臂同样出现在 `live_runs()` 里）。若 NEED 里再算一份 25G，
    同一个 25G 就被算了两遍 ⟹ **余量永远差 25G ⟹ 剩下的臂一辈子起不来**。
    我第一版就是这么写的，幸好在一条都没起时发现（记忆
    `gate-must-print-its-inputs`：门必须打印输入，打印出来才看得出来）。
    """
    return (n_left - 1) * STEADY + FLOOR

# 判读口径（预注册，跑之前写死）
CTRL_RUNS = {42: "ent01_rerun_s42", 43: "ent01_rerun_s43", 44: "ent01_rerun_s44",
             45: "ent01_n5_s45", 46: "ent01_n5_s46"}
ARM_RUNS = {42: "gae90_s42", 43: "gae90_s43", 44: "gae90_s44",
            45: "gae90_n5_s45", 46: "gae90_n5_s46"}
SEEDS = (42, 43, 44, 45, 46)
CRIT5 = 2.776      # df=4

START = time.time()


def log(m):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- 读数

def avail():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / 1e6
    return 0.0


def cmdline(pid):
    """★ 必须把 NUL 换成空格。

    `/proc/<pid>/cmdline` 用 `\\0` 分隔 argv，而正则里的 `\\s` **不匹配** `\\0`。
    漏了这一行 ⟹ `--run-name\\s+(\\S+)` 永远匹配不上 ⟹ `live_runs()` 恒为空
    ⟹ 门里的 n=0、grow=0 ⟹ **条件恒真、无条件启动**。
    这正是 2026-09-20 那次 OOM 的根因（记忆 `gate-must-print-its-inputs`）。
    """
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            return f.read().decode("utf-8", "replace").replace("\0", " ")
    except OSError:
        return ""


def pss_total(pid):
    """父进程及其全部后代的 PSS 之和（RSS 会把共享页重复计数）。"""
    seen, tot = set(), 0.0
    stack = [pid]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            for ln in open("/proc/%d/smaps_rollup" % p):
                if ln.startswith("Pss:"):
                    tot += int(ln.split()[1]) / 1e6
                    break
        except OSError:
            pass
        try:
            out = subprocess.run(["pgrep", "-P", str(p)], capture_output=True,
                                 text=True).stdout
            stack.extend(int(x) for x in out.split())
        except Exception:
            pass
    return tot


def live_runs():
    """[(run_name, pid, pss)]，只认带 `--run-name` 的训练进程。"""
    out = []
    for d in glob.glob("/proc/[0-9]*"):
        pid = int(d.rsplit("/", 1)[1])
        c = cmdline(pid)
        if "train_graph_mappo.py" not in c:
            continue
        m = re.search(r"--run-name\s+(\S+)", c)
        if not m:
            continue
        out.append((m.group(1), pid, pss_total(pid)))
    return out


def fs_runs(window_s=900):
    """独立读数：用文件系统数「最近还在写的 run」。

    与 `live_runs()` 走**完全不同的数据源**（outputs/ 的 mtime vs /proc），
    用来交叉验证门的输入。两条读数矛盾时说明有一边坏了。
    """
    now, out = time.time(), []
    for p in glob.glob(os.path.join(ROOT, "outputs", "*", "metrics.jsonl")):
        try:
            if now - os.path.getmtime(p) <= window_s:
                out.append(os.path.basename(os.path.dirname(p)))
        except OSError:
            pass
    return sorted(out)


def last_update(run):
    p = os.path.join(ROOT, "outputs", run, "metrics.jsonl")
    if not os.path.exists(p):
        return 0
    n = 0
    for ln in open(p, encoding="utf-8"):
        try:
            if json.loads(ln).get("update"):
                n += 1
        except ValueError:
            pass
    return n


# ---------------------------------------------------------------- 门

def gate(n_left):
    """返回 (是否放行, 说明)。**必须打印输入**。"""
    need = need_for(n_left)
    a = avail()
    runs = live_runs()
    n = len(runs)
    # 待涨量：刚起的 run 从当前 PSS 爬到稳态，这段是隐形的
    grow = sum(max(0.0, STEADY - p) for _, _, p in runs)
    margin = a - grow - need
    detail = ("可用 %.1fG ｜ 在跑 %d 条(合计%.1fG) ｜ 待涨 %.1fG ｜ "
              "本波剩余 %d 条需 %.1fG(%d×%.0f+余量%.0f) ｜ **余量 %.1fG**"
              % (a, n, sum(p for _, _, p in runs), grow, n_left, need,
                 max(0, n_left - 1), STEADY, FLOOR, margin))
    return margin >= 0, detail


# ---------------------------------------------------------------- 起臂

def launch(run, cfg, seed):
    env = dict(os.environ, OMP_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS)
    args = ([PY, "-u", "scripts/train/train_graph_mappo.py", "--configs"]
            + BASE + [cfg]
            + ["--checkpoint", CKPT, "--seed", str(seed),
               "--num-updates", str(UPDATES), "--run-name", run])
    lf = open("/tmp/%s.log" % run, "ab")
    p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=lf, stderr=lf,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    return p


def verify_launch(run, proc, wait_s=120):
    """★ 启动后验证：秒死时「失败了」与「还在跑」从外部无法区分。"""
    time.sleep(wait_s)
    alive = proc.poll() is None
    cfg_ok = os.path.exists(os.path.join(ROOT, "outputs", run,
                                         "resolved_config.yaml"))
    if not alive or not cfg_ok:
        tail = ""
        try:
            with open("/tmp/%s.log" % run, encoding="utf-8",
                      errors="replace") as f:
                tail = "".join(f.readlines()[-6:])
        except OSError:
            pass
        return False, ("!! %s **启动失败** alive=%s resolved_config=%s\n%s"
                       % (run, alive, cfg_ok, tail))
    return True, "   %s 已起来（pid %d，resolved_config 已落盘）" % (run, proc.pid)


# ---------------------------------------------------------------- 判读

def read_per_seed(run):
    """[(update, [逐请求种子成功率])] —— 只取 eval_validation 行。"""
    p = os.path.join(ROOT, "outputs", run, "metrics.jsonl")
    out, last = [], None
    if not os.path.exists(p):
        return out
    for ln in open(p, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        o = json.loads(ln)
        if "update" in o:
            last = o["update"]
        ev = o.get("eval_validation")
        if isinstance(ev, dict) and ev.get("per_seed_success") and last is not None:
            out.append((last, [float(x) for x in ev["per_seed_success"]]))
    return out


def u1(run):
    p = os.path.join(ROOT, "outputs", run, "metrics.jsonl")
    if not os.path.exists(p):
        return None
    for ln in open(p, encoding="utf-8"):
        try:
            o = json.loads(ln)
        except ValueError:
            continue
        if o.get("update") == 1:
            return o.get("mean_success_rate")
    return None


def plat(rows):
    d = dict(rows)
    if 25 not in d or 30 not in d or len(d[25]) != len(d[30]):
        return None
    return [st.mean([d[25][i], d[30][i]]) for i in range(len(d[25]))]


def verdict():
    L = []
    A = L.append
    A("=" * 72)
    A("gae90 补种子判读（n=5 训练种子 × 逐请求种子配对）")
    A("命题：λ=0.90 是否优于 λ=0.95（对照 ent01）")
    A("预注册：df=4 ⟹ 临界值 %.3f；**同时报效应量/SD/逐轮曲线**——" % CRIT5)
    A("        越线靠的是功效不是效应量（记忆 entropy-coef-0-01-lead 的后半条）")
    A("=" * 72)
    A("")

    # --- 0. u1 指纹自检（免费且决定性）---
    A("### 0. u1 指纹自检（第一次梯度更新之前，只由 权重+种子+代码+硬件 定）")
    A("   同种子两侧必须逐位相同，否则中间有未记账的变量")
    known = {42: 0.8725097090, 43: 0.8740972490, 44: 0.8710084920}
    bad = []
    for s in SEEDS:
        c, a = u1(CTRL_RUNS[s]), u1(ARM_RUNS[s])
        tag = ""
        if s in known:
            tag = "（已知 s%d 应=%.10f）" % (s, known[s])
            if c is not None and abs(c - known[s]) > 5e-10:
                bad.append("对照 %s u1=%.10f != 已知 %.10f"
                           % (CTRL_RUNS[s], c, known[s]))
        if c is None or a is None:
            if s >= 45:
                bad.append("%s / %s 缺 u1（本波新臂未跑完首轮）"
                           % (CTRL_RUNS[s], ARM_RUNS[s]))
            continue
        same = abs(c - a) < 5e-10
        A("   s%-3d 对照 %-16s %.10f ｜ 臂 %-14s %.10f  %s%s"
          % (s, CTRL_RUNS[s], c, ARM_RUNS[s], a,
             "✓逐位相同" if same else "✗差 %.7f" % (a - c), tag))
        if not same:
            bad.append("s%d 两侧 u1 不同（差 %.7f）⟹ 不可配对"
                       % (s, a - c))
    A("")
    if bad:
        A("   ❌ 自检未过：")
        for b in bad:
            A("      - " + b)
        A("")
    else:
        A("   ✅ 自检通过：全部 %d 个种子两侧 u1 逐位相同" % len(SEEDS))
        A("")

    # --- 1. 主判据 ---
    A("### 1. 主判据：平台 u25/u30，逐请求种子配对")
    ds, used = [], []
    for s in SEEDS:
        Cv, Av = plat(read_per_seed(CTRL_RUNS[s])), plat(read_per_seed(ARM_RUNS[s]))
        if Cv is None or Av is None or len(Cv) != len(Av):
            A("   s%-3d 数据不全（对照 %s / 臂 %s）——跳过"
              % (s, "有" if Cv else "无", "有" if Av else "无"))
            continue
        ds.append(st.mean([Av[i] - Cv[i] for i in range(len(Av))]))
        used.append(s)
    if len(ds) < 2:
        A("   可用种子不足（%d），无法判读" % len(ds))
        return "\n".join(L)
    m = st.mean(ds)
    sd = st.stdev(ds)
    se = sd / math.sqrt(len(ds))
    t = m / se if se else 0.0
    df = len(ds) - 1
    crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(df, 2.0)
    A("   用了 %d 个训练种子 %s（df=%d，临界值 %.3f）" % (len(ds), used, df, crit))
    A("   Δ=%+.4f   SD=%.4f   SE=%.4f   t=%+.2f   %s"
      % (m, sd, se, t, "**过线**" if abs(t) >= crit else "未过线"))
    A("   逐训练种子 Δ: " + " ".join("%+.4f" % x for x in ds))
    A("")
    A("   ★ 读法：过线只说明「比 0 大」，**不等于 +%.4f 是真效应**。" % m)
    A("     wave5 的 n=3 读数 +0.0167 与本次共用 s42/43/44 三个种子，**不是独立复现**。")
    A("")

    # --- 2. 逐轮配对 Δ 与逐轮跨种子 SD ---
    A("### 2. 逐轮配对 Δ（逐请求种子配对后取均值）")
    rounds = [10, 15, 20, 25, 30]
    A("   %-8s %s" % ("轮", "".join("%9s" % ("u%d" % r) for r in rounds)))
    for lbl, src in (("对照组内 SD", CTRL_RUNS), ("gae90", ARM_RUNS)):
        row = []
        for r in rounds:
            vals = []
            for s in SEEDS:
                d = dict(read_per_seed(src[s]))
                if r in d:
                    vals.append(d[r])
            if len(vals) >= 2:
                row.append(st.stdev([st.mean(v) for v in vals]))
            else:
                row.append(float("nan"))
        A("   %-8s %s" % (lbl[:8], "".join("%9.4f" % x for x in row)))
    A("")
    A("   逐轮配对 Δ：")
    for r in rounds:
        diffs = []
        for s in SEEDS:
            dc, da = dict(read_per_seed(CTRL_RUNS[s])), dict(read_per_seed(ARM_RUNS[s]))
            if r in dc and r in da and len(dc[r]) == len(da[r]):
                diffs.append(st.mean([da[r][i] - dc[r][i] for i in range(len(dc[r]))]))
        if len(diffs) >= 2:
            A("      u%-3d Δ=%+.4f  SD=%.4f  (n=%d)" % (r, st.mean(diffs), st.stdev(diffs), len(diffs)))
    A("")

    # --- 3. 绝对水平 vs 专家 ---
    A("### 3. 绝对水平 vs 专家锚 0.6979220689")
    A("   ⚠ 这是 arm-vs-专家，硬件偏置**不相消**（专家是纯启发式、无 matmul）")
    A("     与上面 arm-vs-arm 的 Δ **不是同一类测量**")
    for lbl, src in (("对照 ent01", CTRL_RUNS), ("gae90", ARM_RUNS)):
        per_train = [st.mean(v) for s in SEEDS for r, v in read_per_seed(src[s]) if r == 30]
        if per_train:
            A("   %-10s u30 逐训练种子均值 %s  ⟹ 总均值 %.4f"
              % (lbl, " ".join("%.4f" % x for x in per_train), st.mean(per_train)))
    A("")
    left = [s for s in SEEDS if last_update(ARM_RUNS[s]) < UPDATES]
    if left:
        A("   ⚠ 未到 u30 的臂：%s" % ", ".join("%s=u%d" % (ARM_RUNS[s], last_update(ARM_RUNS[s])) for s in left))
    return "\n".join(L)


# ---------------------------------------------------------------- 主流程

def main():
    log("gae90 n=5 链启动 ｜ 队列 %d 条 %s" % (len(QUEUE), [q[0] for q in QUEUE]))
    log("  门要求（放行**下一条**）：可用 − 待涨 − need(剩余条数) >= %.1f" % FLOOR)
    log("  need(n_left) = (n_left−1)×%.0f + %.0f  ⟹ 起一条就降一条" % (STEADY, FLOOR))
    log("  ★ 为什么不写成 4×%.0f+%.0f=%.0f 的常量：已起臂的爬坡量已计入「待涨」，"
        % (STEADY, FLOOR, len(QUEUE) * STEADY + FLOOR))
    log("    常量写法会把它**算两遍** ⟹ 余量永远差 %.0fG ⟹ 剩下的臂一辈子起不来"
        % STEADY)

    # 前置检查
    if not os.path.exists(os.path.join(ROOT, CKPT)):
        open(VERDICT, "w", encoding="utf-8").write("STATE=failed\nBC 起点不存在\n")
        log("!! BC 起点不存在，退出")
        return 1
    for _, cfg, _ in QUEUE:
        if not os.path.exists(os.path.join(ROOT, "configs", cfg)):
            open(VERDICT, "w", encoding="utf-8").write(
                "STATE=failed\n配置不存在: %s\n" % cfg)
            log("!! 配置不存在 %s，退出" % cfg)
            return 1

    # ---- 阶段 1：按门放行，逐条起 ----
    launched = []
    t0 = time.time()
    while len(launched) < len(QUEUE) and time.time() - t0 < 5 * 3600:
        n_left = len(QUEUE) - len(launched)
        ok, detail = gate(n_left)
        fs = fs_runs()
        log("  门：%s" % detail)
        log("  交叉验证：/proc 数到 %d 条 ｜ 文件系统数到 %d 条 %s"
            % (len(live_runs()), len(fs), fs if len(fs) <= 12 else "(略)"))
        # ★ 独立读数交叉验证：两边差 >1 说明有一边坏了，拒绝启动并吵
        if abs(len(live_runs()) - len(fs)) > 1:
            log("  !! 两个独立读数不一致（/proc %d vs 文件系统 %d）"
                " ⟹ 门的输入不可信，**拒绝启动**"
                % (len(live_runs()), len(fs)))
            time.sleep(120)
            continue
        if ok:
            run, cfg, seed = QUEUE[len(launched)]
            log("  → 放行，起 %s (cfg=%s seed=%d)" % (run, cfg, seed))
            p = launch(run, cfg, seed)
            good, msg = verify_launch(run, p)
            log(msg)
            if not good:
                open(VERDICT, "w", encoding="utf-8").write(
                    "STATE=failed\n%s 启动失败\n%s\n" % (run, msg))
                return 1
            launched.append(run)
        time.sleep(60)

    if len(launched) < len(QUEUE):
        log("!! 5 小时内只起了 %d/%d 条（内存一直不够）" % (len(launched), len(QUEUE)))
        # 起了一部分也继续等，能判多少判多少

    log("阶段 1 完成，已起 %s" % launched)

    # ---- 阶段 2：等全部到 u30 ----
    t1 = time.time()
    while time.time() - t1 < 12 * 3600:
        ks = {r: last_update(r) for r in ARM_RUNS.values()}
        done = sum(1 for s in SEEDS if last_update(CTRL_RUNS[s]) >= UPDATES
                   and last_update(ARM_RUNS[s]) >= UPDATES)
        log("  进度 %d/%d ｜ %s" % (done, len(SEEDS),
             " ".join("%s=u%d" % (r, k) for r, k in sorted(ks.items()))))
        if done == len(SEEDS):
            break
        time.sleep(180)
    else:
        open(VERDICT, "w", encoding="utf-8").write(
            "STATE=timeout\n12 小时内未全部到 u30\n")
        log("!! 超时，写 timeout 标记")
        return 3

    # ---- 阶段 3：判读 ----
    txt = verdict()
    with open(VERDICT, "w", encoding="utf-8") as f:
        f.write("STATE=done\n")
        f.write(txt + "\n")
    log("判读已写入 %s" % VERDICT)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
