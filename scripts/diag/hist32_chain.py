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

ROOT = "/opt/qkd/graph_mappo"
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

# ★★ 命名约定必须**统一**：本文件的 read() 拼 `outputs/<基名>_s<种子>`，
#    而 u1_of() 拼 `outputs/<全名>`。原值 ["hist32v3_s42", "hist32_s43", "hist32_s44"]
#    **混了两种约定** ⟹ read() 对三条臂**全部返回空** ⟹ 判读只会打印「数据不足」，
#    而那看起来像「还没跑完」，不像 bug。2026-09-20 修正。
#    s42 原设计是**从 outputs/hist32v3_s42/checkpoint_update_000010.pt 续跑 20 轮**，
#    与 s43/s44（从 BC 新起 30 轮）不对称；那个 u10 checkpoint 已随旧节点 clnode316
#    丢失 ⟹ 现在三条统一为**从 BC 新起 30 轮**，更干净，但改变了原设计意图。
ARM_RUNS = ["hist32_s42", "hist32_s43", "hist32_s44"]
# TODO 已由 .tmp/launch_wave263.py 接管（它用内存门统一排队，不在这里重复起臂）
TODO = []
CTRL = "ent01_rerun"
SEEDS = (42, 43, 44)
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
    for d in glob.glob("/proc/[0-9]*"):
        try:
            with open(d + "/cmdline", "rb") as f:
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
    p = os.path.join(ROOT, "outputs", run, "metrics.jsonl")
    n = -1
    try:
        n = 0
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


# ============================ 判读 ============================
def read(run, seed):
    p = "%s/outputs/%s_s%d/metrics.jsonl" % (ROOT, run, seed)
    out = []
    if not os.path.exists(p):
        return out
    last = None
    for ln in open(p, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        o = json.loads(ln)
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
    p = "%s/outputs/%s/metrics.jsonl" % (ROOT, run)
    try:
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                return json.loads(ln).get("mean_success_rate")
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
    guard_ok = True
    for s in SEEDS:
        a = u1_of(ARM_RUNS[SEEDS.index(s)])
        b = u1_of("%s_s%d" % (CTRL, s))
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
        run = ARM_RUNS[SEEDS.index(s)]
        if read(run, s) and read(CTRL, s):
            avail_s.append(s)
    A("   可用训练种子：%s" % avail_s)
    if len(avail_s) < 2:
        A("   !! 数据不足，无法判读")
        return "\n".join(L)

    per_seed, nc = [], None
    for s in avail_s:
        run = ARM_RUNS[SEEDS.index(s)]
        Aps, Cps = plat_ps(read(run, s)), plat_ps(read(CTRL, s))
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
            run = ARM_RUNS[SEEDS.index(s)]
            Aa = {x[0]: x[1] for x in read(run, s)}
            Cc = {x[0]: x[1] for x in read(CTRL, s)}
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
                run = CTRL if nm == "ctrl" else ARM_RUNS[SEEDS.index(s)]
                for (uu, p, _, _) in read(run, s):
                    if uu == u:
                        vs.append(st.mean(p))
            row += "%9.4f" % st.stdev(vs) if len(vs) > 1 else "%9s" % "--"
        A(row)

    A("\n### 4. 绝对水平 vs 专家锚 %.4f（⚠ 臂 vs 专家有节点偏置不抵消）" % EXPERT)
    for nm, get in (("ctrl", lambda s: read(CTRL, s)),
                    ("hist32", lambda s: read(ARM_RUNS[SEEDS.index(s)], s))):
        vals = []
        for s in avail_s:
            r = get(s)
            if len(r) >= 2:
                vals.append(st.mean([st.mean(x[1]) for x in r[-2:]]))
        if vals:
            A("   %-10s 平台=%.4f  Δ(臂−专家)=%+.4f   逐种子 %s"
              % (nm, st.mean(vals), st.mean(vals) - EXPERT,
                 " ".join("%.4f" % v for v in vals)))

    A("\n### 5. ★ 吞吐代价（决定「值不值得继续投入」的一半）")
    A("   %-12s %10s %10s %10s" % ("臂", "rollout_s", "update_s", "合计"))
    tot = {}
    for nm, get in (("ctrl", lambda s: read(CTRL, s)),
                    ("hist32", lambda s: read(ARM_RUNS[SEEDS.index(s)], s))):
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
        n = sum(1 for r in ARM_RUNS if last_update(r) >= UPDATES)
        if n == len(ARM_RUNS):
            log("✓ 三臂全部到 u%d（%.0f 分钟）" % (UPDATES, (time.time() - t0) / 60))
            break
        if int(time.time() - t0) % 900 < 130:
            log("  进度 %d/%d (%s)" % (n, len(ARM_RUNS),
                 " ".join("%s=u%d" % (r, last_update(r)) for r in ARM_RUNS)))
        time.sleep(150)
    else:
        log("!! 等待超时，按现有数据判读")

    txt = verdict()
    for ln in txt.splitlines():
        log("  " + ln)
    n_done = sum(1 for r in ARM_RUNS if last_update(r) >= UPDATES)
    with open(VERDICT, "w", encoding="utf-8") as f:
        f.write("STATE=%s\n" % ("done" if n_done == len(ARM_RUNS) else "timeout"))
        f.write(txt)
    log("判读写入 %s" % VERDICT)


if __name__ == "__main__":
    main()
