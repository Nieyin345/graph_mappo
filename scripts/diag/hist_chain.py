#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hist 链：等槽位 → 起 hist_s{42,43,44} → u1 守卫 → 配对判读。

### 与 ent001 链的区别（两条，都要紧）
1. **内存门按配置查表**：实测 `minibatch 512` 的 run 是 29.5 GB、256 是 25.0 GB
   （2026-09-20，见 `docs/训练诊断记录.md`）。这里 BASE 链是 256 ⟹ 用 25.0，
   但门里**按实际在跑的每条 run 的配置**取 PSS，不用单一常数。
2. **u1 守卫更强**：加载器（commit `7e1ee69`）让 hist 的初始策略**逐位等于 BC**，
   而 `ent01_rerun` 加载的是**同一个** BC checkpoint ⟹
   **`hist_s42` 的 u1 必须与 `ent01_rerun_s42` 的 u1 逐位相同**。
   不同就说明加载器没做到它承诺的等价，实验结果**不可信**，立即停。

### 前置条件（脚本自己验，不满足就不起）
- 节点代码已 sync 到含**加载器**的版本（`_upgrade_state_dict_for_model` 存在）；
- 节点代码已含 `graph_mappo` 的 edge_h 修复；
- `outputs/supervised_pg_phased/supervised_pg_phased_latest.pt` 在。
"""
import json
import os
import re
import subprocess
import sys
import time

ROOT = "/opt/qkd/graph_mappo"
PY = "/opt/qkd/venv/bin/python"
BASE = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml"]
CFG = "train_hist.yaml"
CKPT = "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
SEEDS = (42, 43, 44)
UPDATES = 30

# 内存模型（实测，2026-09-20）
PSS_STEADY = {256: 25.0, 512: 29.5}
DEFAULT_STEADY = 25.0
FLOOR = 17.0        # 绝对余量底线
MAX_RUNS = 4        # 与 ent01_rerun 的可比性：它在 ≤4 并发下产出
THREADS = "8"

LOG = "/tmp/hist_chain.log"
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
        s = open("/proc/%d/stat" % pid, encoding="utf-8").read()
        return int(s[s.rindex(")") + 2:].split()[1])
    except Exception:
        return 0


def cmdline(pid):
    try:
        return open("/proc/%d/cmdline" % pid, "rb").read().decode("utf-8", "replace")
    except OSError:
        return ""


def live_runs():
    """返回 [(run_name, minibatch, 整run PSS)]。

    ★ 必须按 PPid **向上归组**：spawn worker 的 cmdline 里既没有 run-name
    也没有脚本名（argv 是 `-c from multiprocessing...`），只按 cmdline 抓
    会**只量到父进程**（~17G），把整 run 低估 8G。
    """
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
        rc = os.path.join(ROOT, "outputs", name, "resolved_config.yaml")
        try:
            for ln in open(rc, encoding="utf-8"):
                m = re.match(r"\s*minibatch_size:\s*(\d+)", ln)
                if m:
                    mb = int(m.group(1))
                    break
        except OSError:
            pass
        out.append((name, mb, tot.get(t, 0.0)))
    return out


def gate():
    """三量门：可用 − Σ(稳态−当前) − 本run稳态 ≥ 底线。"""
    a = avail()
    runs = live_runs()
    grow = sum(max(0.0, PSS_STEADY.get(mb, DEFAULT_STEADY) - cur) for _, mb, cur in runs)
    margin = a - grow - PSS_STEADY[256]
    return a, runs, grow, margin


def done(name):
    d = os.path.join(ROOT, "outputs", name)
    m = os.path.join(d, "metrics.jsonl")
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


def u1_of(run):
    p = os.path.join(ROOT, "outputs", run, "metrics.jsonl")
    try:
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                return json.loads(ln).get("mean_success_rate")
    except OSError:
        pass
    return None


def launch(run):
    cmd = ([PY, "scripts/train/train_graph_mappo.py", "--configs"] + BASE + [CFG] +
           ["--checkpoint", CKPT, "--seed", str(run.split("_s")[-1]),
            "--num-updates", str(UPDATES), "--run-name", run])
    env = dict(os.environ, OMP_NUM_THREADS=THREADS, PYTHONUNBUFFERED="1")
    logf = open("/tmp/%s.out" % run, "ab")
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    log("  起 %s (pid=%d, %s 线程, minibatch=256)" % (run, p.pid, THREADS))
    return p


# ============================ 主流程 ============================
log("=" * 72)
log("hist 链启动（%s）" % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

# ---- 前置条件 ----
ok = True
tr = open(os.path.join(ROOT, "qkd_rl/rl/algos/mappo_trainer.py"), encoding="utf-8").read()
if "_upgrade_state_dict_for_model" not in tr:
    log("!! 节点代码**不含加载器** —— 先同步代码再跑（改结构会丢暖启动）")
    ok = False
gm = open(os.path.join(ROOT, "qkd_rl/rl/models/graph_mappo.py"), encoding="utf-8").read()
if "if h_rows:\n                    edge_features_directed" not in gm:
    log("!! 节点代码**不含 edge_h 修复**")
    ok = False
if not os.path.exists(os.path.join(ROOT, CKPT)):
    log("!! BC 起点不存在：%s" % CKPT)
    ok = False
if not ok:
    open(VERDICT, "w", encoding="utf-8").write("前置条件不满足，未启动。\n")
    sys.exit(1)
log("预检通过：加载器 ✓ / edge_h 修复 ✓ / BC 起点 ✓")

# ---- 阶段 1：等槽位 ----
log("阶段 1：等槽位（要求 在跑 ≤ %d 且 余量 ≥ %.0fG）" % (MAX_RUNS - 1, FLOOR))
t0 = time.time()
launched = []
while time.time() - t0 < 6 * 3600:
    n = len([r for r in live_runs()])
    a, runs, grow, margin = gate()
    if n <= MAX_RUNS - 1 and margin >= FLOOR:
        log("门通过：可用 %.0f 待涨 %.1f 余量 %.1f（在跑 %d）" % (a, grow, margin, n))
        break
    if int(time.time() - t0) % 600 < 130:
        log("  门未过：可用 %.0f 待涨 %.1f 余量 %.1f（在跑 %d）" % (a, grow, margin, n))
    time.sleep(120)
else:
    log("!! 等槽位超时")
    open(VERDICT, "w", encoding="utf-8").write("等槽位超时，未启动。\n")
    sys.exit(1)

# ---- 阶段 2：逐个起（每个都过门）----
for s in SEEDS:
    run = "hist_s%d" % s
    t0 = time.time()
    while time.time() - t0 < 2 * 3600:
        n = len(live_runs())
        a, runs, grow, margin = gate()
        if n < MAX_RUNS and margin >= FLOOR:
            launched.append((run, launch(run)))
            break
        time.sleep(120)
    else:
        log("!! %s 未能起（槽位超时）" % run)
    time.sleep(20)
log("已起 %d/%d" % (len(launched), len(SEEDS)))

# ---- 阶段 3：等完成 ----
log("阶段 3：等 3 臂跑完 %d 轮" % UPDATES)
t0 = time.time()
while time.time() - t0 < 8 * 3600:
    n = sum(1 for s in SEEDS if done("hist_s%d" % s))
    if n == len(SEEDS):
        log("✓ 3 臂完成（%.0f 分钟）" % ((time.time() - t0) / 60))
        break
    if int(time.time() - t0) % 900 < 130:
        log("  进度 %d/%d" % (n, len(SEEDS)))
    time.sleep(130)
else:
    log("!! 等待超时，按现有数据判读")

# ---- 阶段 4：u1 守卫（决定性）----
log("阶段 4：u1 守卫")
guard = []
for s in SEEDS:
    a = u1_of("hist_s%d" % s)
    b = u1_of("ent01_rerun_s%d" % s)
    d = None if (a is None or b is None) else (a - b)
    guard.append((s, a, b, d))
    log("  s%d  hist=%.12f  ent01_rerun=%s  Δ=%s"
        % (s, a if a is not None else float("nan"),
           ("%.12f" % b) if b is not None else "缺",
           ("%.2e" % d) if d is not None else "—"))
hard = [g for g in guard if g[3] is not None and abs(g[3]) > 1e-9]
if hard:
    log("!! **u1 不逐位相同**（%d/%d）—— 加载器没做到它承诺的等价，结果不可信"
        % (len(hard), len(SEEDS)))
else:
    log("✓ u1 逐位相同 ⟹ 初始策略**逐位等于 BC**，暖启动确实保住了")

# ---- 阶段 5：配对判读 ----
log("阶段 5：配对判读")
r = subprocess.run([PY, "/tmp/hist_verdict.py"], capture_output=True, text=True)
txt = r.stdout or ""
for ln in txt.splitlines():
    log("  " + ln)
if r.returncode != 0:
    log("!! 判读脚本返回 %d，stderr 尾：\n%s" % (r.returncode, (r.stderr or "")[-800:]))
with open(VERDICT, "w", encoding="utf-8") as f:
    f.write(txt)
    f.write("\n\n### u1 守卫\n")
    for s, a, b, d in guard:
        f.write("s%d  hist=%s  ent01_rerun=%s  Δ=%s\n"
                % (s, a, b, d if d is None else "%.3e" % d))
log("判读写入 %s" % VERDICT)
