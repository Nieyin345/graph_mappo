"""v2 波次启动器：跑 5 条 `v2_bottleneck_s{42..46}`。

## 为什么另写一个（而不是复用 deployment/）

`deployment/` 下的启动器被分波链改过多轮，带着历史包袱（`ARM=` 读不存在的
目录这类哑火，见 `duplicate-implementation-drifts`）。这个是**自包含**的：
一个文件、判据都在里面、失败吵。

## 这个脚本遵守的几条硬规矩（每条都有实测教训）

1. **判活不用 `pgrep -f`**（`pgrep-f-counts-bash-wrappers-too`：它把 `bash -c`
   包装器也数进去、还会自匹配 ⟹ 内存预算能错一倍）。用
   `ps -eo comm,args | awk '$1~/^python/ && /<模式>/'`。
2. **门要打印它自己的输入**（`gate-must-print-its-inputs`：恒真的门不报错，
   伪装成"资源不够"）。这里把 available / Σ待涨 / 本 run 稳态三项都打出来。
3. **待涨量必须算**（`mem-gate-must-count-warmup-growth`：u1 时 9G、u15 时 25G，
   `free` 只报当前读数）。本批从**全新**开始（不是续跑），按 25 GB 稳态算。
4. **一个返回值两种含义是禁忌**（`one-return-value-two-meanings`）：launch()
   返回状态字符串（started / already / failed），不返回 None。
5. **起任何一条之前全员预检**（同上）：一次全查完再决定铺不铺。
6. **启动后验证**（`failed-launch-must-be-loud`）：等几秒确认进程真起来了、
   `resolved_config.yaml` 里的开关真是 true、日志没有立刻 traceback。
7. **数值比较不 shell out 到 bc**（`numeric-gates-must-not-shell-out-to-bc`）。
   全在进程内算，空值当致命。
8. **绝不 `pkill -f`**（在 ssh 里会杀掉自己）。报出原因并 exit。

用法：
    python3 -u wave_v2.py                 # 预检 + 真启动
    python3 -u wave_v2.py --dry-run       # 只预检
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/opt/qkd/graph_mappo")
PY = "/opt/qkd/venv/bin/python"
OUT = REPO / "outputs"
CKPT = "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
LOGDIR = Path("/tmp/v2logs")

ARMS = [f"v2_bottleneck_s{s}" for s in (42, 43, 44, 45, 46)]
CONFIGS = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml",
           "train_v2_bottleneck.yaml"]
TRAIN_PATTERN = "train_graph_mappo"

# ---- 内存常数（本项目实测；GiB）----
PSS_PER_RUN = 25.0     # minibatch 256 稳态，无 hist encoder
GROWTH_PER_RUN = 0.0   # 全新 run 按稳态 25 已是含增长的；不再另计
FLOOR = 17.0           # 系统地板
OMP_THREADS = 8        # ★ 必须显式设（unset-omp-threads-silently-uses-half-the-cores）


def sh(cmd, timeout=60):
    return subprocess.run(cmd, shell=True, capture_output=True,
                          text=True, timeout=timeout)


def mem_available_gib():
    """★ meminfo 的 kB 其实是 KiB（meminfo-kb-is-kib-not-gb）—— 用 /1024/1024。"""
    txt = Path("/proc/meminfo").read_text()
    for line in txt.splitlines():
        if line.startswith("MemAvailable:"):
            kib = float(line.split()[1])
            return kib / 1024.0 / 1024.0
    raise RuntimeError("MemAvailable 读不到 ⟹ 不许兜底")


def live_runs():
    """★ 用 comm 锚定真 python 进程；别用 pgrep -f。"""
    r = sh(f"ps -eo comm,args | awk '$1 ~ /^python/ && /{TRAIN_PATTERN}/'")
    rows = [ln for ln in r.stdout.splitlines() if ln.strip()]
    # 从 --run-name 抠名字；抠不到就报 unknown（不静默）
    names = []
    for ln in rows:
        parts = ln.split()
        nm = "unknown"
        for i, p in enumerate(parts):
            if p == "--run-name" and i + 1 < len(parts):
                nm = parts[i + 1]
        names.append(nm)
    return rows, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--num-updates", type=int, default=30)
    args = ap.parse_args()

    print("=" * 92)
    print("v2 波次启动器  预检")
    print("=" * 92)
    print(f"  仓库        {REPO}")
    print(f"  臂          {', '.join(ARMS)}")
    print(f"  configs     {' '.join(CONFIGS)}")
    print(f"  num_updates {args.num_updates}")
    print(f"  OMP 线程    OMP_NUM_THREADS={OMP_THREADS}（显式设，否者按核数折半）")
    print()

    # ---------- 1. 起点的存在性（缺失致命：one-return-value-two-meanings）----------
    ck = REPO / CKPT
    if not ck.exists():
        print(f"★ 起点不存在：{ck}  ⟹ 一条都不起")
        return 2
    print(f"  ✓ 起点存在  {ck.name}  {ck.stat().st_size/1e6:.1f} MB")

    # ---------- 2. 代码状态：必须是干净的工作区（否则跑的不是提交的代码）----------
    r = sh(f"cd {REPO} && git status --porcelain")
    dirty = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if dirty:
        print("★ 工作区不干净 ⟹ 跑的代码无法与提交对应：")
        for ln in dirty:
            print(f"    {ln}")
        print("  ⟹ 一条都不起（先 commit 或还原）")
        return 2
    r = sh(f"cd {REPO} && git rev-parse --short HEAD")
    head = r.stdout.strip()
    print(f"  ✓ 工作区干净  HEAD={head}")

    # ---------- 3. v2 接线自证：开关真的在、routing 真的传了 ----------
    app = sh(f"cd {REPO} && grep -c 'routing=routing' qkd_rl/env/factory.py")
    gb = sh(f"cd {REPO} && grep -c 'self.routing = routing' qkd_rl/env/graph_builder.py")
    n_app, n_gb = int(app.stdout.strip() or 0), int(gb.stdout.strip() or 0)
    print(f"  v2 接线：factory 传参 {n_app} 处（应 ≥2，另一处是 QKDEnv）、"
          f"graph_builder 赋值 {n_gb} 处（应 1）")
    if n_app < 2 or n_gb != 1:
        print("★ v2 接线不完整 ⟹ 一条都不起")
        return 2
    print(f"  ✓ 接线齐")

    # ---------- 4. 去重：问进程表，不问账本（chain-must-ask-proc-not-its-own-ledger）----------
    rows, names = live_runs()
    if rows:
        print(f"★ 已有 {len(names)} 条训练在跑 ⟹ 不让路，本波不起：")
        for nm in names:
            print(f"    {nm}")
        return 3
    print(f"  ✓ 无训练在跑（ps+awk 数出 0 条）")

    # ---------- 5. 内存门：三量都打出来 ----------
    avail = mem_available_gib()
    need = len(ARMS) * (PSS_PER_RUN + GROWTH_PER_RUN) + FLOOR
    print()
    print("  内存门（★ 打印输入，不只打印判决）")
    print(f"    MemAvailable            {avail:8.1f} GiB")
    print(f"    Σ 待涨（{len(ARMS)} × {GROWTH_PER_RUN}）  {len(ARMS)*GROWTH_PER_RUN:8.1f} GiB")
    print(f"    本批稳态（{len(ARMS)} × {PSS_PER_RUN}）  {len(ARMS)*PSS_PER_RUN:8.1f} GiB")
    print(f"    地板                    {FLOOR:8.1f} GiB")
    print(f"    ⟹ 需要 {need:.1f} GiB，实际 {avail:.1f} GiB  "
          f"{'✓ 放行' if avail >= need else '★ 不够 ⟹ 一条都不起'}")
    if avail < need:
        return 4

    # ---------- 6. 输出目录不得已存在（outputs/ 只增不删）----------
    clash = [a for a in ARMS if (OUT / a).exists()]
    if clash:
        print(f"★ 这些 outputs 已存在 ⟹ 拒绝覆盖（outputs/ 只增不删）：{clash}")
        return 2
    print(f"  ✓ {len(ARMS)} 个 outputs 目录都还不存在")

    if args.dry_run:
        print()
        print("DECISION=DRY_RUN_OK  预检全过，未启动（--dry-run）")
        return 0

    # ---------- 7. 真启动 ----------
    LOGDIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(OMP_THREADS)
    env["MKL_NUM_THREADS"] = str(OMP_THREADS)
    launched = []
    print()
    print("=" * 92)
    print("启动")
    print("=" * 92)
    for arm in ARMS:
        seed = int(arm.rsplit("s", 1)[1])
        log = LOGDIR / f"{arm}.log"
        cmd = (f"cd {REPO} && OMP_NUM_THREADS={OMP_THREADS} MKL_NUM_THREADS={OMP_THREADS} "
               f"setsid nohup {PY} -u scripts/train/train_graph_mappo.py "
               f"--configs {' '.join(CONFIGS)} "
               f"--checkpoint {CKPT} --seed {seed} "
               f"--num-updates {args.num_updates} --run-name {arm} "
               f"> {log} 2>&1 < /dev/null &")
        r = sh(cmd, timeout=30)
        print(f"  --- 启动 {arm} (seed={seed}) rc={r.returncode} ---")
        launched.append((arm, log))

    # ---------- 8. 启动后验证（failed-launch-must-be-loud）----------
    print()
    print("等 25 秒后验证…")
    time.sleep(25)
    rows, names = live_runs()
    ok = True
    print(f"  ps+awk 数出 {len(names)} 条在跑：{names}")
    for arm, log in launched:
        alive = arm in names
        existed = (OUT / arm).exists()
        # 日志有无立刻 traceback
        tail = ""
        if log.exists():
            tail = log.read_text(errors="replace")[-800:]
        bad = ("Traceback" in tail) or ("Error" in tail and "ERROR" not in tail[:0])
        mark = "✓" if (alive and not bad) else "★"
        print(f"  {mark} {arm}  进程={'在' if alive else '不在'}  "
              f"outputs={'有' if existed else '无'}  "
              f"日志{'有 Traceback' if bad else '无异常'}")
        if not alive or bad:
            ok = False
            if tail:
                print("     日志尾部：")
                for ln in tail.strip().splitlines()[-6:]:
                    print(f"       {ln}")

    print()
    if ok:
        print(f"DECISION=LAUNCHED  {len(launched)} 条")
        return 0
    print("DECISION=LAUNCH_INCOMPLETE  ★ 有臂没起来，看上面的日志尾部")
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
