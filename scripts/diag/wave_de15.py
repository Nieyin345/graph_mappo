"""把对照臂 `ent01_rerun` 补到 n=15（起 s47..s56）。

## 为什么做这个（而不是继续找旋钮）

今晚测了 **7 条结构旋钮**，模式极清晰：
  · 破坏性改动有效应（demand_edge −6.5 点、从零 −8.6 点，都显著）
  · **增益性改动全是 +0.5 点量级**（cmax/layers4/storage5），且**都在可检测效应之下**

算术（BC 族 Δ_s SD ≈ 0.010）：
    要测出 0.005（0.5 点）⟹ n > 31     ← ②③④ 都是这个量级，**不可测**
    要测出 0.0075（0.75 点）⟹ n > 14
    要测出 0.020（2 点）⟹ n > 2

⟹ 继续找第 8、第 9 条旋钮大概率还是"测不出"，而每次 5 种子 × 1 小时。

**更值得的是把 n 提到 15** —— 这能直接回答核心问题：
> **「RL 与专家到底差多少？」**
> 现状 n=5：Δ=−0.0075，可检测 0.0121 ⟹ **连 0.75 点的差距都测不出**
> n=15：可检测 ≈0.0075 ⟹ **能给出判决**

且这是**通用投资**：之后所有实验的对照分辨率都随之提升。

## 分批（内存约束）

10 条 × 25 GiB = 250 GiB > 可用（246.9）。
按预算分批：先 8 条（200 GiB，余 46.9），跑完再起 2 条。
"""
import os as _os
import sys as _sys
_here = _sys.path[0] if _sys.path else ""
if _here and _here not in ("", "."):
    _sys.path[:] = [p for p in _sys.path if p != _here]

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/opt/qkd/graph_mappo")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PY = "/opt/qkd/venv/bin/python"
OUT = REPO / "outputs"
LOGDIR = Path("/tmp/de15logs")
TRAIN_PATTERN = "train_graph_mappo"

# 目标：ent01_rerun 补到 n=15（现有 42..46，新起 47..56）
SEEDS = tuple(range(51, 57))  # demandedge 已有 42..50，补 51..56 到 n=15
ARMS = [f"demandedge_s{s}" for s in SEEDS]
CONFIGS = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml", "train_demandedge.yaml"]
CKPT = "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
EXPECTED_DIFF = {}          # 与对照**同配置** ⟹ 差异字段应为空

PSS_PER_RUN = 25.0
FLOOR = 17.0
RESERVE = 25.0
OMP_THREADS = 8


def sh(cmd, timeout=60):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def mem_available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    raise RuntimeError("MemAvailable 读不到 ⟹ 不许兜底")


def live_names():
    r = sh(f"ps -eo comm,args | awk '$1 ~ /^python/ && /{TRAIN_PATTERN}/'")
    names = []
    for ln in r.stdout.splitlines():
        parts = ln.split()
        for i, p in enumerate(parts):
            if p == "--run-name" and i + 1 < len(parts):
                names.append(parts[i + 1])
                break
    return names


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--num-updates", type=int, default=30)
    args = p.parse_args()

    print("=" * 96)
    print("补 demandedge 到 n=15：demandedge_s51..s56")
    print("=" * 96)
    print(f"  新起 {len(ARMS)} 条（现有 42..50 共 9 条 ⟹ 补 6 条到 n=15）")
    print(f"    目标：把 BC 族的可检测效应从 0.0121 压到 ~0.0075")
    print(f"    ⟹ 才能判「RL 与专家是否相等」")
    print()

    fails = []
    ck = REPO / CKPT
    if not ck.exists():
        fails.append(f"BC 起点不存在：{ck}")
    else:
        print(f"  ✓ 起点存在  {ck.name}  {ck.stat().st_size/1e6:.1f} MB")

    r = sh(f"cd {REPO} && git status --porcelain")
    dirty = [ln for ln in r.stdout.splitlines()
             if ln.strip() and ".tmp/" not in ln]
    head = sh(f"cd {REPO} && git rev-parse --short HEAD").stdout.strip()
    if dirty:
        print("★ 工作区有非 .tmp 改动 ⟹ 一条都不起：")
        for ln in dirty:
            print(f"    {ln}")
        fails.append("工作区不干净")
    else:
        print(f"  ✓ 训练路径干净  HEAD={head}")

    names = live_names()
    print(f"\n  进程表里在跑的 run：{len(names)} 条  {names}")
    clash = [a for a in ARMS if a in names]
    if clash:
        fails.append(f"已有同名臂在跑 ⟹ 会双写：{clash}")

    for a in ARMS:
        d = OUT / a
        if d.exists():
            ex = list(d.glob("metrics.jsonl"))
            if ex and ex[0].stat().st_size > 0:
                fails.append(f"outputs/{a} 已存在且有数据 ⟹ 会追加混入")

    # 分批
    avail = mem_available_gib()
    room = avail - RESERVE
    max_new = max(0, int(room // PSS_PER_RUN))
    batch, rest = ARMS[:max_new], ARMS[max_new:]
    print(f"\n  内存门与分批")
    print(f"    MemAvailable        {avail:.1f} GiB")
    print(f"    预留                −{RESERVE:.1f} GiB")
    print(f"    每条                −{PSS_PER_RUN:.1f} GiB")
    print(f"    ⟹ 本批可起          {max_new} 条")
    print(f"    第一批 ({len(batch)}): {' '.join(batch) if batch else '(无)'}")
    print(f"    后续批次 ({len(rest)}): {' '.join(rest) if rest else '(无)'}")

    print("\n" + "=" * 96)
    if fails:
        print("★ 预检未过 ⟹ 一条都不起：")
        for f in fails:
            print(f"    ✗ {f}")
        return 2
    if not batch:
        print("★ 内存不足以起任何一条 ⟹ 等别的波跑完")
        return 2
    print("✓ 预检全过")
    if args.dry_run:
        print("\n--dry-run：不启动。")
        return 0

    LOGDIR.mkdir(parents=True, exist_ok=True)
    jobs = [(a, a.split("_s")[-1]) for a in batch]
    assert len(jobs) == len(batch)

    print(f"\n启动第一批 {len(jobs)} 条，日志 → {LOGDIR}")
    launched = []
    for arm, seed in jobs:
        log = LOGDIR / f"{arm}.log"
        cmd = (f"cd {REPO} && ulimit -n 65536 && "
               f"OMP_NUM_THREADS={OMP_THREADS} MKL_NUM_THREADS={OMP_THREADS} "
               f"setsid nohup {PY} -u scripts/train/train_graph_mappo.py "
               f"--configs {' '.join(CONFIGS)} "
               f"--checkpoint {CKPT} "
               f"--seed {seed} --num-updates {args.num_updates} "
               f"--run-name {arm} > {log} 2>&1 < /dev/null &")
        print(f"  --- 启动 {arm} (seed {seed}) ---")
        sh(cmd, timeout=25)
        launched.append(arm)
        time.sleep(2)

    print("\n  启动后验证（等 25s）...")
    time.sleep(25)
    now = live_names()
    ok = []
    for a in launched:
        log = LOGDIR / f"{a}.log"
        txt = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        bad = "Traceback" in txt
        if a in now and not bad:
            print(f"    ✓ {a}")
            ok.append(a)
        else:
            print(f"    ✗ {a}: {'进程不在' if a not in now else '日志有 Traceback'}")
            for ln in txt.splitlines()[-6:]:
                print(f"        {ln}")
    print(f"\n  ★ 起来 {len(ok)}/{len(launched)}   剩余 {len(rest)} 条待下批")
    return 0 if len(ok) == len(launched) else 3


if __name__ == "__main__":
    raise SystemExit(main())
