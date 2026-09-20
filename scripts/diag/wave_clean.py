"""判决性对照：**无 BC 起点**（从零训练）。

### 为什么这条对照此前不存在，以及它为什么是判决性的

全库 46 条臂（`ls outputs/`）**全部**从同一个 BC checkpoint 出发：
    `Resumed from outputs/supervised_pg_phased/supervised_pg_phased_latest.pt`
（已逐臂 grep 核实）。**没有一条从零训**。

⟹ 所有既有臂在原理上都答不了这个问题：
    **RL 停在专家水平 0.70，是因为问题本身到 0.70，还是因为 BC 起点锁住了？**

2026-09-20 首测 BC 起点 = **0.6537**（显著低于专家 0.6979，Δ=−0.0442 t=−3.34）；
RL 30 轮抬到 0.698（+3.5 点，4/5 臂显著）。**这两件事合起来只有两种解释**：

  (i)  问题本身的天花板 ≈ 0.70，两条路径（从 BC 起 / 从零起）都收敛到它
  (ii) BC 模仿把策略**初始化在专家的吸引域**里 ⟹ 只能收敛到专家

预测不同：
  (i)  ⟹ from-scratch 臂 u30 **也落在 0.70 附近**（可能更慢，但同终点）
  (ii) ⟹ from-scratch 臂 u30 **与 0.70 显著不同**（也可能根本学不动）

### 本臂的唯一变量

**只有 `--checkpoint`**：不给。
其余逐项与 `ent01_rerun_s{42,43,44}` 相同（同 configs、同种子、同 8 线程、同节点）。

⚠ 不预判方向。**从零训可能因缺 BC 而学得很差**——那也是有信息的结果
（⟹ 说明 BC 起点是**必要的**，论文要把 BC 写成方法的一部分）。
所以判据是**三档**，不是单向。

用法：
    python3 -u wave_scratch.py --dry-run     # 只预检
    python3 -u wave_scratch.py               # 预检 + 起臂
"""
# ★★ 摘掉 sys.path[0]：服务器 `.tmp/` 有 547 个 .py（含 `types.py`）
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
LOGDIR = Path("/tmp/cleanlogs")

SEEDS = (42, 43, 44, 45, 46)
ARMS = [f"clean_s{s}" for s in SEEDS]
CTRL = [f"scratch_s{s}" for s in SEEDS]
CONFIGS = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml",
           "train_clean.yaml"]
TRAIN_PATTERN = "train_graph_mappo"

# 内存常数（GiB）；与本波同配置的实测值
PSS_PER_RUN = 25.0
FLOOR = 17.0
OMP_THREADS = 8


def sh(cmd, timeout=60):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def mem_available_gib():
    """★ meminfo 的 kB 实为 KiB（记忆 meminfo-kb-is-kib-not-gb）。"""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    raise RuntimeError("MemAvailable 读不到 ⟹ 不许兜底")


def live_names():
    """★ 用 comm 锚定真 python 进程，别用 pgrep -f（会把包装器也数进去）。"""
    r = sh(f"ps -eo comm,args | awk '$1 ~ /^python/ && /{TRAIN_PATTERN}/'")
    names = []
    for ln in r.stdout.splitlines():
        parts = ln.split()
        for i, p in enumerate(parts):
            if p == "--run-name" and i + 1 < len(parts):
                names.append(parts[i + 1])
                break
    return names


def live_pss_gib():
    """★ 用 PSS 不 RSS（RSS 重复计共享页）；锚定**父进程**（comm=python 且有
    train_graph_mappo + --run-name）。
    返回 [(run_name, pss_gib, update)]。"""
    r = sh("ps -eo pid,comm,args | awk '$2 ~ /^python/'")
    out = []
    for ln in r.stdout.splitlines():
        parts = ln.split(None, 2)
        if len(parts) < 3:
            continue
        pid, comm, args = parts
        if TRAIN_PATTERN not in args or "--run-name" not in args:
            continue
        toks = args.split()
        nm = toks[toks.index("--run-name") + 1] if "--run-name" in toks else "?"
        # PSS via smaps_rollup
        try:
            sm = Path(f"/proc/{pid}/smaps_rollup").read_text()
            pss_kb = 0
            for line in sm.splitlines():
                if line.startswith("Pss:"):
                    pss_kb = int(line.split()[1])
                    break
            pss = pss_kb / 1024.0 / 1024.0
        except Exception:
            pss = 0.0
        # 当前轮号（只读，缺就记 -1）
        upd = -1
        mf = OUT / nm / "metrics.jsonl"
        if mf.exists():
            last = None
            for line in mf.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    o = __import__("json").loads(line)
                except Exception:
                    continue
                if "update" in o:
                    last = o["update"]
            if last is not None:
                upd = int(last)
        out.append((nm, pss, upd))
    return out


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = d
    return out


def cfg_for(configs, run_name, seed=42, checkpoint=None):
    import argparse as ap
    from scripts.train.train_graph_mappo import build_config
    args = ap.Namespace(configs=configs, mode="random_episode", run_name=run_name,
                        num_updates=30, seed=seed, checkpoint=checkpoint, device="cpu")
    return flatten(build_config(args))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--num-updates", type=int, default=30)
    args = p.parse_args()

    print("=" * 96)
    print("clean 波：**拿掉专家先验特征**（relay_importance / req_hop / on_pending_path）")
    print("=" * 96)
    print(f"  臂      {' '.join(ARMS)}")
    print(f"  对照    {' '.join(CTRL)}（**从零训 + 全特征**，u30 已跑完）")
    print(f"  configs {' '.join(CONFIGS)}")
    print(f"  唯一变量 特征集（三项）—— 两侧都**不给 --checkpoint**（从零训）")
    print(f"  ⚠ edge_dim 44→39（变窄，补零救不回）⟹ 必须从零")
    print()

    fails = []

    # ---------- 1. 对照臂的起点（本臂**不给**，但对照需要；缺失不致命） ----------
    ck = REPO / "outputs/scratch_s42/checkpoint_final.pt"
    print(f"  对照臂 scratch_* 已跑完 ⟹ 本波不依赖 BC 起点")
    print(f"  （本臂与对照都**不给 --checkpoint**，从零训）")

    # ---------- 2. 工作区（.tmp 的改动不算，训练路径不含 .tmp） ----------
    r = sh(f"cd {REPO} && git status --porcelain")
    dirty = [ln for ln in r.stdout.splitlines()
             if ln.strip() and ".tmp/" not in ln]
    head = sh(f"cd {REPO} && git rev-parse --short HEAD").stdout.strip()
    if dirty:
        print(f"★ 工作区有非 .tmp 改动 ⟹ 跑的代码无法与提交对应：")
        for ln in dirty:
            print(f"    {ln}")
        fails.append("工作区不干净")
    else:
        print(f"  ✓ 训练路径干净  HEAD={head}")

    # ---------- 3. ★★ 幂等与去重：问进程表，不问账本 ----------
    names = live_names()
    print(f"\n  进程表里在跑的 run（comm 锚定）：{len(names)} 条")
    for n in names:
        print(f"      {n}")
    clash = [a for a in ARMS if a in names]
    if clash:
        fails.append(f"已有同名臂在跑 ⟹ 会双写：{clash}")

    # ---------- 4. outputs 目录不得已存在（防续写混入） ----------
    for a in ARMS:
        d = OUT / a
        if d.exists():
            existing = list(d.glob("metrics.jsonl"))
            if existing and existing[0].stat().st_size > 0:
                fails.append(f"outputs/{a} 已存在且有数据 ⟹ 会追加混入")

    # ---------- 5. ★★ 唯一变量自证：差异集合必须 = 三项开关(+派生的两维) ----------
    print("\n  唯一变量核对（走 build_config()，不读 yaml）")
    try:
        ctrl = cfg_for(CONFIGS[:-1], "probe_ctrl", checkpoint=None)
        arm = cfg_for(CONFIGS, "probe_arm", checkpoint=None)
    except Exception as e:
        fails.append(f"build_config() 构造失败: {type(e).__name__}: {e}")
        ctrl = arm = None
    EXPECTED = {
        "features.edge.include_relay_importance",
        "features.edge.include_req_hop",
        "features.edge.include_on_pending_path",
        # 前两项的**派生**结果（维度由开关算出，不是额外变量）
        "features.dims.edge_dim_resolved",
        "features.dims.physical_edge_dim_resolved",
    }
    if ctrl and arm:
        ignore = {"project.run_name"}
        diffs = {k: (ctrl.get(k, "<缺>"), arm.get(k, "<缺>"))
                 for k in set(ctrl) | set(arm)
                 if k not in ignore and ctrl.get(k, "<缺>") != arm.get(k, "<缺>")}
        print(f"    基线字段数 {len(ctrl)}；差异字段 {len(diffs)} 个（期望 {len(EXPECTED)}）")
        for k, (a, b) in sorted(diffs.items()):
            mark = "  ← 预期" if k in EXPECTED else "  ★★ 意外差异"
            print(f"      {k}: {a} → {b}{mark}")
        if set(diffs) != EXPECTED:
            fails.append(f"差异集合 {sorted(diffs)} != 预期 {sorted(EXPECTED)}")
        # 维度必须变小
        ed_b = ctrl.get("features.dims.edge_dim_resolved")
        ed_a = arm.get("features.dims.edge_dim_resolved")
        print(f"    edge_dim {ed_b} → {ed_a}"
              f"  {'✓ 变小 ⟹ 必须从零训' if ed_a and ed_b and ed_a < ed_b else '★ 未变小'}")
        if not (ed_a and ed_b and ed_a < ed_b):
            fails.append("edge_dim 没有变小 ⟹ 特征没真正关掉")

    # ---------- 6. 内存门：三量都打出来（★ 待涨量按实测 PSS 与实测轮号推） ----------
    avail = mem_available_gib()
    live = live_pss_gib()
    print(f"\n  在跑的 run 实测 PSS（{len(live)} 条）：")
    warm = 0.0
    for nm, pss, upd in live:
        # 稳态按 25.0 GiB 算；当前轮号越低，待涨越多
        # u1≈9G，u15+≈25G ⟹ 线性插值到 u15，之后记 0
        if upd < 0:
            est_now = pss
        elif upd >= 15:
            est_now = max(pss, PSS_PER_RUN)
        else:
            est_now = 9.0 + (PSS_PER_RUN - 9.0) * (upd / 15.0)
        delta = max(0.0, PSS_PER_RUN - est_now)
        warm += delta
        print(f"      {nm:<20} PSS={pss:5.1f} GiB  u={upd:<3} 待涨={delta:5.1f}")
    need = PSS_PER_RUN * len(ARMS)
    print(f"\n  内存门（三量）")
    print(f"    MemAvailable            {avail:.1f} GiB")
    print(f"    在跑 {len(live)} 条的待涨量       −{warm:.1f} GiB")
    print(f"    本批 {len(ARMS)} 条稳态需求      −{need:.1f} GiB")
    residual = avail - warm - need
    print(f"    ⟹ 剩余                   {residual:.1f} GiB   （地板 {FLOOR}）")
    if residual < FLOOR:
        fails.append(f"内存不足：剩余 {residual:.1f} < 地板 {FLOOR}")

    # ---------- 汇总 ----------
    print("\n" + "=" * 96)
    if fails:
        print("★ 预检未过 ⟹ 一条都不起：")
        for f in fails:
            print(f"    ✗ {f}")
        return 2
    print("✓ 预检全过")

    if args.dry_run:
        print("\n--dry-run：不启动。")
        return 0

    LOGDIR.mkdir(parents=True, exist_ok=True)
    print(f"\n启动 {len(ARMS)} 条臂，日志 → {LOGDIR}")
    launched = []
    for arm, seed in zip(ARMS, SEEDS):
        log = LOGDIR / f"{arm}.log"
        cmd = (f"cd {REPO} && OMP_NUM_THREADS={OMP_THREADS} MKL_NUM_THREADS={OMP_THREADS} "
               f"setsid nohup {PY} -u scripts/train/train_graph_mappo.py "
               f"--configs {' '.join(CONFIGS)} "
               f"--seed {seed} --num-updates {args.num_updates} "
               f"--run-name {arm} > {log} 2>&1 < /dev/null &")
        print(f"  --- 启动 {arm} (seed {seed}) ---")
        sh(cmd, timeout=25)
        launched.append(arm)
        time.sleep(2)

    # ---------- 7. 启动后验证（失败必须吵） ----------
    print("\n  启动后验证（等 20s）...")
    time.sleep(20)
    now = live_names()
    ok = []
    for a in launched:
        if a in now:
            log = LOGDIR / f"{a}.log"
            txt = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
            bad = "Traceback" in txt or "Error" in txt
            if bad:
                print(f"    ✗ {a}: 进程在，但日志有 Traceback/Error")
                for ln in txt.splitlines()[-8:]:
                    print(f"        {ln}")
            else:
                print(f"    ✓ {a}: 进程在，无 traceback")
                ok.append(a)
        else:
            print(f"    ✗ {a}: **进程不在** ⟹ 启动失败")
            log = LOGDIR / f"{a}.log"
            if log.exists():
                for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()[-10:]:
                    print(f"        {ln}")

    print(f"\n  ★ 实际起来 {len(ok)}/{len(launched)} 条")
    if len(ok) < len(launched):
        print("  ★★ 有臂没起来 —— 见上面的日志尾。**不要**当作'还在跑'。")
        return 3
    print("\n  验证模式（判据先写死）：")
    print("    ① 从零臂 u30 ≈ 0.70 且与对照配对不显著 ⟹ 天花板是**问题本身**")
    print("    ② 从零臂 u30 显著低于对照         ⟹ BC 起点**必要**，写进方法")
    print("    ③ 从零臂 u30 显著高于对照         ⟹ BC 起点是**枷锁**，大发现")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
