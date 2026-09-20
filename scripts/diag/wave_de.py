"""cmax 波次启动器：critic 加一路 **max 池化**（`cmax_s{42..46}`）。

### 唯一变量（由 `build_config()` 自证，不读 yaml）

    对照：rl_algorithm + train_full_rl + train_ent01
    臂  ：上面 + train_cmax.yaml     （`model.critic.pool_include_max: true`）

差异集合**必须恰好** = `{"model.critic.pool_include_max"}`，否则一条都不起。

### 为什么是这条

本项目瓶颈是 **min 型**（`routing.py:237` `serve_now = min(hop_levels)`），
而 critic 只有 `.mean(dim=0)` 池化（全仓 grep `amax|scatter_reduce|max_pool` 零命中）
⟹ 单点瓶颈被稀释成 1/N，价值网络看不见它。actor 不池化，故主要受害的是 critic。

### 已做的极性自证（`probe_cmax_polarity.py`，全过）

  · false（默认）⟹ value_head 宽度 387，**与改动前逐位相同** ⟹ 46 条既有臂不受影响
  · true ⟹ 宽度 771，新增段**逐位等于 max**、且**不等于** mean（造反证）
  · critic value_head 本来每次运行就重置（`mappo_trainer.py:1691`）
    ⟹ 本改动不额外损失 BC 暖启动；actor 与 encoder 的权重保留

### 判据（预注册）

窗口 u25/u30；**族级**配对 n=5（df=4，临界 2.776）——**不是逐臂挑**
（`first-config-to-beat-the-expert` 的教训：高方差族里挑最大值必然"显著"）。

用法：
    python3 -u wave_cmax.py --dry-run
    python3 -u wave_cmax.py
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
LOGDIR = Path("/tmp/delogs")

SEEDS = (47, 48, 49, 50)
ARMS = [f"demandedge_s{s}" for s in SEEDS]
# @@ 2026-09-21：对照 ent01_rerun **只有 s42..46** ⟹ 扩到 n=9 必须**同时起对照**，
#    否则 s47..50 没有配对对象。CTRL 在本波里也要真起。
CONTROL_ARMS = [f"ent01_rerun_s{s}" for s in SEEDS]
LAUNCH = ARMS + CONTROL_ARMS
CTRL = [f"ent01_rerun_s{s}" for s in SEEDS]
CONFIGS = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml",
           "train_demandedge.yaml"]
TRAIN_PATTERN = "train_graph_mappo"
CKPT = "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"

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
    print("demand_edge 波次：model.mode mixed -> demand_edge")
    print("=" * 96)
    print(f"  臂      {' '.join(ARMS)}")
    print(f"  对照    {' '.join(CTRL)}（同配置、同 BC 起点、同种子、u30 已跑完）")
    print(f"  configs {' '.join(CONFIGS)}")
    print(f"  唯一变量 model.mode: 对照=mixed（缺省） / 臂=demand_edge\n  本波**同时起对照**（ent01_rerun 只有 s42..46，扩到 n=9 需补）")
    print(f"  两侧都给 --checkpoint（BC 暖启动）")
    print()

    fails = []

    # ---------- 1. 起点确实存在（对照要用；臂不给） ----------
    ck = REPO / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
    if not ck.exists():
        fails.append(f"对照起点不存在：{ck}")
    else:
        print(f"  ✓ 对照起点存在  {ck.name}  {ck.stat().st_size/1e6:.1f} MB")

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
    clash = [a for a in LAUNCH if a in names]
    if clash:
        fails.append(f"已有同名臂在跑 ⟹ 会双写：{clash}")

    # ---------- 4. outputs 目录不得已存在（防续写混入） ----------
    for a in LAUNCH:
        d = OUT / a
        if d.exists():
            existing = list(d.glob("metrics.jsonl"))
            if existing and existing[0].stat().st_size > 0:
                fails.append(f"outputs/{a} 已存在且有数据 ⟹ 会追加混入")

    # ---------- 5. ★★ 唯一变量自证：差异集合必须**恰好**是 pool_include_max ----------
    print("\n  唯一变量核对（走 build_config()，不读 yaml）")
    try:
        ctrl = cfg_for(CONFIGS[:-1], "probe_ctrl", checkpoint=str(ck))
        arm = cfg_for(CONFIGS, "probe_arm", checkpoint=str(ck))
    except Exception as e:
        fails.append(f"build_config() 构造失败: {type(e).__name__}: {e}")
        ctrl = arm = None
    EXPECTED = {"model.mode": "demand_edge"}
    if ctrl and arm:
        ignore = {"project.run_name"}
        diffs = {k: (ctrl.get(k, "<缺>"), arm.get(k, "<缺>"))
                 for k in set(ctrl) | set(arm)
                 if k not in ignore and ctrl.get(k, "<缺>") != arm.get(k, "<缺>")}
        print(f"    基线字段数 {len(ctrl)}；差异字段 {len(diffs)} 个（期望恰好 1）")
        for k, (a, b) in sorted(diffs.items()):
            mark = "  ← 预期" if EXPECTED.get(k) == b else "  ★★ 意外差异"
            print(f"      {k}: {a} → {b}{mark}")
        if set(diffs) != set(EXPECTED) or any(
                diffs[k][1] != v for k, v in EXPECTED.items()):
            fails.append(f"差异集合 {sorted(diffs)} != 预期 {sorted(EXPECTED)}")

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
    need = PSS_PER_RUN * len(LAUNCH)
    print(f"\n  内存门（三量）")
    print(f"    MemAvailable            {avail:.1f} GiB")
    print(f"    在跑 {len(live)} 条的待涨量       −{warm:.1f} GiB")
    print(f"    本批 {len(LAUNCH)} 条稳态需求      −{need:.1f} GiB")
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

    # ★ 先把启动清单**完整构造**出来（本次就是在这里炸过：构造写在循环里，
    #   4 条臂被拉起后父进程抛 NameError，臂跟着一起死）。
    jobs = [(a, a.split("_s")[-1], CONFIGS) for a in ARMS] + \
           [(a, a.split("_s")[-1], CONFIGS[:-1]) for a in CONTROL_ARMS]
    assert len(jobs) == len(LAUNCH), f"jobs({len(jobs)}) != LAUNCH({len(LAUNCH)})"

    LOGDIR.mkdir(parents=True, exist_ok=True)
    print(f"\n启动 {len(jobs)} 条（{len(ARMS)} 臂 + {len(CONTROL_ARMS)} 对照），日志 → {LOGDIR}")
    launched = []
    for arm, seed, cfgs in jobs:
        log = LOGDIR / f"{arm}.log"
        cmd = (f"cd {REPO} && OMP_NUM_THREADS={OMP_THREADS} MKL_NUM_THREADS={OMP_THREADS} "
               f"setsid nohup {PY} -u scripts/train/train_graph_mappo.py "
               f"--configs {' '.join(cfgs)} "
               + (f"--checkpoint {CKPT} " if CKPT else "")
               + f"--seed {seed} --num-updates {args.num_updates} "
               f"--run-name {arm} > {log} 2>&1 < /dev/null &")
        tag = "臂" if cfgs is CONFIGS else "对照"
        print(f"  --- 启动 {arm} (seed {seed}, {tag}) ---")
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
