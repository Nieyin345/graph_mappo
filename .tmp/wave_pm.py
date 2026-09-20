"""B′ 波次启动器：`pm_decode_s{42..46}`（解码器消融）。

继承 `/tmp/wave_v2.py` 的**全部**防线（那份是已跑通的），并按本臂的实际前提改造：

  1. 起点存在性（缺失致命）
  2. 工作区干净（跑的代码要能与提交对应）
  3. ★ **本臂自己的接线自证**：`priority_matching` 必须已经在
     `DETERMINISTIC_RESOLVER_MODES` 里，且三个调用点都用的是这个常量。
     不满足 ⟹ 第 1 步就抛 `RuntimeError`，白等 2 小时才知道
     （`failed-launch-must-be-loud` / `never-run-code-path-hides-bugs`）。
  4. ★★ **唯一变量由 `build_config()` 自证**，不读 yaml 猜：
     差异集合必须**恰好** = {"action_resolver.mode"}（`verify_arms.py` 的形状）
  5. 去重：问进程表，不问自己的账本
  6. 内存门：三量都打出来
  7. outputs 不得已存在
  8. 启动后验证：进程真起来 / resolved_config 里 mode 真是 priority_matching
     / 日志没有立刻 traceback

用法：
    python3 -u wave_pm.py --dry-run     # 只预检
    python3 -u wave_pm.py               # 预检 + 真启动
"""
# ★★ 第一件事：摘掉 sys.path[0]。
# 服务器 `.tmp/` 下有 547 个 .py（含 `types.py`），从那里启动会让
# `import enum → from types import ...` 拿到那个影子模块 ⟹ 循环导入必炸。
# 改 cwd 没用，只有摘路径有效。见记忆 `server-tmp-has-547-py-shadowing-stdlib`。
# 必须在**任何其它 import 之前**执行，所以这一段长得不像启动器的开头。
import os as _os
import sys as _sys
_here = _sys.path[0] if _sys.path else ""
if _here and _here not in ("", "."):
    _sys.path[:] = [p for p in _sys.path if p != _here]

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/opt/qkd/graph_mappo")
# ★ 摘掉 .tmp 之后必须把**仓库根**放回来：否则 `from scripts.train...`
#   找不到（sys.path[0] 是脚本所在目录 = .tmp/，正是我们要摘的那个）。
#   这就是 probe_reach_avail.py 里的同一段。
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PY = "/opt/qkd/venv/bin/python"
OUT = REPO / "outputs"
CKPT = "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
LOGDIR = Path("/tmp/pmlogs")

SEEDS = (42, 43, 44, 45, 46)
ARMS = [f"pm_decode_s{s}" for s in SEEDS]
CTRL = [f"v2_bottleneck_s{s}" for s in SEEDS]
BASE_CONFIGS = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml",
                "train_v2_bottleneck.yaml"]
ARM_CONFIGS = BASE_CONFIGS + ["train_pm_decode.yaml"]
TRAIN_PATTERN = "train_graph_mappo"

# 唯一变量：这一项**必须**是臂相对对照唯一的差异
EXPECTED_DIFF = {"action_resolver.mode": "priority_matching"}

# ---- 内存常数（本项目实测；GiB）----
PSS_PER_RUN = 25.0     # minibatch 256 稳态，无 hist encoder（与 wave_v2 一致）
FLOOR = 17.0           # 系统地板
OMP_THREADS = 8        # ★ 必须显式设，与对照一致（否则按核数折半）


def sh(cmd, timeout=60):
    return subprocess.run(cmd, shell=True, capture_output=True,
                          text=True, timeout=timeout)


def mem_available_gib():
    """★ meminfo 的 kB 其实是 KiB —— 用 /1024/1024（meminfo-kb-is-kib-not-gb）。"""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    raise RuntimeError("MemAvailable 读不到 ⟹ 不许兜底")


def live_runs():
    """★ 用 comm 锚定真 python 进程；别用 pgrep -f。"""
    r = sh(f"ps -eo comm,args | awk '$1 ~ /^python/ && /{TRAIN_PATTERN}/'")
    rows = [ln for ln in r.stdout.splitlines() if ln.strip()]
    names = []
    for ln in rows:
        parts = ln.split()
        nm = "unknown"
        for i, p in enumerate(parts):
            if p == "--run-name" and i + 1 < len(parts):
                nm = parts[i + 1]
                break
        names.append(nm)
    return rows, names


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = d
    return out


def cfg_for(configs, run_name, seed=42):
    """走训练**真正的**构造路径。"""
    import argparse as ap
    from scripts.train.train_graph_mappo import build_config
    args = ap.Namespace(configs=configs, mode="random_episode", run_name=run_name,
                        num_updates=30, seed=seed, checkpoint=CKPT, device="cpu")
    return flatten(build_config(args))


# ══════════════════════════════════════════════════════════════════════════
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--num-updates", type=int, default=30)
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                   help="只影响打印；臂集合由 SEEDS 常量决定")
    args = p.parse_args()

    arms = ARMS
    print("=" * 96)
    print("B′：`mutual_choice` → `priority_matching`（解码器消融）")
    print("=" * 96)
    print(f"  臂          {' '.join(arms)}")
    print(f"  对照        {' '.join(CTRL[:len(arms)])}")
    print(f"  configs     {' '.join(ARM_CONFIGS)}")
    print(f"  num_updates {args.num_updates}")
    print(f"  OMP 线程    OMP_NUM_THREADS={OMP_THREADS}（显式设，与对照一致）")
    print()

    # ---------- 1. 起点存在性 ----------
    ck = REPO / CKPT
    if not ck.exists():
        print(f"★ 起点不存在：{ck}  ⟹ 一条都不起")
        return 2
    print(f"  ✓ 起点存在  {ck.name}  {ck.stat().st_size/1e6:.1f} MB")

    # ---------- 2. 工作区干净 ----------
    r = sh(f"cd {REPO} && git status --porcelain")
    dirty = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if dirty:
        print("★ 工作区不干净 ⟹ 跑的代码无法与提交对应：")
        for ln in dirty:
            print(f"    {ln}")
        print("  ⟹ 一条都不起（先 commit 或还原）")
        return 2
    head = sh(f"cd {REPO} && git rev-parse --short HEAD").stdout.strip()
    print(f"  ✓ 工作区干净  HEAD={head}")

    # ---------- 3. ★ 本臂的接线自证 ----------
    # 这一条是**本臂特有的**，而且是致命的：若服务器代码还没有 f3f9c79，
    # priority_matching 会在第 1 步抛 RuntimeError（实测 0/20）。
    const = sh(f"cd {REPO} && grep -n 'DETERMINISTIC_RESOLVER_MODES = ' "
               f"qkd_rl/rl/algos/policy.py").stdout.strip()
    n_rw = sh(f"cd {REPO} && grep -c 'DETERMINISTIC_RESOLVER_MODES' "
              f"qkd_rl/rl/algos/rollout_workers.py").stdout.strip()
    n_mt = sh(f"cd {REPO} && grep -c 'DETERMINISTIC_RESOLVER_MODES' "
              f"qkd_rl/rl/algos/mappo_trainer.py").stdout.strip()
    print(f"  priority_matching 接线：")
    print(f"    policy.py:54     {const}")
    print(f"    rollout_workers  {n_rw} 处（应 ≥2）")
    print(f"    mappo_trainer    {n_mt} 处（应 ≥6）")
    ok_const = '"priority_matching"' in const.replace("'", '"')
    if not ok_const or int(n_rw or 0) < 2 or int(n_mt or 0) < 6:
        print("★ `priority_matching` 接线不完整 ⟹ 第 1 步就会抛 RuntimeError ⟹ 一条都不起")
        print("  （修法见 docs/定稿结论.md §3.4 第 2 条，commit f3f9c79）")
        return 2
    print(f"  ✓ 接线齐（本臂不会在第 1 步崩）")

    # ---------- 4. ★★ 唯一变量：build_config() 自证 ----------
    print()
    print("  唯一变量核对（走 build_config()，不读 yaml）")
    try:
        base = cfg_for(BASE_CONFIGS, "probe_ctrl")
        arm = cfg_for(ARM_CONFIGS, "probe_arm")
    except Exception as e:  # noqa: BLE001
        print(f"★ build_config() 构造失败: {type(e).__name__}: {e}  ⟹ 一条都不起")
        return 2
    ignore = {"project.run_name"}
    diffs = {}
    for k in set(base) | set(arm):
        if k in ignore:
            continue
        if base.get(k, "<缺>") != arm.get(k, "<缺>"):
            diffs[k] = (base.get(k, "<缺>"), arm.get(k, "<缺>"))
    print(f"    基线字段数 {len(base)}；差异字段 {len(diffs)} 个：")
    for k, (a, b) in sorted(diffs.items()):
        mark = "  ← 预期" if EXPECTED_DIFF.get(k) == b else "  ★★ 意外差异"
        print(f"      {k}: {a} → {b}{mark}")
    mismatched = set(diffs) != set(EXPECTED_DIFF) or any(
        diffs[k][1] != v for k, v in EXPECTED_DIFF.items())
    if mismatched:
        print(f"  ✗ 差异集合 {sorted(diffs)} != 预期 {sorted(EXPECTED_DIFF)} ⟹ 一条都不起")
        return 2
    print(f"  ✓ 唯一变量成立（差异恰好 = {sorted(EXPECTED_DIFF)}）")

    # ---------- 5. 去重：问进程表 ----------
    rows, names = live_runs()
    if rows:
        print(f"★ 已有 {len(names)} 条训练在跑 ⟹ 不让路，本波不起：")
        for nm in names:
            print(f"    {nm}")
        return 3
    print(f"  ✓ 无训练在跑（ps+awk 数出 0 条）")

    # ---------- 6. 内存门：打印输入 ----------
    avail = mem_available_gib()
    need = len(arms) * PSS_PER_RUN + FLOOR
    print()
    print("  内存门（★ 打印输入，不只打印判决）")
    print(f"    MemAvailable                {avail:8.1f} GiB")
    print(f"    本批稳态（{len(arms)} × {PSS_PER_RUN:.1f}）      {len(arms)*PSS_PER_RUN:8.1f} GiB")
    print(f"    地板                        {FLOOR:8.1f} GiB")
    print(f"    ⟹ 需要 {need:.1f} GiB，实际 {avail:.1f} GiB  "
          f"{'✓ 放行' if avail >= need else '★ 不够 ⟹ 一条都不起'}")
    if avail < need:
        return 4

    # ---------- 7. 对照臂必须在、且完整 ----------
    print()
    miss, short = [], []
    for c in CTRL[:len(arms)]:
        f = OUT / c / "metrics.jsonl"
        if not f.exists():
            miss.append(c)
            continue
        n_upd = sh(f"grep -c '\"update\"' {f}").stdout.strip() or "0"
        if int(n_upd) < args.num_updates:
            short.append(f"{c}({n_upd}/{args.num_updates})")
    if miss or short:
        print(f"★ 对照臂缺失 {miss} / 轮数不足 {short} ⟹ 一条都不起")
        return 2
    print(f"  ✓ {len(arms)} 条对照都在且已跑满 u{args.num_updates}"
          f"（{' '.join(CTRL[:len(arms)])}）")

    # ---------- 8. outputs 不得已存在 ----------
    clash = [a for a in arms if (OUT / a).exists()]
    if clash:
        print(f"★ 这些 outputs 已存在 ⟹ 拒绝覆盖（outputs/ 只增不删）：{clash}")
        return 2
    print(f"  ✓ {len(arms)} 个 outputs 目录都还不存在")

    if args.dry_run:
        print()
        print("DECISION=DRY_RUN_OK  预检全过，未启动（--dry-run）")
        return 0

    # ---------- 9. 真启动 ----------
    LOGDIR.mkdir(parents=True, exist_ok=True)
    print()
    print("=" * 96)
    print("启动")
    print("=" * 96)
    for arm in arms:
        seed = int(arm.rsplit("s", 1)[1])
        log = LOGDIR / f"{arm}.log"
        cmd = (f"cd {REPO} && OMP_NUM_THREADS={OMP_THREADS} MKL_NUM_THREADS={OMP_THREADS} "
               f"setsid nohup {PY} -u scripts/train/train_graph_mappo.py "
               f"--configs {' '.join(ARM_CONFIGS)} "
               f"--checkpoint {CKPT} --seed {seed} "
               f"--num-updates {args.num_updates} --run-name {arm} "
               f"> {log} 2>&1 < /dev/null &")
        # ★ 掐掉 ssh 是不可能的（这是本地进程），但 setsid+nohup 已足够；
        #   注意 wave_v2 的经验：后台子 shell 会持有 stdout 管道。
        r = sh(cmd, timeout=30)
        print(f"  --- 启动 {arm} (seed={seed}) rc={r.returncode} ---")

    # ---------- 10. 启动后验证 ----------
    print()
    print("等 30 秒后验证…")
    time.sleep(30)
    rows, names = live_runs()
    ok = True
    print(f"  ps+awk 数出 {len(names)} 条在跑：{names}")
    if len(names) < len(arms):
        print(f"★ 少了 {len(arms) - len(names)} 条 ⟹ 失败要吵")
        ok = False

    for arm in arms:
        log = LOGDIR / f"{arm}.log"
        txt = log.read_text(errors="replace") if log.exists() else ""
        trace = ("Traceback" in txt) or ("RuntimeError" in txt) or ("Error" in txt)
        cfg = OUT / arm / "resolved_config.yaml"
        mode = "<无>"
        if cfg.exists():
            for ln in cfg.read_text(errors="replace").splitlines():
                if ln.strip().startswith("mode:") and "priority_matching" in ln:
                    mode = "priority_matching"
                    break
                if ln.strip().startswith("mode:"):
                    mode = ln.strip()
        flag = "✓" if (mode == "priority_matching" and not trace) else "★"
        if flag == "★":
            ok = False
        print(f"    {flag} {arm}: resolved mode={mode}  traceback={trace}")
        if trace:
            print(f"      --- {arm} 日志尾部 ---")
            for ln in txt.splitlines()[-12:]:
                print(f"      {ln}")

    print()
    print(f"DECISION={'LAUNCHED_OK' if ok else 'LAUNCH_HAS_PROBLEM'}")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
