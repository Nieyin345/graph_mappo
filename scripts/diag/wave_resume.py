"""续跑波：把「慢」与「差」分开（u30 → u60）—— **按内存预算分批**。

## 为什么必须做

实测发现两族曲线形状**完全不同**：

| 族 | 末段斜率 | 逐臂在涨 | kl(u30) | clip(u30) |
|---|---|---|---|---|
| 从零（无BC） n=9 | **+0.0327**/5轮 | **6/9** | 5e-4~2e-3 | 0.013~0.061 |
| BC 暖启动 n=5 | −0.0032/5轮 | **0/5** | **3e-5~4e-4** | **0.000~0.014** |

⟹ 「从零族在 u30 显著更差（−8.6 点）」含**"更慢"**成分，只有续跑能分开。
⟹ BC 族的 kl 塌了 10~100 倍、clip 几乎恒 0 ⟹ **它已收敛，续跑是为验证这一点**。

## ⚠ 必须分批（不是"一次全铺"）

13 条 × 25 GiB = **325 GiB > 总内存 251 GiB**。
分批策略：**先铺满约 6 条**（150 GiB，留足余量），一批跑完再铺下一批。
所有臂都从 u30 续到 u60，判读时**只比都到 u60 的**。

## 判据（预注册）

  (a) 从零族续跑后**追平** BC 族 ⟹ 「更差」是「更慢」，BC 的价值 = 降方差
  (b) 从零族续跑后**仍显著更差** ⟹ 从零训在该问题上确实不行
  (c) ★ **BC 族续跑若反而涨了** ⟹ §九 的「已收敛」判断错了
      —— 这是反例，比 (a)/(b) 更有信息

## ⚠ 三条硬约束

1. **续跑读新代码**（进程新起）⟹ 配置里 `pool_include_max` 必须仍是 **false**，
   否则续跑臂 = BC + cmax 两个变量（本脚本第 2 项自证）。
2. **不双写**：续跑写同一目录 `outputs/<臂名>`（追加 metrics），
   必须先确认原臂**不在跑**（问进程表，不问账本）。
3. **`--num-updates` 是增量**（`mappo_trainer.py:1569`
   `target_updates = update_count + num_updates`），给 30 就跑 u31→u60。

用法：
    python3 -u wave_resume.py --dry-run     # 预检 + 打印分批计划
    python3 -u wave_resume.py               # 只起**第一批**
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
LOGDIR = Path("/tmp/resumelogs")
TRAIN_PATTERN = "train_graph_mappo"

BASE_CONFIGS = ["rl_algorithm.yaml", "train_full_rl.yaml", "train_ent01.yaml"]

# 从零族 9 条（主问题）+ BC 族 4 条（对照）。顺序**有意义**：先铺从零族。
# @@ 2026-09-21 20:1x：scratch 族已续跑完（u58~60），本波改为**只续 BC 族**
# —— 作为 resume 判读的对照：没有它，从零族续跑后只能报单侧曲线。
ALL_ARMS = [f"ent01_rerun_s{s}" for s in range(42, 46)]

RESUME_UPDATES = 30          # 增量：u30 -> u60
PSS_PER_RUN = 25.0
FLOOR = 17.0
RESERVE = 25.0               # 给别的启动器/杂项留一条臂的余量
OMP_THREADS = 8


def sh(cmd, timeout=60):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def mem_available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    raise RuntimeError("MemAvailable 读不到 ⟹ 不许兜底")


def live_names():
    """★ 问进程表，不问账本（chain-must-ask-proc-not-its-own-ledger）。"""
    r = sh(f"ps -eo comm,args | awk '$1 ~ /^python/ && /{TRAIN_PATTERN}/'")
    names = []
    for ln in r.stdout.splitlines():
        parts = ln.split()
        for i, p in enumerate(parts):
            if p == "--run-name" and i + 1 < len(parts):
                names.append(parts[i + 1])
                break
    return names


def live_with_update():
    """[(run_name, pss_gib, update)] —— 按**进程组**聚合（父 + 8 workers）。

    ★ 别只抓父进程：worker 的 cmdline 是
      `python -c from multiprocessing.spawn import spawn_main ...`，
      **不含** `train_graph_mappo` ⟹ 只抓父会低估约 6.6 GiB/条。
    """
    r = sh("ps -eo pid,ppid,comm,args")
    parents, workers = {}, []
    for ln in r.stdout.splitlines():
        parts = ln.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, comm, args = parts
        if not comm.startswith("python"):
            continue
        if TRAIN_PATTERN in args and "--run-name" in args:
            toks = args.split()
            parents[int(pid)] = toks[toks.index("--run-name") + 1]
        elif "multiprocessing" in args or "spawn_main" in args:
            workers.append((int(pid), int(ppid)))
    agg = {nm: [pss_of(pid), 0.0] for pid, nm in parents.items()}
    for wpid, wppid in workers:
        nm = parents.get(wppid)
        if nm in agg:
            agg[nm][1] += pss_of(wpid)
    out = []
    for nm, (p, w) in agg.items():
        out.append((nm, p + w, last_update(nm) or -1))
    return out


def pss_of(pid):
    """单进程 PSS（GiB）。读不到返回 0.0。"""
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) / 1024.0 / 1024.0
    except Exception:
        pass
    return 0.0


def last_update(run):
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        return None
    last = None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = __import__("json").loads(line)
        except Exception:
            continue
        if "update" in o:
            last = o["update"]
    return last


def plan_batches(candidates, avail):
    """还有多少条能起 ⟹ 切出第一批。

    ★★ 判据 = 可用 − **在跑臂的待涨量** − 预留，不是核数。

    @@ 2026-09-21 修：原先硬编码 `warm = 0.0`，注释写"在跑的臂都在 u>25
    已稳态"。**链式触发时那是假的** —— clean 波刚起在 u1、每条才 17.5 GiB
    却要涨到 25 ⟹ 待涨量被记成 0 ⟹ 超发 3 条 ⟹ 实测 MemAvailable 掉到
    21.0 GiB（地板 17），**差几分钟就 OOM**。
    现在按**实测轮号**推每条自己的待涨量（u1≈9G → u15+≈25G 线性插值）。
    """
    warm = 0.0
    rows = live_with_update()
    for nm, _pss, upd in rows:
        if upd < 0:
            est_now = PSS_PER_RUN       # 读不到轮号 ⟹ 按稳态算（保守）
        elif upd >= 15:
            est_now = PSS_PER_RUN
        else:
            est_now = 9.0 + (PSS_PER_RUN - 9.0) * (upd / 15.0)
        warm += max(0.0, PSS_PER_RUN - est_now)
    room = avail - warm - RESERVE
    max_new = int(room // PSS_PER_RUN)
    return candidates[:max_new], candidates[max_new:], max_new


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--updates", type=int, default=RESUME_UPDATES)
    args = p.parse_args()

    print("=" * 96)
    print("续跑波：u30 → u60（把「慢」与「差」分开）")
    print("=" * 96)
    print(f"  候选 {len(ALL_ARMS)} 条（从零族 9 + BC 族 4），每条 +{args.updates} 轮（增量语义）")
    print()

    fails = []
    names = live_names()
    print(f"  进程表里在跑的 run：{len(names)} 条")
    for n in names:
        print(f"      {n}")

    # ---------- 1. 每条臂体检 ----------
    # @@ 2026-09-21：判据从 `checkpoint_final.pt` 改成 **metrics 的 u 号**。
    #   ★ 实测发现 `checkpoint_final.pt` 对**续跑臂不可靠**：
    #     `scratch_s42` 还在跑 u58，但它的 final.pt mtime 是 **16:12**
    #     （第一次 u30 跑留下的**陈旧文件**）—— 照它判会误以为已完工。
    #     `trainer.train()` 只有返回时才重写 final，中途一直是旧的。
    print(f"\n  {'臂':<20}{'u':>5}  状态")
    ready = []
    for a in ALL_ARMS:
        ck = OUT / a / "checkpoint_final.pt"
        u = last_update(a)
        st = []
        if u is None:
            st.append("★ 无 metrics")
            fails.append(f"{a}: 无 metrics.jsonl")
        else:
            u_stale = ck.exists() and ck.stat().st_mtime < (
                OUT / a / "metrics.jsonl").stat().st_mtime
            if u_stale:
                st.append("（final.pt 陈旧于 metrics）")
        if a in names:
            st.append("★ 正在跑（会双写）")
            fails.append(f"{a}: 正在跑，双写风险")
        if u is not None and u != 30:
            st.append(f"⚠ u={u}（预期 30）")
            fails.append(f"{a}: u={u} 不是 30 ⟹ 语义不是「从 u30 续」")
        if not any("★" in x or "⚠" in x for x in st):
            ready.append(a)
        print(f"  {a:<20}{u if u is not None else '—':>5}  "
              f"{' '.join(st) if st else '✓ 可续跑'}")

    # ---------- 2. ★ 续跑配置自证 ----------
    print("\n  ★ 续跑配置自证：`pool_include_max` 必须为 false")
    try:
        import argparse as ap
        from scripts.train.train_graph_mappo import build_config
        cfg = build_config(ap.Namespace(
            configs=BASE_CONFIGS, mode="random_episode", run_name="probe_resume",
            num_updates=args.updates, seed=42, checkpoint=None, device="cpu"))
        v = cfg["model"]["critic"].get("pool_include_max", False)
        print(f"      model.critic.pool_include_max = {v}   "
              f"{'✓ 与对照一致' if not v else '★ 会混进 cmax 变量'}")
        if v:
            fails.append("续跑配置里 pool_include_max=true ⟹ 混入 cmax 变量")
    except Exception as e:
        fails.append(f"build_config() 失败: {type(e).__name__}: {e}")

    # ---------- 3. 分批计划 ----------
    avail = mem_available_gib()
    live = live_with_update()
    warm = 0.0
    print(f"\n  在跑的 run（按进程组聚合 = 父 + workers）：")
    for nm, pss, upd in sorted(live):
        if upd < 0:
            est = PSS_PER_RUN
        elif upd >= 15:
            est = PSS_PER_RUN
        else:
            est = 9.0 + (PSS_PER_RUN - 9.0) * (upd / 15.0)
        d = max(0.0, PSS_PER_RUN - est)
        warm += d
        print(f"      {nm:<20} PSS={pss:5.1f}  u={upd:<4} 待涨={d:5.1f}")
    batch, rest, max_new = plan_batches(ready, avail)
    print(f"\n  内存门与分批计划")
    print(f"    MemAvailable            {avail:.1f} GiB")
    print(f"    在跑 {len(live)} 条的待涨量     −{warm:.1f} GiB")
    print(f"    预留（别的启动器）        −{RESERVE:.1f} GiB")
    print(f"    每条稳态                 −{PSS_PER_RUN:.1f} GiB")
    print(f"    ⟹ 本批最多可起           {max_new} 条")
    print(f"\n    第一批 ({len(batch)}): {' '.join(batch) if batch else '(无)'}")
    print(f"    后续批次 ({len(rest)}): {' '.join(rest) if rest else '(无)'}")

    print("\n" + "=" * 96)
    if fails:
        print("★ 预检未过 ⟹ 一条都不起：")
        for f in fails:
            print(f"    ✗ {f}")
        return 2
    if not batch:
        print("★ 可用内存不足以再起任何一条 ⟹ 等 cmax 波结束后再跑")
        return 2
    print("✓ 预检全过")
    if args.dry_run:
        print("\n--dry-run：不启动。")
        return 0

    LOGDIR.mkdir(parents=True, exist_ok=True)
    print(f"\n启动第一批 {len(batch)} 条，日志 → {LOGDIR}")
    launched = []
    for a in batch:
        log = LOGDIR / f"{a}.resume.log"
        ck = OUT / a / "checkpoint_final.pt"
        seed = a.split("_s")[-1]
        cmd = (f"cd {REPO} && OMP_NUM_THREADS={OMP_THREADS} MKL_NUM_THREADS={OMP_THREADS} "
               f"setsid nohup {PY} -u scripts/train/train_graph_mappo.py "
               f"--configs {' '.join(BASE_CONFIGS)} "
               f"--checkpoint {ck} "
               f"--seed {seed} --num-updates {args.updates} "
               f"--run-name {a} > {log} 2>&1 < /dev/null &")
        print(f"  --- 续跑 {a} (seed {seed}) ---")
        sh(cmd, timeout=25)
        launched.append(a)
        time.sleep(1.5)

    print("\n  启动后验证（等 25s）...")
    time.sleep(25)
    now = live_names()
    ok = []
    for a in launched:
        log = LOGDIR / f"{a}.resume.log"
        txt = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        bad = "Traceback" in txt or "RuntimeError" in txt
        started_ok = ("Resumed from" in txt)
        if a in now and not bad and started_ok:
            print(f"    ✓ {a}")
            ok.append(a)
        else:
            why = ("进程不在" if a not in now else
                   ("日志有 error" if bad else "没有 'Resumed from' 行"))
            print(f"    ✗ {a}: {why}")
            for ln in txt.splitlines()[-6:]:
                print(f"        {ln}")
    print(f"\n  ★ 实际起来 {len(ok)}/{len(launched)}")
    print(f"  ★ 剩余 {len(rest)} 条待下一批（等本批跑完再跑本脚本）")
    return 0 if len(ok) == len(launched) else 3


if __name__ == "__main__":
    raise SystemExit(main())
