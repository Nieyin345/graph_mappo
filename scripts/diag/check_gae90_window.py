#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""★ `gae90`(λ=0.90) 判据的**窗口敏感性**复核 —— 纯标准库、不依赖服务器。

支持两代基线：
  - **第一代（旧节点 clnode316）**：`--results <.tmp/results>`，扁平命名
    `<臂>.metrics.jsonl`。它有**已知答案**（预注册 Δ=+0.0146 / t=+3.61），
    所以第 [0] 节是**硬门**：复现不出来就 `return 2`，后面一律不输出。
  - **第二代（新节点 node0.qinglong-317045）**：`--outputs <server_results/outputs>`，
    目录命名 `<臂>/metrics.jsonl`。**它没有已知答案**（第一次测量）
    ⟹ 第 [0] 节退化为**结构自检**，而不是假装复现过。

★ 这个"没有 oracle 就明说"的设计是刻意的：本项目的规矩是
**「跳过核对」不是「核对通过」，但它长得像通过**。所以缺 oracle 时不能静默跳过，
要打印出"本次退化为结构自检"。

### 为什么必须有这个脚本
`gae90` 是本项目**唯一**通过预注册判据的方向，预注册窗口是 **u25/u30**。
但"过线"对窗口极其敏感：旧节点上换掉窗口，同一个机制可以从**显著为负**
（u5：Δ=−0.0244、t=−3.23）变成**显著为正**（u25/u30：+0.0146、+3.61），
全程 u5–u30 只有 +0.0005。

### 三条自我约束（每条本项目都踩过）
1. **先复现已知答案，再谈新结论**（记忆 `thresholds-and-transcribed-numbers`）。
2. **观测单位 = 训练种子**，不是轮次。u10…u30 是**同几条 run** 的多个时刻、
   强相关；当独立样本用会虚高 df（记忆 `window-aggregation-must-align-both-sides`）。
3. **p 值 / 临界值现算**（数值积分 t 分布尾部 + 二分反解），不查表、不手抄。
   df 未知一律抛错，不许静默回退（记忆 `silent-lenient-fallback-in-thresholds`）。

### 用法
    # 第一代（有硬门）
    PYTHONIOENCODING=utf-8 python scripts/diag/check_gae90_window.py

    # 第二代（结构自检，n=4）
    python scripts/diag/check_gae90_window.py \
        --outputs server_results/outputs \
        --pairs 42:gae90_v2_s42:ent01_v2_s42 \
        --pairs 43:gae90_v2_s43:ent01_v2_s43 \
        --pairs 44:gae90_v2_s44:ent01_v2_s44 \
        --pairs 45:gae90_v2_s45:ent01_v2_s45
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

# Windows 控制台默认 GBK，符号（★ ⟹ Δ）会 UnicodeEncodeError。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 第一代的配对（旧节点，扁平命名）与**它的**预注册常量
PAIRS_G1 = ["42:gae90_s42:ent01_rerun_s42",
            "43:gae90_s43:ent01_rerun_s43",
            "44:gae90_s44:ent01_rerun_s44",
            "45:gae90_n5_s45:ent01_n5_s45",
            "46:gae90_n5_s46:ent01_n5_s46"]
PREREG_G1 = [25, 30]
DELTA_G1 = 0.0146      # 自检的期望值
T_G1 = 3.61


def read_run(path: Path):
    """→ {轮号: mean_success_rate}。

    ★★ 轮号 = **该 eval 行前面最近那个非 eval 行的 `update` 值**。

    这是**唯一正确**的规则，三条错法都已在别处踩过（记忆
    `eval-update-number-not-from-position`）：
      - 用 `seq * eval_interval` → 续跑臂整体错位；
      - 用"累计行数" → 对从头跑的臂碰巧对，对续跑臂错
        （续跑的 `update` **续着编**：实测 `ent01_s42_u30to50` 是 31…50，
        不是 1…20；计数器不归零，见 `mappo_trainer.py:1319`）；
      - 用 `update` 的顺序位置 → 同上。

    续跑又是**追加**（`open("a")`，`mappo_trainer.py:1517`）而非重开，
    所以续跑段与前段在文件里**看起来是连续的**，错法不容易暴露。
    """
    out, last = {}, None
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if "update" in o:
                last = int(o["update"])
            elif "eval_validation" in o and last is not None:
                out[last] = o["eval_validation"]["mean_success_rate"]
    return out


def t_two_sided_p(t: float, df: int) -> float:
    """双侧 p：数值积分 t 分布尾部。x = |t| + u/(1-u) 换元 + Simpson。"""
    if df <= 0:
        raise ValueError("df 必须 > 0")
    nu = float(df)
    const = math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
                     - 0.5 * math.log(nu * math.pi))

    def f(x):
        return const * (1.0 + x * x / nu) ** (-(nu + 1) / 2)

    n, h, s, a = 20000, 1.0 / 20000, 0.0, abs(t)
    for i in range(n + 1):
        u = min(i * h, 1.0 - 1e-12)
        w = 1.0 if i in (0, n) else (4.0 if i % 2 else 2.0)
        s += w * f(a + u / (1.0 - u)) / (1.0 - u) ** 2
    return 2.0 * s * h / 3.0


def t_crit(df: int, p: float = 0.05) -> float:
    """临界值现算（二分反解），不手抄 4.303 / 2.776 / 2.571。"""
    lo, hi = 0.0, 500.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if t_two_sided_p(mid, df) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def parse_pairs(specs, flat_dir, out_dir):
    """`seed:exp:ctrl` → [(seed, 实验轨迹 dict, 对照轨迹 dict)]，附带缺失报告。"""
    got, missing = [], []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit("!! --pairs 要 `seed:实验臂:对照臂`，收到 %r" % spec)
        seed, exp, ctrl = parts
        if flat_dir is not None:
            pe, pc = flat_dir / ("%s.metrics.jsonl" % exp), flat_dir / ("%s.metrics.jsonl" % ctrl)
        else:
            pe, pc = out_dir / exp / "metrics.jsonl", out_dir / ctrl / "metrics.jsonl"
        re_, rc = read_run(pe), read_run(pc)
        if not re_ or not rc:
            missing.append((seed, pe.is_file(), pc.is_file()))
            continue
        got.append((seed, re_, rc))
    return got, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None,
                    help="第一代：metrics.jsonl 所在目录（默认 <repo>/.tmp/results）")
    ap.add_argument("--outputs", default=None,
                    help="第二代：outputs 目录（含 <臂>/metrics.jsonl）")
    ap.add_argument("--pairs", action="append", default=None,
                    help="seed:实验臂:对照臂，可重复")
    ap.add_argument("--prereg", default=None, help="预注册窗口，如 25,30")
    ap.add_argument("--prereg-delta", type=float, default=None)
    ap.add_argument("--prereg-t", type=float, default=None)
    args = ap.parse_args()

    here = Path(__file__).resolve()
    repo = here.parents[2]                      # scripts/diag/x.py → repo 根

    # ── 决定用哪一代的默认值 ──
    if args.outputs:
        out_dir = Path(args.outputs)
        if not out_dir.is_absolute():
            out_dir = repo / out_dir
        flat_dir, gen = None, 2
        specs = args.pairs
        if not specs:
            raise SystemExit("!! 第二代必须显式给 --pairs（臂名与旧节点不同）")
        prereg = [int(x) for x in args.prereg.split(",")] if args.prereg else [25, 30]
        exp_delta, exp_t = args.prereg_delta, args.prereg_t
    else:
        flat_dir = Path(args.results) if args.results else (repo / ".tmp" / "results")
        out_dir, gen = None, 1
        specs = args.pairs or PAIRS_G1
        prereg = [int(x) for x in args.prereg.split(",")] if args.prereg else PREREG_G1
        exp_delta = DELTA_G1 if args.prereg_delta is None else args.prereg_delta
        exp_t = T_G1 if args.prereg_t is None else args.prereg_t

    print("=" * 100)
    print("第 %d 代基线 ｜ 数据：%s ｜ 预注册窗口 u%s"
          % (gen, flat_dir if flat_dir else out_dir, prereg))
    print("=" * 100)

    pairs, missing = parse_pairs(specs, flat_dir, out_dir)
    if missing:
        print("!! 缺数据，无法配对的：")
        for seed, e, c in missing:
            print("   seed %s：实验臂 %s ｜ 对照臂 %s" % (seed, "有" if e else "**缺**",
                                                         "有" if c else "**缺**"))
    if len(pairs) < 2:
        print("!! 可配对种子不足（%d < 2），无法做 t 检验" % len(pairs))
        return 1

    # 逐种子逐轮 Δ
    per_seed = {}
    for seed, re_, rc in pairs:
        common = sorted(set(re_) & set(rc))
        per_seed[seed] = {u: re_[u] - rc[u] for u in common}
    us = sorted(set.intersection(*[set(d) for d in per_seed.values()]))
    n = len(per_seed)
    df = n - 1
    crit = t_crit(df)
    print("可配对种子 %d 个（%s）⟹ df=%d，**临界值 = %.4f**（现算）"
          % (n, ", ".join(sorted(per_seed)), df, crit))
    print("各臂共同轮号：%s" % us)

    def win_stats(win, label):
        vals, dropped = [], []
        for s in per_seed:
            if all(u in per_seed[s] for u in win):
                vals.append(statistics.mean([per_seed[s][u] for u in win]))
            else:
                dropped.append(s)
        if len(vals) < 2:
            return None
        m = statistics.mean(vals)
        sd = statistics.stdev(vals)
        se = sd / math.sqrt(len(vals))
        t = m / se if se > 0 else float("inf")
        return dict(label=label, win=win, n=len(vals), df=len(vals) - 1, m=m, sd=sd, t=t,
                    p=t_two_sided_p(t, len(vals) - 1), crit=t_crit(len(vals) - 1),
                    pos=sum(1 for v in vals if v > 0), dropped=dropped)

    def show(r):
        flag = ""
        if r["dropped"]:
            flag = "  ⚠ 丢了 %s（窗口内轮次不全）" % ",".join(r["dropped"])
        print("  %-24s u%-16s Δ=%+.4f  SD=%.4f  t=%+.2f  p=%.4f  %d/%d同向  %s%s"
              % (r["label"], ",".join(str(u) for u in r["win"]), r["m"], r["sd"],
                 r["t"], r["p"], r["pos"], r["n"],
                 "**过线**" if abs(r["t"]) > r["crit"] else "未过线", flag))

    # ── [0] 门：第一代复现已知答案；第二代做结构自检 ──
    print("\n" + "=" * 100)
    if exp_delta is not None and exp_t is not None:
        print("[0] 硬门：复现预注册数字（复现不了 ⟹ 不输出任何新结论）")
        print("=" * 100)
        print("  期望：u%s 上 Δ=%+.4f、t=%+.2f" % (prereg, exp_delta, exp_t))
        r0 = win_stats(prereg, "★预注册")
        if r0 is None:
            print("  !! 窗口 u%s 数据不全" % prereg)
            return 2
        show(r0)
        if abs(r0["m"] - exp_delta) > 5e-4 or abs(r0["t"] - exp_t) > 0.05:
            print("  ✗ **没复现**，停下（先查数据/窗口，不要往下读）")
            return 2
        print("  ✓ 逐位复现 ⟹ 脚本可信，继续")
    else:
        print("[0] 结构自检（★ 本次**没有已知答案可复现** —— 这是第一代测量）")
        print("=" * 100)
        print("  本项目的规矩：「跳过核对」不是「核对通过」，但它长得像通过。")
        print("  所以这里不假装复现，改成**可机械判定的结构检查**：")
        bad = 0
        for s in sorted(per_seed):
            off = [u for u in per_seed[s] if u % 5 != 0]
            if off:
                print("   ✗ seed %s 的 eval 落在非 5 的倍数轮号：%s ⟹ 轮号规则存疑" % (s, off))
                bad = 1
        lens = {s: len(per_seed[s]) for s in per_seed}
        if len(set(lens.values())) != 1:
            print("   ✗ 各臂轮数不等 %s ⟹ **不可比**（窗口聚合必须两侧逐轮对齐）" % lens)
            bad = 1
        else:
            print("   ✓ 每个种子的共同轮号数一致：%d 轮（%s…%s）"
                  % (list(lens.values())[0], us[0], us[-1]))
        print("   ✓ 所有轮号都是 5 的倍数" if not bad else "")
        if bad:
            print("   ⟹ 结构检查失败，**不要往下读**")
            return 2
        print("   ✓ 结构检查通过（注意：这只排除了错位/不对齐，**不证明数值正确**）")

    # ── [1] 窗口扫描 ──
    print("\n" + "=" * 100)
    print("[1] 窗口敏感性：同一份数据，只换窗口（观测单位 = 训练种子，配对）")
    print("=" * 100)
    wins = [([u], "单轮 u%d" % u) for u in us if u <= 30]
    wins += [([u for u in us if 5 <= u <= 20], "早期 u5–u20"),
             (prereg, "★预注册 u%s" % ",".join(str(u) for u in prereg)),
             ([u for u in us if 20 <= u <= 30], "晚期 u20–u30"),
             (us, "全程 u%s–u%s" % (us[0], us[-1]))]
    for win, lab in wins:
        if win:
            r = win_stats(win, lab)
            if r:
                show(r)

    # ── [2] 逐种子逐轮原始 Δ ──
    print("\n" + "=" * 100)
    print("[2] 逐种子逐轮 Δ（配对，单位：成功率）")
    print("=" * 100)
    print("  %-8s%s" % ("u", "".join("%11s" % ("s" + s) for s in sorted(per_seed))))
    for u in us:
        print("  u%-7d%s" % (u, "".join("%+11.4f" % per_seed[s][u] for s in sorted(per_seed))))

    print("\n★ 读法：报数**必须带窗口**（docs/测试规范.md：结论要能脱离服务器复算）。")
    print("  早期显著为负 + 晚期显著为正 + 全程≈0 ⟹ 是**交叉**，不是「提升」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
