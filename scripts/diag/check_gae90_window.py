#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""★ `gae90`(λ=0.90) 判据的**窗口敏感性**复核 —— 纯标准库、不依赖服务器。

### 为什么有这个脚本
`gae90` 是本项目**唯一**通过预注册判据的方向，预注册窗口是 **u25/u30**
（见 `.tmp/recompute_gae90_n5.py` 的 `PLATFORM = [25, 30]`）。
但"过线"对窗口极其敏感：换掉窗口，同一个机制可以从**显著为负**变成
**显著为正**。本项目的 `docs/测试规范.md` 要求结论可脱离服务器复算，
所以把这件事固化成脚本、随代码走，而不是留在 gitignored 的 `.tmp/` 里。

### 结论（2026-09-20 实测，数据来自已抓回本地的 `metrics.jsonl` 镜像）
| 窗口 | Δ | t | 判定（df=4，临界 2.776） |
|---|---|---|---|
| 单轮 u5 | −0.0244 | −3.23 | **过线（负）** |
| 早期 u5–u20 | −0.0070 | −0.69 | 未过线 |
| **预注册 u25/u30** | **+0.0146** | **+3.61** | **过线** |
| 晚期 u20–u30（只多含 u20） | +0.0125 | +2.04 | **未过线** |
| 全程 u5–u30 | +0.0005 | +0.07 | 未过线 |

⟹ 这**不是**「λ=0.90 提升成功率」，是**一次「早负晚正」的交叉**。

### 三条自我约束（每条本项目都踩过）
1. **先复现已知答案，再谈新结论**（第 [0] 节是硬门：复现不出 +0.0146/t=+3.61
   就 `return 2`，后面的窗口扫描一律不输出）。记忆 `thresholds-and-transcribed-numbers`。
2. **观测单位 = 训练种子**，不是轮次。u10…u30 是**同 5 条 run** 的 5 个时刻、
   强相关；当独立样本用会虚高 df。
3. **p 值 / 临界值现算**（数值积分 t 分布尾部 + 二分反解），不查表、不手抄。
   df 未知一律抛错，不许静默回退（记忆 `silent-lenient-fallback-in-thresholds`）。

### 用法
    PYTHONIOENCODING=utf-8 python scripts/diag/check_gae90_window.py
    # 数据目录默认 <repo>/.tmp/results，可用 --results 指定
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

# 配对：实验臂（λ=0.90） → 同训练种子的对照臂（λ=0.95）
PAIRS = [("s42", "gae90_s42", "ent01_rerun_s42"),
         ("s43", "gae90_s43", "ent01_rerun_s43"),
         ("s44", "gae90_s44", "ent01_rerun_s44"),
         ("s45", "gae90_n5_s45", "ent01_n5_s45"),
         ("s46", "gae90_n5_s46", "ent01_n5_s46")]

# ★ 预注册窗口。**必须与 .tmp/recompute_gae90_n5.py 的 PLATFORM 一致**；
#   要改窗口就得同时改预注册脚本，否则两处会各自漂移。
PREREG = [25, 30]
PREREG_DELTA = 0.0146      # 自检的期望值
PREREG_T = 3.61


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
                if last % 5 != 0:
                    print("  !! %s 的 eval 落在 update=%d（非 5 的倍数）"
                          "⟹ 轮号规则存疑" % (path.parent.name, last))
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
    """临界值现算（二分反解），不手抄 4.303 / 2.776。"""
    lo, hi = 0.0, 500.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if t_two_sided_p(mid, df) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None,
                    help="metrics.jsonl 所在目录（默认 <repo>/.tmp/results）")
    args = ap.parse_args()

    here = Path(__file__).resolve()
    repo = here.parents[2]                      # scripts/diag/x.py → repo 根
    res = Path(args.results) if args.results else (repo / ".tmp" / "results")
    print("数据目录：%s" % res)

    per_seed = {}
    missing = []
    for lab, g, c in PAIRS:
        pg, pc = res / ("%s.metrics.jsonl" % g), res / ("%s.metrics.jsonl" % c)
        if not (pg.is_file() and pc.is_file()):
            missing.append(lab)
            continue
        rg, rc = read_run(pg), read_run(pc)
        per_seed[lab] = {u: rg[u] - rc[u] for u in sorted(set(rg) & set(rc))}
    if missing:
        print("!! 缺文件，无法配对的种子：%s" % missing)
    if len(per_seed) < 2:
        print("!! 可配对种子不足，无法做 t 检验")
        return 1

    us = sorted(set.intersection(*[set(d) for d in per_seed.values()]))
    print("各臂共同轮号：%s" % us)
    print("预注册窗口：u%s（期望 Δ=%+.4f、t=%+.2f）" % (PREREG, PREREG_DELTA, PREREG_T))

    def win_stats(win, label):
        """逐种子在 win 内求均 Δ → 对**种子**做单样本 t（df=n-1）。"""
        vals = [statistics.mean([per_seed[l][u] for u in win])
                for l in per_seed if all(u in per_seed[l] for u in win)]
        n = len(vals)
        if n < 2:
            return None
        df = n - 1
        m = statistics.mean(vals)
        sd = statistics.stdev(vals)
        se = sd / math.sqrt(n)
        t = m / se if se > 0 else float("inf")
        return dict(label=label, win=win, n=n, df=df, m=m, sd=sd, t=t,
                    p=t_two_sided_p(t, df), crit=t_crit(df),
                    pos=sum(1 for v in vals if v > 0))

    def show(r):
        print("  %-24s u%-16s Δ=%+.4f  SD=%.4f  t=%+.2f  p=%.4f  %d/%d同向  %s"
              % (r["label"], ",".join(str(u) for u in r["win"]), r["m"], r["sd"],
                 r["t"], r["p"], r["pos"], r["n"],
                 "**过线**" if abs(r["t"]) > r["crit"] else "未过线"))

    # ── [0] 硬门：先复现预注册数字 ──
    print("\n" + "=" * 100)
    print("[0] 硬门：复现预注册数字（复现不了 ⟹ 不输出任何新结论）")
    print("=" * 100)
    r0 = win_stats(PREREG, "★预注册 u25/u30")
    if r0 is None:
        print("  !! 窗口 u%s 数据不全" % PREREG)
        return 2
    show(r0)
    if abs(r0["m"] - PREREG_DELTA) > 5e-4 or abs(r0["t"] - PREREG_T) > 0.05:
        print("  ✗ 期望 Δ=%+.4f、t=%+.2f —— **没复现**，停下（先查数据/窗口，"
              "不要往下读）" % (PREREG_DELTA, PREREG_T))
        return 2
    print("  ✓ 逐位复现 ⟹ 脚本可信，继续")

    # ── [1] 窗口扫描 ──
    print("\n" + "=" * 100)
    print("[1] 窗口敏感性：同一份数据，只换窗口（观测单位 = 训练种子，配对）")
    print("=" * 100)
    for win, lab in [([5], "单轮 u5"), ([10], "单轮 u10"), ([15], "单轮 u15"),
                     ([20], "单轮 u20"), ([25], "单轮 u25"), ([30], "单轮 u30"),
                     ([5, 10, 15, 20], "早期 u5–u20"),
                     ([25, 30], "★预注册 u25/u30"),
                     ([20, 25, 30], "晚期 u20–u30"),
                     ([10, 15, 20, 25, 30], "全程 u10–u30"),
                     ([5, 10, 15, 20, 25, 30], "全程 u5–u30")]:
        r = win_stats(win, lab)
        if r:
            show(r)

    # ── [2] 逐种子逐轮原始 Δ ──
    print("\n" + "=" * 100)
    print("[2] 逐种子逐轮 Δ（配对，单位：成功率）")
    print("=" * 100)
    print("  %-8s%s" % ("u", "".join("%11s" % l for l in per_seed)))
    for u in us:
        print("  u%-7d%s" % (u, "".join("%+11.4f" % per_seed[l][u] for l in per_seed)))

    print("\n★ 读法：早期显著为负、晚期显著为正、全程≈0 ⟹ **交叉**，不是「提升」。")
    print("  报数必须带窗口（docs/测试规范.md：结论要能脱离服务器复算）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
