# -*- coding: utf-8 -*-
"""`pss_trend.fit_slopes` 的回归测试：**去重后点太少必须拒绝拟合**（B-14）。

配对先例：`test_mem_pss_views.py`（对 `mem_pss.py` 的外推判据做已知答案断言）、
`test_launch_gate.py`。本文件对 `pss_trend.py` 的斜率判据做同类断言。
**纯算术 + 临时 CSV，不连服务器。**

## 锁住的缺陷（B-14，2026-09-21 核实）

旧实现：

    if len(pts) < 3:            # ← 查的是**原始采集数**
        continue
    seen = {}; ...
    us = sorted(seen)
    n = len(xs)                 # ← n 可能只有 1
    den = sum((x - mx) ** 2 for x in xs)
    slope = (...) / den if den else 0.0     # ← den=0 ⟹ 斜率打印成 +0.000

⟹ 一个 run 被采样 6 次但**只推进了 1 轮**时，去重后剩 1 点 ⟹ `den = 0` ⟹
  `slope = 0.0` ⟹ 打印 `+0.000 GB/轮 (10→10 轮, 12.41→12.41 GB)`。
  **看起来是一次有效的、平坦的测量** —— 而实际上根本没量。

同族（本项目已记）：
  · `silent-lenient-fallback-in-thresholds` —— `.get(df, 2.0)` 制造**假显著**
  · 本条 —— `else 0.0` 制造**假平坦**
  · `gate-must-print-its-inputs` —— 恒真的门不报错
共同点：**`if 不可用 then 给个默认值` 让"不可用"与"量出来是那个值"无法区分。**

## 判据

  1. 反证：**旧逻辑**在同一夹具上确实返回 0.0（证明缺陷真实存在）
  2. 新逻辑：`slope is None` 且 `reason` 说明去重后只有 1 点
  3. 正对照：真在涨的 run **必须照常给出斜率** ≈ +0.21
     （否则"拒绝拟合"可以靠"永远拒绝"来假通过）
  4. 边界：去重后恰好 2 点也要拒绝；恰好 3 点要接受
"""
from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_ok = [True]


def check(label, got, want):
    good = (got == want)
    _ok[0] = _ok[0] and good
    print("  %s %-58s got=%r want=%r" % ("✓" if good else "✗", label, got, want))


SRC = Path(__file__).resolve().parent / "pss_trend.py"
spec = importlib.util.spec_from_file_location("pss_trend_mod", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)          # 只定义函数；main() 在 __main__ 守卫内


def write_csv(rows) -> str:
    """rows: [(run, update, pss_gb)] —— 列序照抄 pss_trend 的真实表头。"""
    fd = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                     newline="", encoding="utf-8")
    w = csv.writer(fd)
    w.writerow(["t_min", "run", "pss_gb", "procs", "update",
                "memavail_gb", "swap_free_gb"])
    for i, (run, u, p) in enumerate(rows):
        w.writerow(["%.1f" % i, run, "%.2f" % p, 9, u, "200.00", "0.00"])
    fd.close()
    return fd.name


def old_fit(path):
    """**旧实现**（B-14）：查原始点数，去重后可能剩 1 点 ⟹ 返回 0.0。"""
    data = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["update"]) > 0:
                data[r["run"]].append((float(r["update"]), float(r["pss_gb"])))
    res = {}
    for nm, pts in data.items():
        if len(pts) < 3:
            continue
        seen = {}
        for u, p in pts:
            seen.setdefault(u, p)
        us = sorted(seen)
        xs = [float(u) for u in us]
        ys = [seen[u] for u in us]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        res[nm] = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den \
            if den else 0.0
    return res


print("=" * 78)
print("1. 反证：旧逻辑在「6 次采样但只有 1 个轮号」时返回 **0.0**")
print("=" * 78)
p = write_csv([("arm_single", 10, 12.41)] * 6
              + [("arm_trend", 10, 20.00), ("arm_trend", 11, 20.21),
                 ("arm_trend", 12, 20.42), ("arm_trend", 13, 20.63)])
old = old_fit(p)
check("旧逻辑：arm_single 给出 0.0（**假平坦**）", old.get("arm_single"), 0.0)
check("旧逻辑：arm_trend 正常（正对照，n=4）",
      round(old.get("arm_trend", 0), 4), 0.21)

print()
print("=" * 78)
print("2. 新逻辑：同一夹具上 arm_single **必须拒绝拟合**")
print("=" * 78)
new = {r["run"]: r for r in mod.fit_slopes(p)}
s = new["arm_single"]
check("arm_single 的 slope 是 None（不是 0.0）", s["slope"], None)
check("arm_single 原始采样数为 6", s["n_raw"], 6)
check("arm_single 去重后只剩 1 个轮号", s["n_uniq"], 1)
check("★ `slope is None` 而不是 `slope == 0`（0.0 会被读成『平的』）",
      s["slope"] is None and s["slope"] != 0, True)
check("reason 里说明了去重后的点数", "1 个不同轮号" in (s["reason"] or ""), True)

print()
print("=" * 78)
print("3. 正对照：真在涨的 run **必须照常**给出斜率")
print("=" * 78)
t = new["arm_trend"]
check("arm_trend 有斜率", t["slope"] is not None, True)
check("arm_trend 斜率 ≈ +0.210 GB/轮", round(t["slope"], 3), 0.21)
check("arm_trend 区间 10→13", (t["u0"], t["u1"]), (10, 13))
check("arm_trend n_uniq=4", t["n_uniq"], 4)

print()
print("=" * 78)
print("4. 边界：去重后恰好 2 点拒绝、恰好 3 点接受")
print("=" * 78)
p2 = write_csv([("two", 10, 1.0)] * 4 + [("two", 11, 1.2)] * 4
               + [("three", 10, 1.0), ("three", 11, 1.2), ("three", 12, 1.4)])
r2 = {r["run"]: r for r in mod.fit_slopes(p2)}
check("2 个不同轮号 ⟹ 拒绝", r2["two"]["slope"], None)
check("3 个不同轮号 ⟹ 接受（n_uniq=3）", r2["three"]["n_uniq"], 3)
check("3 点斜率 = +0.200", round(r2["three"]["slope"], 3), 0.2)

print()
print("=" * 78)
print("5. 源码断言：`if den else 0.0` 这个假平坦的写法不得复活")
print("=" * 78)
raw = SRC.read_text(encoding="utf-8")
live = "\n".join(l.split("#")[0] for l in raw.splitlines())
check("活代码里不得再有 `if den else 0.0`", "if den else 0.0" in live, False)
check("活代码里不得再用原始点数当唯一门槛", "if len(pts) < 3" in live, False)
check("判据已拎成可调用的纯函数 fit_slopes()", "def fit_slopes(" in live, True)
check("n_uniq 参与判据", "if n_uniq < 3:" in live, True)

print()
print("=" * 78)
print("全部通过 ✓" if _ok[0] else "**有失败 ✗**")
sys.exit(0 if _ok[0] else 1)
