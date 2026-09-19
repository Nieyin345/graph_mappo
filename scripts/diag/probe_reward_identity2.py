# -*- coding: utf-8 -*-
"""上一个探针的 ③④ 要重做：112 个点是 **5 个簇**，不是 112 个独立样本。

### 我错在哪

`probe_reward_identity.py` 把 112 个 (run, update) 点当独立样本回归，
报出 "reward vs success_rate R²=0.246"。但 5 个 run 内部逐轮几乎不变
（served 65~67K），**变异几乎全部来自 run 之间**。于是这个 R² 实际是在
用 5 个簇心做 112 次计数 —— 簇内越稳，样本越"多"，读数越假。

正确做法：**先按 run 取均值，再跨 run 看（n=5）**；或做**簇内去心**，
只看同一 run 内 reward 与指标是否同步动。

### 同时要查的一件事

若 `success_rate = served/arrived` 且 arrived 近似常数（CV 1.6%），
那么 success_rate 应与 served **高度共线**。③ 给出的 R²=0.20 与这个
预期冲突 —— 要么 arrived 没我说的那么稳，要么 `mean_success_rate`
**不是** served/arrived（那就是又一个"同名不同定义"的隐患，
本项目吃过这个亏）。这一条必须查清楚，不能猜。

用法（服务器上）：python3 /tmp/probe_reward_identity2.py
"""
import json
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")
RUNS = ["ent01_s42", "ent01_s43", "ent01_s44", "ent01_s45", "ent01_s46"]


def load(run):
    p = OUT / run / "rollout_debug.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def ols(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx if sxx else 0.0
    b = my - a * mx
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ss_res / ss_tot if ss_tot else float("nan"))


per_run = {}
for run in RUNS:
    rows = load(run)
    if not rows:
        continue
    n = len(rows)
    per_run[run] = {k: sum(r[k] for r in rows) / n for k in (
        "mean_reward", "mean_reward_served", "mean_served_keys",
        "mean_arrived_keys", "mean_success_rate", "mean_generated_keys")}
    per_run[run]["n"] = n

print("=" * 96)
print("① 逐 run 均值（n=5 这才是独立样本）")
print("=" * 96)
hdr = f"{'run':<14}{'n':>4}{'reward':>10}{'served_r':>11}{'served_keys':>13}{'arrived':>11}{'sr(reported)':>14}{'served/arrived':>16}"
print(hdr)
print("-" * 96)
for run, d in per_run.items():
    ratio = d["mean_served_keys"] / d["mean_arrived_keys"]
    flag = "" if abs(ratio - d["mean_success_rate"]) < 1e-3 else "  ★不一致"
    print(f"{run:<14}{d['n']:>4}{d['mean_reward']:>10.5f}{d['mean_reward_served']:>11.5f}"
          f"{d['mean_served_keys']:>13,.0f}{d['mean_arrived_keys']:>11,.0f}"
          f"{d['mean_success_rate']:>14.6f}{ratio:>16.6f}{flag}")

runs = list(per_run)
print("\n" + "=" * 96)
print("② 跨 run 回归（n=5，真正的独立样本）")
print("=" * 96)
for label, xk, yk in (
    ("reward        ~ reward_served", "mean_reward_served", "mean_reward"),
    ("reward_served ~ served_keys", "mean_served_keys", "mean_reward_served"),
    ("★ reward       ~ success_rate", "mean_success_rate", "mean_reward"),
    ("★ reward       ~ generated_keys", "mean_generated_keys", "mean_reward"),
    ("sr            ~ served_keys", "mean_served_keys", "mean_success_rate"),
):
    xs = [per_run[r][xk] for r in runs]
    ys = [per_run[r][yk] for r in runs]
    a, b, r2 = ols(xs, ys)
    print(f"  {label:<32} slope={a:>12.4e}  R²={r2:>9.5f}  r={r2**0.5:>8.5f}")

print("\n" + "=" * 96)
print("③ 簇内去心回归（只问：同一个 run 内，两者同步吗）")
print("=" * 96)
for label, xk, yk in (
    ("reward ~ success_rate", "mean_success_rate", "mean_reward"),
    ("reward ~ generated_keys", "mean_generated_keys", "mean_reward"),
    ("reward ~ served_keys", "mean_served_keys", "mean_reward"),
):
    xs, ys = [], []
    for run in runs:
        rows = load(run)
        mx = sum(r[xk] for r in rows) / len(rows)
        my = sum(r[yk] for r in rows) / len(rows)
        # 按各自 SD 归一，否则量纲大的 run 主导
        sx = (sum((r[xk] - mx) ** 2 for r in rows) / len(rows)) ** 0.5 or 1.0
        sy = (sum((r[yk] - my) ** 2 for r in rows) / len(rows)) ** 0.5 or 1.0
        xs += [(r[xk] - mx) / sx for r in rows]
        ys += [(r[yk] - my) / sy for r in rows]
    a, b, r2 = ols(xs, ys)
    print(f"  {label:<32} slope={a:>12.4f}  R²={r2:>9.5f}  r={r2**0.5:>8.5f}  (n={len(xs)})")

print("\n" + "=" * 96)
print("④ 每个 run 内 arrived 稳不稳（决定 sr 是否 = served 的仿射）")
print("=" * 96)
for run in runs:
    rows = load(run)
    arr = [r["mean_arrived_keys"] for r in rows]
    sv = [r["mean_served_keys"] for r in rows]
    sr = [r["mean_success_rate"] for r in rows]
    ma = sum(arr) / len(arr)
    sa = (sum((x - ma) ** 2 for x in arr) / len(arr)) ** 0.5
    ms = sum(sv) / len(sv)
    ss = (sum((x - ms) ** 2 for x in sv) / len(sv)) ** 0.5
    a, _, r2 = ols(sv, sr)
    print(f"  {run:<14} arrived CV={sa/ma*100:6.3f}%   served CV={ss/ms*100:6.3f}%   "
          f"sr~served 簇内 R²={r2:.5f}")
