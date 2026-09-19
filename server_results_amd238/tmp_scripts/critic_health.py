#!/usr/bin/env python
"""Critic 健康度体检：扫所有历史 run，看价值头是不是"死"的。

动机（来自 th_cold 单轮日志）：
    V_std=0.028   R_std=0.40   corr(V,R)=0.157
价值输出的标准差只有回报标准差的 7%，相关性 0.157 -> R²≈0.025。
若普遍如此，advantage 的降噪几乎为零，PPO 的 actor 更新是在噪声上做梯度。
症状吻合：kl=0.0023（几乎不动）、ratio=0.9625、actor_loss=-0.0046。

跑法：/opt/qkd/venv/bin/python .tmp/critic_health.py outputs
"""
import json
import math
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")


def rows(p):
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "update" in d:
            out.append(d)
    return out


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return st.mean(xs) if xs else float("nan")


def f(x, w=8, p=3):
    if isinstance(x, float) and not math.isnan(x):
        return f"{x:>{w}.{p}f}"
    return f"{'--':>{w}}"


def main():
    runs = sorted(ROOT.glob("*/metrics.jsonl"))
    if not runs:
        print(f"没找到 metrics.jsonl（在 {ROOT} 下）")
        return
    print(f"{len(runs)} 个 run；每行取末尾 5 轮的均值\n")
    hdr = (f"{'run':<34}{'轮':>4}{'corr':>8}{'R2':>7}{'V_std':>9}"
           f"{'R_std':>8}{'V/R':>6}{'kl':>9}{'entropy':>9}{'ratio':>8}{'成功率':>8}")
    print(hdr)
    print("-" * 105)
    low = 0
    best = None
    for p in runs:
        rs = rows(p)
        if not rs:
            continue
        n = len(rs)
        tail = rs[max(0, n - 5):]
        c = mean([r.get("value_return_corr") for r in tail])
        vs = mean([r.get("value_std") for r in tail])
        rstd = mean([r.get("return_std") for r in tail])
        if isinstance(c, float) and not math.isnan(c) and c < 0.3:
            low += 1
        if best is None or n > best[0]:
            best = (n, p.parent.name, rs)
        print(f"{p.parent.name[:34]:<34}{n:>4}{f(c,8)}{f(c * c,7)}"
              f"{f(vs,9,4)}{f(rstd,8)}{f(vs / rstd if rstd else float('nan'),6,2)}"
              f"{f(mean([r.get('kl') for r in tail]),9,4)}"
              f"{f(mean([r.get('entropy') for r in tail]),9)}"
              f"{f(mean([r.get('mean_ratio') for r in tail]),8,4)}"
              f"{f(mean([r.get('mean_success_rate') for r in tail]),8,4)}")
    print()
    print(f"corr<0.3 的 run：{low} / {len(runs)}")
    if best:
        n, name, rs = best
        print(f"\n最长的 run：{name}（{n} 轮），corr(V,R) 逐轮：")
        cells = []
        for r in rs:
            c = r.get("value_return_corr")
            cells.append(f"{c:.3f}" if isinstance(c, (int, float)) else "  -- ")
        for i in range(0, len(cells), 12):
            print("  " + " ".join(cells[i:i + 12]))


if __name__ == "__main__":
    main()
