#!/usr/bin/env python
"""H1 假说体检 + 验证集序列。

背景（来自 .tmp/screen_r6.sh 的记录）：
  真实任务是 **验证集** success rate（不是训练成功率 0.855）。
  real_base 验证分在训练中 0.701 -> 0.622 一路退化，这是"落地阻断项"。
  主假说 H1：critic 把 return 学得太好（corr -> 0.9）-> 真优势趋零 ->
  归一化把噪声放大到单位尺度 -> 策略在噪声方向小步随机游走，漂出好区域。

  base 签名（据 screen_r6.sh 注释）：KL 极小(0.001-0.004)、熵不塌(3.8->4.0)、
  mean_abs_advantage 1.08->0.32 **单调收缩**、value_return_corr -> 0.9、
  训练成功率缓降(0.873->0.856)。

本脚本：把所有 run 的验证集序列 + H1 签名逐轮打出来，看 H1 是否复现。
跑法：/opt/qkd/venv/bin/python .tmp/h1_signature.py
"""
import json
import math
import pathlib
import statistics as st
import sys

MAIN = pathlib.Path("/opt/qkd/graph_mappo")


def load(run):
    p = MAIN / "outputs" / run / "metrics.jsonl"
    if not p.exists():
        return [], []
    tr, ev = [], []
    for line in p.open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        if "update" in d:
            tr.append(d)
        elif "eval_validation" in d:
            ev.append(d["eval_validation"])
    return tr, ev


def f(x, w=8, p=4):
    if isinstance(x, (int, float)) and not math.isnan(x):
        return f"{x:>{w}.{p}f}"
    return f"{'--':>{w}}"


def seg_mean(rs, key, lo, hi):
    xs = [r.get(key) for r in rs[lo:hi]]
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return st.mean(xs) if xs else float("nan")


def main():
    runs = sorted(p.parent.name for p in MAIN.glob("outputs/*/metrics.jsonl"))
    if not runs:
        print("没找到 metrics.jsonl")
        return

    data = {r: load(r) for r in runs}

    # ---- 1. 验证集序列（真正的目标指标）----
    print("=" * 100)
    print("验证集 success_rate 序列（eval 每 5 轮一次）")
    print("=" * 100)
    for r in runs:
        tr, ev = data[r]
        if not ev:
            continue
        seq = " ".join(f"{e.get('mean_success_rate', float('nan')):.4f}" for e in ev)
        first = ev[0].get("mean_success_rate", float("nan"))
        last = ev[-1].get("mean_success_rate", float("nan"))
        d = last - first
        flag = "  <== 退化" if d < -0.02 else ("  <== 提升" if d > 0.02 else "")
        print(f"{r:<24} n={len(ev):<2} {seq}   Δ={d:+.4f}{flag}")

    # ---- 2. H1 签名：优势是否单调收缩 ----
    print()
    print("=" * 100)
    print("H1 签名（前 1/3 与后 1/3 对比；优势收缩 = 支持 H1）")
    print("=" * 100)
    hdr = (f"{'run':<24}{'轮':>4}"
           f"{'|A|前':>9}{'|A|后':>9}{'收缩':>8}"
           f"{'corr前':>8}{'corr后':>8}"
           f"{'kl后':>9}{'ent后':>8}"
           f"{'激活前':>8}{'激活后':>8}")
    print(hdr)
    print("-" * 100)
    h1_votes = 0
    h1_total = 0
    for r in runs:
        tr, _ = data[r]
        if len(tr) < 6:
            continue
        k = len(tr) // 3
        a1, a2 = seg_mean(tr, "mean_abs_advantage", 0, k), seg_mean(tr, "mean_abs_advantage", len(tr) - k, len(tr))
        c1, c2 = seg_mean(tr, "value_return_corr", 0, k), seg_mean(tr, "value_return_corr", len(tr) - k, len(tr))
        kl = seg_mean(tr, "kl", len(tr) - k, len(tr))
        ent = seg_mean(tr, "entropy", len(tr) - k, len(tr))
        e1 = seg_mean(tr, "mean_activated_edges", 0, k)
        e2 = seg_mean(tr, "mean_activated_edges", len(tr) - k, len(tr))
        shrink = a2 / a1 if a1 else float("nan")
        if isinstance(shrink, float) and not math.isnan(shrink):
            h1_total += 1
            if shrink < 0.8:
                h1_votes += 1
        print(f"{r:<24}{len(tr):>4}{f(a1,9)}{f(a2,9)}{f(shrink,8,2)}"
              f"{f(c1,8,3)}{f(c2,8,3)}{f(kl,9,4)}{f(ent,8,3)}"
              f"{f(e1,8,1)}{f(e2,8,1)}")
    print()
    print(f"H1 投票（|A| 后/前 < 0.8，即明显收缩）：{h1_votes} / {h1_total}")

    # ---- 3. 最长 run 的 |A| 与 corr 逐轮 ----
    if runs:
        longest = max(runs, key=lambda r: len(data[r][0]))
        tr = data[longest][0]
        if len(tr) >= 8:
            print()
            print(f"最长 run = {longest}（{len(tr)} 轮）逐轮：")
            for key in ("mean_abs_advantage", "value_return_corr", "kl", "mean_activated_edges",
                        "mean_success_rate", "mean_served_keys"):
                cells = []
                for r in tr:
                    v = r.get(key)
                    if isinstance(v, (int, float)):
                        cells.append(f"{v:6.3f}" if abs(v) < 1000 else f"{v:6.0f}")
                    else:
                        cells.append("   -- ")
                print(f"  {key:<22}")
                for i in range(0, len(cells), 12):
                    print("    " + " ".join(cells[i:i + 12]))


if __name__ == "__main__":
    main()
