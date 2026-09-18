#!/usr/bin/env python
"""对比两臂在训练过程中的 entropy / 关键诊断量轨迹。

动机：entropy_coef 0.01 的收益只出现在早期（+0.0324 @u5 → +0.0133 @u15）。
若机理是"高 entropy 促进早期探索"，那么两臂的**熵轨迹**应该收敛到同一水平。
若是这样，正确的改动不是"更高的固定 entropy_coef"，而是**退火**
（早期高、后期降到 0.001），既拿早期的探索收益，又不付后期的代价。

本脚本不训练，只读 metrics.jsonl，纯文本。

用法：
  python scripts/diag/entropy_trace.py --runs r8_base_s42 ent01_s42 \
      --fields entropy kl mean_ratio value_return_corr
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def rows(name: str):
    p = OUT / name / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "update" in r:
                out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+",
                    default=["entropy", "kl", "mean_ratio", "value_return_corr"])
    args = ap.parse_args()

    for fld in args.fields:
        print(f"\n=== {fld} ===")
        series = {}
        for n in args.runs:
            rs = rows(n)
            series[n] = {r["update"]: r.get(fld) for r in rs
                         if r.get(fld) is not None}
        ups = sorted(set().union(*[set(s) for s in series.values()])) if series else []
        if not ups:
            print("  (无数据)")
            continue
        # 只显示有值的轮，全打会很长
        show = [u for u in ups if u <= 20]
        hdr = f"{'u':>3} " + " ".join(f"{n[:20]:>21}" for n in args.runs)
        print(hdr)
        for u in show:
            cells = []
            for n in args.runs:
                v = series[n].get(u)
                cells.append(f"{v:>21.4f}" if isinstance(v, (int, float)) else f"{'-':>21}")
            print(f"{u:>3} " + " ".join(cells))

    # 熵的两臂差（配对，同轮）
    if len(args.runs) == 2:
        a, b = args.runs
        ra = {r["update"]: r.get("entropy") for r in rows(a)}
        rb = {r["update"]: r.get("entropy") for r in rows(b)}
        common = sorted(set(ra) & set(rb))
        diffs = [(u, rb[u] - ra[u]) for u in common
                 if isinstance(ra.get(u), float) and isinstance(rb.get(u), float)]
        if diffs:
            print(f"\n=== entropy 差（{b} − {a}，同轮配对）===")
            for u, d in diffs[:20]:
                print(f"  u{u:>2}  {d:+.4f}")
            first = diffs[0][1]
            last = diffs[-1][1]
            print(f"\n  首轮 {first:+.4f} → 末轮 {last:+.4f}"
                  f"   衰减 {100*(1-last/first) if first else 0:.0f}%")


if __name__ == "__main__":
    main()
