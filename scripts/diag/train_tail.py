#!/usr/bin/env python
"""打印指定 run 在 u>=n 的训练诊断量，用于判断"某轮验证值突变"是否伴随训练异常。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--from-update", type=int, default=17)
    ap.add_argument("--fields", nargs="+", default=[
        "mean_success_rate", "mean_abs_advantage", "kl", "entropy",
        "rollout_s", "update_s", "actor_grad_norm",
    ])
    args = ap.parse_args()

    for run in args.runs:
        p = OUT / run / "metrics.jsonl"
        if not p.exists():
            print(f"=== {run}: 缺 metrics.jsonl ===")
            continue
        print(f"=== {run} ===")
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "update" in r:
                rows.append(r)
        rows.sort(key=lambda r: r["update"])
        for r in rows:
            if r["update"] < args.from_update:
                continue
            cells = []
            for f in args.fields:
                v = r.get(f)
                if v is None:
                    cells.append(f"{f}=--")
                elif f in ("rollout_s", "update_s"):
                    cells.append(f"{f}={float(v):.1f}s")
                elif f == "kl":
                    cells.append(f"{f}={float(v):.5f}")
                else:
                    cells.append(f"{f}={float(v):.4f}")
            print(f"  u{r['update']:>3}  " + "  ".join(cells))
        print()


if __name__ == "__main__":
    main()
