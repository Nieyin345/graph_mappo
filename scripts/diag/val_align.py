#!/usr/bin/env python
"""定位每个验证点落在第几轮。

metrics.jsonl 里验证是**独立一行**（{"eval_validation": {...}}），自身不带
update 号，所以必须靠它前面最近的训练行的 update 来定位。之前的表把
它显示成 uNone，等于没有对齐信息 —— 这会让"u5 vs u15"这类判断失去依据。

用法：python .tmp/val_align.py r8_base_s42 ent01_s42 ...
     python .tmp/val_align.py --prefix ent01
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def align(d: Path, interval: int | None):
    m = d / "metrics.jsonl"
    if not m.exists():
        return None, []
    out = []
    last_u = None
    with m.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "update" in r:
                last_u = r["update"]
            ev = r.get("eval_validation")
            if isinstance(ev, dict) and ev.get("mean_success_rate") is not None:
                out.append((last_u, float(ev["mean_success_rate"]),
                            ev.get("per_seed_success") or []))
    return last_u, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--prefix", action="append", default=None)
    args = ap.parse_args()

    names = list(args.runs)
    for p in args.prefix or []:
        names += sorted(os.path.basename(x) for x in glob.glob(str(OUT / f"{p}*"))
                        if os.path.isdir(x))
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)

    print(f"{'run':<24}{'末轮':>5}{'验证点数':>9}  验证点(轮:值)")
    print("-" * 92)
    for n in uniq:
        last_u, vals = align(OUT / n, None)
        if last_u is None:
            continue
        s = "  ".join(f"{u}:{v:.4f}" for u, v, _ in vals) or "(无)"
        print(f"{n:<24}{last_u:>5}{len(vals):>9}  {s}")


if __name__ == "__main__":
    main()
