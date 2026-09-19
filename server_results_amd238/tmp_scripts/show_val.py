#!/usr/bin/env python
"""读 outputs/<run>/metrics.jsonl，打印每个 run 的 update 数与验证点。

用法：
  python /tmp/show_val.py ent01_s42 ent01_s43 ...      # 指定 run
  python /tmp/show_val.py --prefix ent01               # 前缀匹配
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def read_run(d: Path):
    m = d / "metrics.jsonl"
    if not m.exists():
        return None
    rows = []
    with m.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return None
    vals = []
    for r in rows:
        v = r.get("validation_success_rate")
        if v is None:
            v = (r.get("validation") or {}).get("success_rate")
        if v is not None:
            vals.append((r.get("update"), float(v)))
    last = rows[-1]
    return {
        "updates": len(rows),
        "vals": vals,
        "last_entropy": last.get("entropy"),
        "last_success": last.get("success_rate"),
        "mtime": os.path.getmtime(m),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--prefix", default=None)
    args = ap.parse_args()

    if args.prefix:
        names = sorted(os.path.basename(p) for p in glob.glob(str(OUT / f"{args.prefix}*"))
                       if os.path.isdir(p))
    else:
        names = args.runs

    for n in names:
        info = read_run(OUT / n)
        if info is None:
            print(f"{n:<24} (无 metrics)")
            continue
        vs = "  ".join(f"u{u}:{v:.4f}" for u, v in info["vals"]) or "(无验证点)"
        print(f"{n:<24} u={info['updates']:<3} {vs}")


if __name__ == "__main__":
    main()
