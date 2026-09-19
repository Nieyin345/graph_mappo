"""把 metrics.jsonl 里**所有**键列出来，看哪些诊断量已经在记了。

读了半天 checkpoint 才发现「critic 漂移 1.43 / actor 0.047」，
但如果 `metrics.jsonl` 里本来就有 `value_return_corr` 这类字段，
那这条线索**一直摆在那儿没人看** —— 先确认有没有，再决定要不要写新探针。

用法（服务器上）：/opt/qkd/venv/bin/python /tmp/probe_metric_keys.py [run-name ...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def walk(o, prefix="", out=None):
    """展开嵌套 dict，叶子记成点分路径。"""
    if out is None:
        out = Counter()
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{prefix}{k}.", out)
    elif isinstance(o, list):
        out[prefix.rstrip(".") + "[]"] += 1
    else:
        out[prefix.rstrip(".")] += 1
    return out


def main(argv: list[str]) -> int:
    runs = argv[1:] or ["mode_de_s42"]
    for run in runs:
        p = OUT / run / "metrics.jsonl"
        if not p.exists():
            print(f"✗ {run}: 无 metrics.jsonl")
            continue
        keys: Counter = Counter()
        n = 0
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            keys.update(walk(o))
        print(f"=== {run}  （{n} 行，{len(keys)} 个叶子键）===")
        for k in sorted(keys):
            print(f"    {k}")
        print()

    # 专门找"critic 拟合质量"相关的键
    print("=== 含 corr / value / adv / kl / entropy 的键 ===")
    hit = [k for k in sorted(keys)
           if any(s in k.lower() for s in ("corr", "value", "adv", "kl", "entrop"))]
    for k in hit:
        print(f"    {k}")
    if not hit:
        print("    （无）—— 那 critic 拟合质量**当前没有被记录**，这条线索得靠新探针")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
