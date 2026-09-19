#!/usr/bin/env python
"""对比两个 run 的 resolved_config，只打印**有差异**的字段。

为什么需要：outputs/ 下有几十个历史 run，配置差异埋在几百行 yaml 里。
找「哪个改动让结果变好」必须做差异对比，不能靠通读。

用法：
  python /tmp/diff_runs.py r7_base r7_fix_ent
  python /tmp/diff_runs.py r8_base_s43 r7_fix_ent
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

R = Path("/opt/qkd/graph_mappo/outputs")


def flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        out[prefix[:-1]] = str(d)
    else:
        out[prefix[:-1]] = d
    return out


def load(name):
    p = R / name / "resolved_config.yaml"
    if not p.exists():
        print(f"找不到 {p}")
        sys.exit(1)
    with p.open(encoding="utf-8") as fh:
        return flat(yaml.safe_load(fh) or {})


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    a_name, b_name = sys.argv[1], sys.argv[2]
    a, b = load(a_name), load(b_name)

    keys = sorted(set(a) | set(b))
    diffs = [(k, a.get(k, "<缺失>"), b.get(k, "<缺失>"))
             for k in keys if a.get(k, "<缺失>") != b.get(k, "<缺失>")]

    print(f"A = {a_name}")
    print(f"B = {b_name}")
    print(f"差异字段 {len(diffs)} / {len(keys)}\n")
    w = max((len(k) for k, _, _ in diffs), default=10)
    print(f"{'字段':<{w}}  {'A':<28}  {'B':<28}")
    print("-" * (w + 62))
    for k, va, vb in diffs:
        sa, sb = str(va)[:27], str(vb)[:27]
        print(f"{k:<{w}}  {sa:<28}  {sb:<28}")


if __name__ == "__main__":
    main()
