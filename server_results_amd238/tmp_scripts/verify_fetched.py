#!/usr/bin/env python
"""比对服务器 outputs/ 与本地 server_results/ 的 run 清单，找缺的。

用法：python .tmp/verify_fetched.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

LOCAL = Path("server_results/outputs")
OLD = Path("server_results/runs/outputs")


def remote_runs():
    out = subprocess.run(
        ["ssh", "qkd", "ls /opt/qkd/graph_mappo/outputs"],
        capture_output=True, text=True, check=True).stdout
    return {l.strip() for l in out.splitlines() if l.strip()}


def local_runs(root: Path):
    if not root.exists():
        return set()
    return {p.name for p in root.iterdir()
            if p.is_dir() and (p / "metrics.jsonl").exists()}


def main():
    rem = remote_runs()
    new = local_runs(LOCAL)
    old = local_runs(OLD)
    print(f"服务器 outputs/     : {len(rem)} 个 run")
    print(f"本地 outputs/       : {len(new)} 个（本次抓取）")
    print(f"本地 runs/outputs/  : {len(old)} 个（早先抓取）")

    missing = rem - new - old
    print(f"\n两边都没有的：{len(missing)}")
    for m in sorted(missing)[:40]:
        print(f"  !! {m}")

    only_old = old - new - rem
    if only_old:
        print(f"\n只在 runs/ 里有（服务器已删或改名）：{len(only_old)}")
        for m in sorted(only_old)[:20]:
            print(f"  - {m}")

    print(f"\n合计本地可读 run：{len(new | old)}")
    need = rem - (new | old)
    print("结论：" + ("齐全" if not need else f"缺 {len(need)} 个"))


if __name__ == "__main__":
    main()
