"""`actor.idle_scorer` 的 Δ 打印成 0.0000 —— 是精度问题，还是**真的没梯度**？

这条要问清楚，因为两者的处置完全不同：
  · 精度问题   → 无事发生
  · 真的没梯度 → 16,641 个参数是死重，且说明 IDLE 动作的路径有问题

判据（不靠打印的小数位）：
  1. `torch.equal(cur, base)` —— 逐位完全相同？
  2. 最大单元素差 `(cur-base).abs().max()`
  3. 该 run 的 `rollout_debug.jsonl` 里 IDLE 动作出现过没有

用法（服务器上）：/opt/qkd/venv/bin/python /tmp/probe_idle_dead.py [run-name]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

OUT = Path("/opt/qkd/graph_mappo/outputs")
BC = OUT / "supervised_pg_phased" / "supervised_pg_phased_latest.pt"
UPD_RE = re.compile(r"checkpoint_update_(\d+)\.pt$")
TARGETS = ["actor.idle_scorer", "actor.stop_logit", "actor.edge_scorer"]


def load(p: Path):
    return torch.load(p, map_location="cpu", weights_only=False)["model_state"]


def main(argv: list[str]) -> int:
    run = argv[1] if len(argv) > 1 else "mode_de_s42"
    d = OUT / run
    pts = sorted((int(UPD_RE.search(p.name).group(1)), p)
                 for p in d.glob("checkpoint_update_*.pt") if UPD_RE.search(p.name))
    if not pts:
        print(f"✗ {run} 无检查点")
        return 1
    base = load(BC)

    print(f"BC 起点 vs {run}   逐检查点看 idle_scorer")
    print()
    print(f"{'u':>4}  {'idle逐位相同?':<14}{'idle最大差':>13}{'stop最大差':>13}"
          f"{'edge最大差':>13}")
    print("-" * 62)
    for u, p in pts:
        cur = load(p)
        row = {}
        for t in TARGETS:
            ks = [k for k in base if k.startswith(t + ".") or k == t]
            if not ks:
                row[t] = ("n/a", float("nan"))
                continue
            same = all(torch.equal(cur[k], base[k]) for k in ks if k in cur)
            mx = max(float((cur[k] - base[k]).abs().max()) for k in ks if k in cur)
            row[t] = (same, mx)
        print(f"{u:>4}  {str(row['actor.idle_scorer'][0]):<14}"
              f"{row['actor.idle_scorer'][1]:>13.3e}"
              f"{row['actor.stop_logit'][1]:>13.3e}"
              f"{row['actor.edge_scorer'][1]:>13.3e}")

    # IDLE 动作到底出现过没有
    print()
    rj = d / "rollout_debug.jsonl"
    if not rj.exists():
        print("（无 rollout_debug.jsonl，跳过 IDLE 计数）")
        return 0
    print("rollout_debug.jsonl 里找 IDLE / stop / idle 字段：")
    keys: set[str] = set()
    idle_hits = 0
    lines = 0
    for line in rj.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        lines += 1
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict):
            for k, v in o.items():
                if any(s in k.lower() for s in ("idle", "stop", "skip")):
                    keys.add(k)
                    if isinstance(v, (int, float)) and v:
                        idle_hits += 1
    print(f"  读了 {lines} 行；含 idle/stop/skip 的键: {sorted(keys) or '（无）'}")
    print(f"  其中非零出现次数: {idle_hits}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
