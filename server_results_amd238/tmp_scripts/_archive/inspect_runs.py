"""Read the real experiment runs: per-update timings and the config they used.

The two phases are timed separately by the trainer, so the interesting question
is whether a slow round is slow in BOTH phases (external contention -- something
else was competing for the CPU) or in only one (an algorithmic/implementation
effect). This prints the wall-clock gaps between consecutive updates, which
answers that directly.

Usage:
    python .tmp/inspect_runs.py full_rnd15 full_d8
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/opt/qkd/graph_mappo/outputs")


def parse_ts(value):
    if not isinstance(value, str):
        return None
    text = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def main() -> None:
    names = sys.argv[1:] or ["full_rnd15", "full_d8"]
    for name in names:
        path = ROOT / name / "metrics.jsonl"
        print(f"=== {name} ===")
        if not path.exists():
            print("  (missing)")
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        prev_end = None
        print(f"  {'upd':>3} {'rollout_s':>10} {'update_s':>9} {'elapsed_s':>10} "
              f"{'gap since prev end':>20} {'nb':>4}")
        for r in rows:
            start = parse_ts(r.get("timestamp"))
            gap = "—"
            if start is not None and prev_end is not None:
                gap = f"{(start - prev_end).total_seconds():.1f} s"
            if start is not None and r.get("elapsed_s"):
                prev_end = start + __import__("datetime").timedelta(seconds=float(r["elapsed_s"]))
            print(f"  {r.get('update', '?'):>3} {r.get('rollout_s', 0):>10.1f} "
                  f"{r.get('update_s', 0):>9.1f} {r.get('elapsed_s', 0):>10.1f} "
                  f"{gap:>20} {r.get('n_minibatches', '?')!s:>4}")
        print()


if __name__ == "__main__":
    main()
