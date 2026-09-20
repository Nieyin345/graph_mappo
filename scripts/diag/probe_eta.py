"""cmax 波的速率与 ETA。"""
import datetime
import json
import sys
from pathlib import Path

OUT = Path("/opt/qkd/graph_mappo/outputs")


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "cmax_s42"
    rows = []
    for line in (OUT / run / "metrics.jsonl").read_text(
            encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "update" in o:
            rows.append(o)
    if len(rows) < 3:
        print(f"  {run}: 只有 {len(rows)} 轮，样本不够")
        return
    last3 = rows[-3:]
    rs = sum(r.get("rollout_s", 0.0) for r in last3) / 3
    us = sum(r.get("update_s", 0.0) for r in last3) / 3
    per = rs + us
    done = rows[-1]["update"]
    left = 30 - done
    eta = datetime.datetime.now() + datetime.timedelta(seconds=left * per)
    print(f"  {run}: u{done}/30   每轮 {per:.0f}s (rollout {rs:.0f} + update {us:.0f})")
    print(f"    剩余 {left} 轮 = {left * per / 60:.0f} 分钟")
    print(f"    预计完成 {eta.strftime('%H:%M')} UTC")
    return per, left


if __name__ == "__main__":
    main()
