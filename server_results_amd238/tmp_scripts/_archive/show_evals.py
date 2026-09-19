"""Held-out validation series for one or more runs.

metrics.jsonl interleaves two record kinds: per-update training records
(`"update": N`) and held-out validation records (`"eval_validation": {...}`).
Counting lines therefore overstates the update count -- read the keys.

    python .tmp/show_evals.py outputs/full_rnd15/metrics.jsonl outputs/full_d8/metrics.jsonl
"""
import json
import sys


def load(path: str) -> tuple[list, list]:
    updates, evals = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "eval_validation" in rec:
                evals.append(rec["eval_validation"])
            elif "update" in rec:
                updates.append(rec)
    return updates, evals


for path in sys.argv[1:]:
    name = path.replace("\\", "/").split("/")[-2]
    updates, evals = load(path)
    print(f"=== {name} ===")
    print(f"  training updates logged: {len(updates)}")
    if updates:
        tr = [u.get("mean_success_rate") for u in updates if u.get("mean_success_rate") is not None]
        if tr:
            print(f"  training success: first {tr[0]:.4f}  last {tr[-1]:.4f}  "
                  f"min {min(tr):.4f}  max {max(tr):.4f}")
    print(f"  HELD-OUT validation points: {len(evals)}")
    for i, v in enumerate(evals, 1):
        print(f"    eval {i:2d}: success={v.get('mean_success_rate'):.4f}  "
              f"reward={v.get('mean_reward'):8.3f}  served={v.get('mean_served_keys'):.0f}")
    if len(evals) >= 2:
        a, b = evals[0].get("mean_success_rate"), evals[-1].get("mean_success_rate")
        print(f"  held-out trend: {a:.4f} -> {b:.4f}  ({b - a:+.4f})")
    print()
