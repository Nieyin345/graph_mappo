"""Aggregate a run_baselines.py summary.json into one row per policy.

The per-episode `episode_log` is what carries the physical counters, but it is
long (one entry per episode per policy) and unreadable as a whole. What the
structural-failure question needs is the mean over episodes, plus the failure
rate implied by served/arrived, which is the quantity the reward penalizes.

    python .tmp/summarize_baselines.py outputs/eval/heur_vs_rl/summary.json
"""
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)

rows = []
for name, entry in data.get("policies", {}).items():
    for run in entry.get("runs", []):
        log = run.get("episode_log", [])
        if not log:
            continue
        n = len(log)
        sr = sum(e.get("success_rate", 0.0) for e in log) / n
        served = sum(e.get("served_keys", 0.0) for e in log) / n
        failed = sum(e.get("failed_keys", 0.0) for e in log) / n
        arrived = sum(e.get("arrived_keys", 0.0) for e in log) / n
        rows.append((name, n, sr, served, failed, arrived))

if not rows:
    print("no episodes found in", path)
    raise SystemExit(1)

rows.sort(key=lambda r: -r[2])
print(f"{'policy':<26}{'eps':>4}{'success':>10}{'served':>16}{'failed':>16}{'arrived':>16}{'fail%':>8}")
for name, n, sr, served, failed, arrived in rows:
    fail_pct = failed / arrived * 100 if arrived else float("nan")
    print(f"{name:<26}{n:>4}{sr:>10.4f}{served:>16,.0f}{failed:>16,.0f}{arrived:>16,.0f}{fail_pct:>7.1f}%")

srs = [r[2] for r in rows]
print()
print(f"spread across {len(rows)} policies: min {min(srs):.4f}  max {max(srs):.4f}  "
      f"range {max(srs) - min(srs):.4f}")
