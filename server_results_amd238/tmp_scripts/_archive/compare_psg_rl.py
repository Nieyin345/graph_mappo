"""Paired comparison: the project's heuristic vs the RL checkpoint, on identical scenarios.

Both were evaluated with the standard validation profile (configs/global.yaml:
days 330-365, 240-step episodes, random_day start) on the SAME seed list, so
seed i is the same day and the same request stream for both. Pairing cancels the
day-to-day variance -- which here is enormous, per-seed success spans 0.29 to
0.88 -- and is the only way this comparison can resolve anything: unpaired, a
15-seed mean carries a standard error around 0.04, far too coarse to separate
two policies that differ by 0.02.

    python .tmp/compare_psg_rl.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

psg = json.loads((ROOT / "outputs" / "eval" / "psg_standard_240.json").read_text(encoding="utf-8"))
summary = json.loads((ROOT / "outputs" / "eval" / "heur_vs_rl" / "summary.json").read_text(encoding="utf-8"))

rl_policy = "rl_d8_final"
runs = summary["policies"][rl_policy]["runs"]
rl = {int(e["seed"]): float(e["success_rate"]) for e in runs[-1]["episode_log"]}
expert = {int(s): float(v) for s, v in zip(psg["seeds"], psg["success"])}

common = sorted(set(rl) & set(expert))
if not common:
    raise SystemExit("no overlapping seeds between the two evaluations")

diffs = []
print(f"{'seed':>5} {'expert':>10} {'RL':>10} {'diff':>10}")
for s in common:
    d = rl[s] - expert[s]
    diffs.append(d)
    print(f"{s:>5} {expert[s]:>10.4f} {rl[s]:>10.4f} {d:>+10.4f}")

n = len(diffs)
mean_e = sum(expert[s] for s in common) / n
mean_r = sum(rl[s] for s in common) / n
mean_d = sum(diffs) / n
var = sum((d - mean_d) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
se = math.sqrt(var / n) if n > 1 else float("nan")
t = mean_d / se if se and se > 0 else float("nan")

print()
print(f"  expert (PathScoreGreedy phased)  mean = {mean_e:.4f}")
print(f"  RL     ({rl_policy})             mean = {mean_r:.4f}")
print(f"  paired difference (RL - expert) = {mean_d:+.4f}")
print(f"  paired stderr = {se:.5f}   t = {t:+.2f}   dof = {n - 1}")
if abs(t) >= 2:
    print(f"  -> {'RL better' if mean_d > 0 else 'EXPERT better'} (|t| >= 2)")
else:
    print("  -> no reliable difference (|t| < 2)")

sa = [expert[s] for s in common]
sb = [rl[s] for s in common]
va = sum((x - mean_e) ** 2 for x in sa) / (n - 1)
vb = sum((x - mean_r) ** 2 for x in sb) / (n - 1)
se_un = math.sqrt(va / n + vb / n)
print(f"  (unpaired stderr would be {se_un:.5f} -- pairing is {se_un / se:.1f}x tighter)")
