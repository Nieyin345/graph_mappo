"""Pair expert / BC warm start / trained RL on the held-out protocol.

All three were evaluated on the same 15 seeds (7..21) with the same profile
(days 330-365, 240 steps, random_day), so seed i is the same day and the same
request streams for all of them. That makes every pairwise difference pairable,
which matters enormously here: per-seed success spans 0.29-0.88, so an unpaired
15-seed mean carries a standard error near 0.06 and cannot separate these
policies at all.

What each difference means:
    expert - BC    the clone loss. BC is distilled from this expert, so this is
                   what imitation failed to capture, and it is the distance PPO
                   has to climb before it can even reach the expert.
    RL - BC        what 30 PPO updates actually bought.
    RL - expert    the headline: does the trained policy beat the heuristic.

    python .tmp/compare_three.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "eval"


def from_expert_json(path: Path) -> dict[int, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {int(s): float(v) for s, v in zip(d["seeds"], d["success"])}


def from_summary(path: Path, policy: str) -> dict[int, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    log = d["policies"][policy]["runs"][-1]["episode_log"]
    return {int(e["seed"]): float(e["success_rate"]) for e in log}


series = {
    "expert": from_expert_json(EVAL / "psg_standard_240.json"),
    "BC": from_summary(EVAL / "bc_heldout" / "summary.json", "bc_warmstart"),
    "RL": from_summary(EVAL / "heur_vs_rl" / "summary.json", "rl_d8_final"),
}

common = sorted(set.intersection(*(set(s) for s in series.values())))
n = len(common)
print(f"paired on {n} seeds: {common}")
print()
print(f"  {'seed':>5} {'expert':>9} {'BC':>9} {'RL':>9}")
for s in common:
    print(f"  {s:>5} {series['expert'][s]:>9.4f} {series['BC'][s]:>9.4f} {series['RL'][s]:>9.4f}")


def paired(a: str, b: str) -> None:
    diffs = [series[b][s] - series[a][s] for s in common]
    md = sum(diffs) / n
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    t = md / se if se > 0 else float("nan")
    verdict = "no reliable difference"
    if abs(t) >= 2:
        verdict = f"{b} better" if md > 0 else f"{a} better"
    print(f"  {b:>6} - {a:<6} = {md:+.4f}   se {se:.5f}   t {t:+.2f}   -> {verdict}")


print()
for a, b in (("expert", "BC"), ("BC", "RL"), ("expert", "RL")):
    paired(a, b)

print()
for name, s in series.items():
    print(f"  mean {name:<7} = {sum(s[k] for k in common) / n:.4f}")
