"""Trend analysis for a finished arm.

The two record kinds in metrics.jsonl need different treatment:

  * training records are only comparable across updates when the arm replays the
    same scenario (fixed_episode_seed: true). For the random-day arm they are
    different samples each update, so a "trend" there is meaningless -- which is
    exactly why the pair exists.
  * eval_validation records use held-out days, so they are comparable across
    updates for BOTH arms, but they run on a shorter horizon than training
    (documents a different task -- see validation.episode_steps).

For each series this prints the least-squares slope per step with its standard
error, so "no trend" can be stated with a number rather than eyeballed.

    python scripts/eval/analyze_arms.py outputs/full_rnd15/metrics.jsonl outputs/full_d8/metrics.jsonl
"""

from __future__ import annotations
import json
import math
import sys


def series(path: str, key: str) -> list[float]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "eval_validation" in rec:
                if key == "eval":
                    out.append(float(rec["eval_validation"]["mean_success_rate"]))
            elif "update" in rec:
                if key == "train" and rec.get("mean_success_rate") is not None:
                    out.append(float(rec["mean_success_rate"]))
    return out


def slope(values: list[float]) -> tuple[float, float, float, float]:
    """Least-squares slope per step: (slope, stderr, first, last)."""
    n = len(values)
    if n < 3:
        return float("nan"), float("nan"), values[0] if values else float("nan"), float("nan")
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, values))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, values)]
    dof = n - 2
    s2 = sum(r * r for r in resid) / dof
    se = math.sqrt(s2 / sxx) if sxx > 0 else float("nan")
    return b, se, values[0], values[-1]


def report(name: str, values: list[float], unit: str) -> None:
    if not values:
        print(f"  {name}: (none)")
        return
    b, se, first, last = slope(values)
    n = len(values)
    half = n // 2
    h1 = sum(values[:half]) / half
    h2 = sum(values[half:]) / (n - half)
    t = b / se if se and se > 0 else float("nan")
    verdict = "NO TREND" if abs(t) < 2 else ("UP" if b > 0 else "DOWN")
    print(f"  {name} (n={n}, {unit})")
    print(f"    first {first:.4f}  last {last:.4f}  "
          f"min {min(values):.4f}  max {max(values):.4f}")
    print(f"    first-half mean {h1:.4f}  second-half mean {h2:.4f}  "
          f"(diff {h2 - h1:+.4f})")
    print(f"    slope {b:+.5f}/step  stderr {se:.5f}  t={t:+.2f}  -> {verdict}")


for path in sys.argv[1:]:
    name = path.replace("\\", "/").split("/")[-2]
    print(f"=== {name} ===")
    report("training success", series(path, "train"), "comparable only for the fixed arm")
    report("held-out success", series(path, "eval"), "held-out days, short horizon")
    print()
