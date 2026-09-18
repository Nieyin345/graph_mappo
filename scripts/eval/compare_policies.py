"""Paired comparison of policies evaluated on the same seeds.

Every policy evaluated for the paper runs the same validation profile and the
same seed list, so seed i is the same day and the same request streams for all
of them. Differences taken within the seed therefore cancel the day-to-day
variance, which is the dominant term here: per-seed success spans roughly
0.29-0.88, so an unpaired 15-seed mean carries a standard error near 0.06 and
cannot separate policies that differ by 0.02. Pairing typically tightens that
5x (measured on the held-out protocol: 0.062 -> 0.012).

Two accepted sources:
    <summary.json>::<policy_key>   output of scripts/baselines/run_baselines.py
    <expert.json>                  output of scripts/eval/eval_expert.py
The source kind is detected by shape, so the ::key suffix is only needed for
the first kind.

    python scripts/eval/compare_policies.py \
        --policy expert=outputs/eval/expert_240.json \
        --policy BC=outputs/eval/bc_heldout/summary.json::bc_warmstart \
        --policy RL=outputs/eval/heur_vs_rl/summary.json::rl_d8_final
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(spec: str) -> tuple[str, dict[int, float]]:
    if "::" in spec:
        name, rest = spec.split("=", 1)
        path_part, key = rest.split("::", 1)
    else:
        name, path_part = spec.split("=", 1)
        key = None

    data = json.loads(Path(path_part).read_text(encoding="utf-8"))
    if "success" in data and "seeds" in data:
        return name, {int(s): float(v) for s, v in zip(data["seeds"], data["success"])}
    if key is None:
        raise SystemExit(f"{path_part}: looks like a run_baselines summary, "
                         f"so it needs a ::<policy_key> suffix")
    log = data["policies"][key]["runs"][-1]["episode_log"]
    return name, {int(e["seed"]): float(e["success_rate"]) for e in log}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", action="append", required=True,
                    metavar="NAME=PATH[::KEY]")
    args = ap.parse_args()

    series = dict(load(s) for s in args.policy)
    if len(series) < 2:
        raise SystemExit("need at least two policies to compare")

    names = list(series)
    common = sorted(set.intersection(*(set(v) for v in series.values())))
    n = len(common)
    if n < 3:
        raise SystemExit(f"only {n} shared seeds -- not enough to compare")

    width = max(len(x) for x in names) + 2
    print(f"paired on {n} seeds: {common}")
    print()
    print("  " + "seed".rjust(5) + "".join(x.rjust(width) for x in names))
    for s in common:
        print("  " + f"{s:>5}" + "".join(f"{series[x][s]:>{width}.4f}" for x in names))

    print()
    for name in names:
        print(f"  mean {name:<10} = {sum(series[name][s] for s in common) / n:.4f}")

    print()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diffs = [series[b][s] - series[a][s] for s in common]
            md = sum(diffs) / n
            var = sum((d - md) ** 2 for d in diffs) / (n - 1)
            se = math.sqrt(var / n)
            t = md / se if se > 0 else float("nan")
            verdict = "no reliable difference (|t| < 2)"
            if abs(t) >= 2:
                verdict = f"{b} better" if md > 0 else f"{a} better"
            print(f"  {b:>8} - {a:<8} = {md:+.4f}   se {se:.5f}   t {t:+.2f}   -> {verdict}")

    if len(names) > 2:
        # Unpaired contrast, to make the value of pairing explicit.
        print()
        for name in names:
            v = [series[name][s] for s in common]
            m = sum(v) / n
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
            print(f"  unpaired stderr for {name:<10} would be {sd / math.sqrt(n):.5f}")


if __name__ == "__main__":
    main()
