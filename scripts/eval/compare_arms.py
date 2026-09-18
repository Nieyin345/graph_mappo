"""Paired comparison of two arms that replay the SAME scenarios.

Both arms here run train_full_rl_fixdays.yaml, whose `fixed_episode_seed: true`
pins `_seed_stride` to 0: every update of every arm replays the identical
(base_seed + episode) seeds, hence the identical days and request streams. So
update i of arm A and update i of arm B are observations of the SAME scenario
under two policies -- a paired design, not two independent samples.

That matters for sensitivity. Comparing two means treats the between-scenario
variance (which days are hard) as noise; differencing within the pair cancels it
exactly, and what is left is only "did the policy do better on this same day".
With 8 fixed days the paired test resolves differences several times smaller
than an unpaired comparison of the same data.

Deliberately NOT compared: `mean_reward`. The arms have different reward
functions, so the scalar is on a different scale by construction -- a lower
total for the heavier penalty is expected and is not evidence about the policy.
Success rate and the physical counters (served keys) are the comparable ones.

    python .tmp/compare_arms.py outputs/rwd_ctrl/metrics.jsonl outputs/rwd_fail40/metrics.jsonl [--field mean_success_rate]
"""
import json
import math
import sys

args = sys.argv[1:]
FIELD = "mean_success_rate"
if "--field" in args:
    i = args.index("--field")
    FIELD = args[i + 1]
    del args[i:i + 2]


def updates(path: str, field: str) -> dict[int, float]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "update" in rec and rec.get(field) is not None:
                out[int(rec["update"])] = float(rec[field])
    return out


path_a, path_b = args[0], args[1]
name_a = path_a.replace("\\", "/").split("/")[-2]
name_b = path_b.replace("\\", "/").split("/")[-2]
a = updates(path_a, FIELD)
b = updates(path_b, FIELD)

common = sorted(set(a) & set(b))
if not common:
    print("no overlapping updates")
    raise SystemExit(1)

print(f"field = {FIELD}")
print(f"  {name_a:<22} n={len(a)}")
print(f"  {name_b:<22} n={len(b)}")
print(f"  paired on {len(common)} updates (both replay the same scenarios)")
print()
print(f"  {'upd':>4} {name_a:>12} {name_b:>12} {'diff':>10}")
diffs = []
for u in common:
    d = b[u] - a[u]
    diffs.append(d)
    print(f"  {u:>4} {a[u]:>12.4f} {b[u]:>12.4f} {d:>+10.4f}")

n = len(diffs)
mean_d = sum(diffs) / n
print()
print(f"  mean {name_a} = {sum(a[u] for u in common) / n:.4f}")
print(f"  mean {name_b} = {sum(b[u] for u in common) / n:.4f}")
print(f"  mean paired difference (B - A) = {mean_d:+.4f}")

if n >= 3:
    var = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    t = mean_d / se if se > 0 else float("nan")
    print(f"  paired stderr = {se:.5f}   t = {t:+.2f}   dof = {n - 1}")
    if abs(t) >= 2:
        print(f"  -> {'B better' if mean_d > 0 else 'A better'} (|t| >= 2)")
    else:
        print("  -> no reliable difference (|t| < 2)")
    # Unpaired, for contrast: how much sensitivity the pairing bought.
    sa = [a[u] for u in common]
    sb = [b[u] for u in common]
    ma, mb = sum(sa) / n, sum(sb) / n
    va = sum((x - ma) ** 2 for x in sa) / (n - 1)
    vb = sum((x - mb) ** 2 for x in sb) / (n - 1)
    se_un = math.sqrt(va / n + vb / n)
    print(f"  (unpaired stderr would be {se_un:.5f} -- "
          f"pairing is {se_un / se:.1f}x tighter)" if se > 0 else "")
