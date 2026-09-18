"""Side-by-side reward-component breakdown for two runs.

`rollout_debug.jsonl` is written once per update by the single-process rollout
path (n_rollout_workers=1). Every key is already a per-step mean -- the writer
suffixes them all with `mean_`, including the reward terms (`mean_reward_served`
and friends). Keys that are zero across BOTH runs are still printed: "this term
contributes nothing" is the finding, not noise to be filtered.

    python .tmp/show_reward_breakdown.py A/rollout_debug.jsonl B/rollout_debug.jsonl
"""
import json
import sys

# Update-level training stats that ride along in the same record; they are
# already reported in metrics.jsonl, so keep them out of the reward table.
STATS = {
    "update", "actor_loss", "critic_loss", "entropy", "kl", "mean_ratio",
    "actor_grad_norm", "critic_grad_norm", "mean_return", "mean_abs_advantage",
    "mean_success_rate", "mean_reward", "value_std", "return_std",
    "value_return_corr", "n_minibatches",
}


def last_record(path: str) -> dict:
    rec = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
    return rec


paths = sys.argv[1:]
names = [p.replace("\\", "/").split("/")[-2] for p in paths]
recs = [last_record(p) for p in paths]

keys = []
for r in recs:
    for k in r:
        if k not in keys:
            keys.append(k)

reward_keys = [k for k in keys if "reward" in k and k not in STATS]
other_keys = [k for k in keys if k not in reward_keys and k not in STATS]

w = 13


def head() -> str:
    return f"  {'':<30}" + "".join(f"{n:>{w}}" for n in names) + f"{'delta':>{w}}"


def row(k: str) -> str:
    vals = [r.get(k) for r in recs]
    cells = ""
    for v in vals:
        cells += f"{v:>{w}.4g}" if isinstance(v, (int, float)) else f"{str(v):>{w}}"
    d = ""
    if all(isinstance(v, (int, float)) for v in vals) and len(vals) == 2:
        d = f"{vals[1] - vals[0]:>+{w}.4g}"
    return f"  {k:<30}{cells}{d}"


def total(rec: dict) -> float:
    return sum(rec.get(k, 0.0) for k in reward_keys if k != "mean_reward")


for title, ks in (("REWARD COMPONENTS (per-step means)", reward_keys),
                  ("PHYSICAL / DIAGNOSTIC COUNTERS", other_keys)):
    print(title)
    print(head())
    for k in ks:
        print(row(k))
        if "reward" in k and k != "mean_reward":
            a = recs[0].get(k, 0.0)
            ta = total(recs[0])
            if ta:
                print(f"  {'  ^ share of reward total':<30}{a / ta * 100:>{w}.1f}%")
    if title.startswith("REWARD"):
        print(f"  {'TOTAL':<30}" + "".join(f"{total(r):>{w}.4g}" for r in recs)
              + f"{total(recs[-1]) - total(recs[0]):>+{w}.4g}")
    print()
