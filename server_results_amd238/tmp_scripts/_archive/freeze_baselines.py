"""Merge the non-learning baselines into one frozen reference file.

The point of freezing: the heuristics are not what is being iterated on. They
depend only on the scenario and their own fixed parameters, so re-running them
every time an RL variant is tested burns hours to reproduce numbers that cannot
have changed. This writes them once to a canonical file that later RL evals read
instead.

What is frozen: random, the six greedy_* baselines, and the project's own
expert (PathScoreGreedy phased + ServeProbe). What is deliberately NOT frozen:
any RL checkpoint -- that is the thing under test, and it changes every run.

Two protocols, because they are different tasks and answer different questions:

  heldout_240    days 330-365, 240 steps, 15 seeds -- the held-out
                 generalisation protocol (configs/global.yaml's own profile)
  fullday_1440   days 0-295, 1440 steps, 8 seeds (7..14) -- one full day on the
                 training window, i.e. the scenario the training arms replay

Sources are the four directories already produced by
.tmp/run_baselines_eval.sh, .tmp/run_baselines_fullday.sh, .tmp/eval_psg.py.

    python .tmp/freeze_baselines.py
"""
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "eval"


def mean_success_from_summary(path: Path, policy: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    run = data["policies"][policy]["runs"][-1]
    log = run["episode_log"]
    srs = [float(e["success_rate"]) for e in log]
    return {
        "episodes": len(srs),
        "mean_success": sum(srs) / len(srs),
        "std_success": statistics.pstdev(srs) if len(srs) > 1 else 0.0,
        "failure_rate": (
            sum(float(e["failed_keys"]) for e in log)
            / sum(float(e["arrived_keys"]) for e in log)
        ),
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
    }


def expert_from_json(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    srs = [float(v) for v in d["success"]]
    return {
        "episodes": len(srs),
        "mean_success": sum(srs) / len(srs),
        "std_success": statistics.pstdev(srs) if len(srs) > 1 else 0.0,
        "failure_rate": sum(d["failed"]) / sum(d["arrived"]),
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
    }


BASELINE_POLICIES = [
    "random", "greedy_rate", "greedy_qkp", "greedy_demand",
    "greedy_matching", "greedy_relay", "greedy_relay_diffusion_v3",
]

PROTOCOLS = {
    "heldout_240": {
        "description": "held-out generalisation window, quarter-day episodes",
        "window_days": [330, 365],
        "episode_steps": 240,
        "seeds": list(range(7, 22)),
        "episodes_per_policy": 15,
        "summary": EVAL / "heur_vs_rl" / "summary.json",
        "expert_json": EVAL / "psg_standard_240.json",
    },
    "fullday_1440": {
        "description": "training window, one full day per episode (the scenario training replays)",
        "window_days": [0, 295],
        "episode_steps": 1440,
        "seeds": list(range(7, 15)),
        "episodes_per_policy": 8,
        "summary": EVAL / "heur_full_day" / "summary.json",
        "expert_json": EVAL / "psg_standard_1440.json",
    },
}

out = {
    "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "note": (
        "Non-learning baselines only. Re-run nothing here when testing an RL "
        "variant -- these numbers depend only on the scenario and fixed policy "
        "parameters. RL checkpoints are deliberately absent: they are the thing "
        "under test."
    ),
    "protocols": {},
}

problems = []
for name, spec in PROTOCOLS.items():
    entry = {
        k: v for k, v in spec.items()
        if k not in ("summary", "expert_json")
    }
    entry["policies"] = {}

    if not spec["summary"].exists():
        problems.append(f"{name}: missing {spec['summary']}")
    else:
        for pol in BASELINE_POLICIES:
            try:
                entry["policies"][pol] = mean_success_from_summary(spec["summary"], pol)
            except KeyError:
                problems.append(f"{name}: policy '{pol}' absent from {spec['summary'].name}")

    if not spec["expert_json"].exists():
        problems.append(f"{name}: missing {spec['expert_json']} (run .tmp/eval_psg.py)")
    else:
        entry["policies"]["path_score_greedy_phased"] = expert_from_json(spec["expert_json"])

    out["protocols"][name] = entry

try:
    rev = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
except Exception:
    rev = "unknown"
out["git_rev"] = rev

dest = EVAL / "frozen_baselines.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

for name, entry in out["protocols"].items():
    print(f"=== {name}  ({entry['description']}) ===")
    print(f"    days {entry['window_days'][0]}-{entry['window_days'][1]}, "
          f"{entry['episode_steps']} steps, {entry['episodes_per_policy']} seeds")
    rows = sorted(entry["policies"].items(), key=lambda kv: -kv[1]["mean_success"])
    for pol, m in rows:
        print(f"      {pol:<30} {m['mean_success']:.4f} ± {m['std_success']:.4f}"
              f"   fail {m['failure_rate']*100:5.1f}%")
    print()

print(f"wrote {dest.relative_to(ROOT)}  (git {rev})")
if problems:
    print("INCOMPLETE:")
    for p in problems:
        print("  -", p)
else:
    print("all policies present in both protocols")
