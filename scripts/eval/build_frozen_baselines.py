"""Merge the non-learning baselines into one frozen reference file.

The heuristics are not what gets iterated on. They depend only on the scenario
and their own fixed parameters, so re-running them every time an RL variant is
tested burns hours reproducing numbers that cannot have changed. Run this once,
freeze the result, and have later RL evals read the file instead.

Frozen: `random`, the six `greedy_*` baselines, and the project's expert
(PathScoreGreedy phased + ServeProbe).
Deliberately NOT frozen: any RL checkpoint -- that is the thing under test.

Two protocols, because they are different tasks:

  heldout_240    days 330-365, 240 steps, seeds 7..21 -- held-out
                 generalisation (configs/global.yaml's own profile)
  fullday_1440   days 0-295, 1440 steps, seeds 7..14 -- one full day on the
                 training window, the scenario the training arms replay

Inputs are produced by:
  scripts/baselines/run_baselines.py       -> <out>/summary.json
  scripts/eval/eval_expert.py              -> outputs/eval/expert_<steps>.json

    python scripts/eval/build_frozen_baselines.py
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "outputs" / "eval"

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
        "summary_dirs": ["bc_heldout", "heur_vs_rl"],
    },
    "fullday_1440": {
        "description": "training window, one full day per episode (what training replays)",
        "window_days": [0, 295],
        "episode_steps": 1440,
        "seeds": list(range(7, 15)),
        "summary_dirs": ["heur_full_day"],
    },
}


def merge_runs_summaries(dirs: list[str]) -> dict[str, dict]:
    """Later dirs are scanned too, so one policy set can be split across runs."""
    out: dict[str, dict] = {}
    for d in dirs:
        path = EVAL / d / "summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for pol, entry in data.get("policies", {}).items():
            log = entry.get("runs", [{}])[-1].get("episode_log", [])
            if log:
                out.setdefault(pol, {"log": log, "source": str(path.relative_to(ROOT)).replace("\\", "/")})
    return out


def stats_from_log(log: list[dict]) -> dict:
    srs = [float(e["success_rate"]) for e in log]
    return {
        "episodes": len(srs),
        "mean_success": sum(srs) / len(srs),
        "std_success": statistics.pstdev(srs) if len(srs) > 1 else 0.0,
        "failure_rate": (sum(float(e["failed_keys"]) for e in log)
                         / sum(float(e["arrived_keys"]) for e in log)),
    }


def expert_stats(steps: int) -> tuple[dict, str] | tuple[None, list[str]]:
    for name in (f"expert_{steps}.json", f"psg_standard_{steps}.json"):
        path = EVAL / name
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        srs = [float(v) for v in d["success"]]
        return ({
            "episodes": len(srs),
            "mean_success": sum(srs) / len(srs),
            "std_success": statistics.pstdev(srs) if len(srs) > 1 else 0.0,
            "failure_rate": sum(d["failed"]) / sum(d["arrived"]),
        }, str(path.relative_to(ROOT)).replace("\\", "/"))
    return None, [f"expert_{steps}.json"]


def main() -> None:
    out = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Non-learning baselines only. Do not re-run these when testing an "
                 "RL variant: they depend only on the scenario and on fixed policy "
                 "parameters. RL checkpoints are deliberately absent."),
        "protocols": {},
    }
    problems: list[str] = []

    for name, spec in PROTOCOLS.items():
        steps = spec["episode_steps"]
        entry = {k: v for k, v in spec.items() if k != "summary_dirs"}
        entry["policies"] = {}

        found = merge_runs_summaries(spec["summary_dirs"])
        for pol in BASELINE_POLICIES:
            if pol in found:
                entry["policies"][pol] = stats_from_log(found[pol]["log"])
                entry["policies"][pol]["source"] = found[pol]["source"]
            else:
                problems.append(f"{name}: policy '{pol}' not found in {spec['summary_dirs']}")

        exp, src = expert_stats(steps)
        if exp is None:
            problems.append(f"{name}: missing {src} (run scripts/eval/eval_expert.py)")
        else:
            entry["policies"]["path_score_greedy_phased"] = exp
            entry["policies"]["path_score_greedy_phased"]["source"] = src

        out["protocols"][name] = entry

    try:
        out["git_rev"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        out["git_rev"] = "unknown"

    dest = EVAL / "frozen_baselines.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for name, entry in out["protocols"].items():
        print(f"=== {name}  ({entry['description']}) ===")
        print(f"    days {entry['window_days'][0]}-{entry['window_days'][1]}, "
              f"{entry['episode_steps']} steps, {len(entry['seeds'])} seeds")
        rows = sorted(entry["policies"].items(), key=lambda kv: -kv[1]["mean_success"])
        for pol, m in rows:
            print(f"      {pol:<30} {m['mean_success']:.4f} ± {m['std_success']:.4f}"
                  f"   fail {m['failure_rate'] * 100:5.1f}%")
        print()

    print(f"wrote {dest.relative_to(ROOT)}  (git {out['git_rev']})")
    if problems:
        print("INCOMPLETE:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("all policies present in both protocols")


if __name__ == "__main__":
    main()
