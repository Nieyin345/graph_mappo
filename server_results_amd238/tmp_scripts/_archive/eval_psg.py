"""Evaluate the project's own heuristic on the standard validation protocol.

Why this script exists: `configs/baselines.yaml` does not list the project's
expert, so `scripts/baselines/run_baselines.py` never runs it. Benchmarking
"against the best baseline" therefore compared against `greedy_relay`, a much
weaker family -- docs/启发式与强化学习算法说明.md §3.2 measures 0.435 for
`greedy_relay` and 0.810 for the expert on the SAME protocol. Any comparison
that omits the expert is off by a factor of ~1.9.

The expert is constructed exactly as the BC warm start builds it
(scripts/train/supervised_train_pg_phased.py:300):
    PathScoreGreedy(weights=(1,10,1,0.5,0.2), phased=True, principles=False,
                    router=ServeProbe(env))

Protocol mirrors run_baselines.py: the profile comes from configs/global.yaml's
`global.validation`, and the env is driven the way Evaluator drives a baseline --
`act(obs)` then `env.step(actions, scores)`. The default seed list matches
.tmp/run_baselines_eval.sh so the two sets of numbers sit side by side.

    python .tmp/eval_psg.py --seeds 7-21
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_spec = importlib.util.spec_from_file_location(
    "_tp", ROOT / "qkd_rl" / "evaluation" / "test_protocol.py")
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

from qkd_rl.env.factory import build_env_from_config
from qkd_rl.baselines.path_greedy import PathScoreGreedy
from qkd_rl.baselines.serve_probe import ServeProbe


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ROOT / "configs" / "global.yaml"))
ap.add_argument("--seeds", default="7-21")
ap.add_argument("--steps", type=int, default=0, help="0 = take the profile's episode_steps")
ap.add_argument("--weights", nargs=5, type=float, default=[1.0, 10.0, 1.0, 0.5, 0.2])
args = ap.parse_args()

profile = _tp.load_validation_profile(Path(args.config))
steps = args.steps or int(profile["episode_steps"]) or 240
seeds = parse_seeds(args.seeds)

print(f"profile: window {profile['window_start_day']}-{profile['window_end_day']}, "
      f"steps {steps}, start_mode {profile['start_mode']}, seeds {seeds}")

config = _tp.build_validation_env_config(
    profile, include_baselines=False, episode_steps=steps,
    start_mode=profile["start_mode"])

rows = []
t0 = time.perf_counter()
for seed in seeds:
    env = build_env_from_config(config)
    obs = env.reset(seed=seed, start_seed=int(profile.get("start_seed", 0)) + seed)
    # Built per env: ServeProbe holds a reference to the live env.
    expert = PathScoreGreedy(
        weights=tuple(args.weights),
        phased=True,
        principles=False,
        router=ServeProbe(env),
    )

    done = False
    served = 0.0
    while not done:
        actions, scores = expert.act(obs)
        obs, _reward, terminated, truncated, info = env.step(actions, scores)
        served += float(info.get("served_keys", 0.0))
        done = terminated or truncated

    summary = env.metrics.episode_summary()
    arrived = float(summary.get("arrived_keys", 0.0))
    failed = float(summary.get("failed_keys", 0.0))
    sr = served / arrived if arrived > 0 else 0.0
    rows.append((seed, sr, served, failed, arrived))
    print(f"  seed {seed:>3}  success {sr:.4f}  served {served:>13,.0f}  "
          f"failed {failed:>13,.0f}  arrived {arrived:>13,.0f}")

n = len(rows)
mean_sr = sum(r[1] for r in rows) / n
print()
print(f"PathScoreGreedy(phased=True)  seeds={n}  steps={steps}")
print(f"  mean success = {mean_sr:.4f}")
print(f"  min {min(r[1] for r in rows):.4f}  max {max(r[1] for r in rows):.4f}  "
      f"(per-seed spread {max(r[1] for r in rows) - min(r[1] for r in rows):.4f})")
print(f"  mean served  = {sum(r[2] for r in rows) / n:,.0f}")
print(f"  mean failed  = {sum(r[3] for r in rows) / n:,.0f}")
print(f"  mean arrived = {sum(r[4] for r in rows) / n:,.0f}")
print(f"  failure rate = {sum(r[3] for r in rows) / sum(r[4] for r in rows) * 100:.1f}%")
print(f"  elapsed {time.perf_counter() - t0:.1f}s")

# Persist per-seed values so the side-by-side with the RL checkpoint can be
# paired on seed (same days, same request streams -> the day-to-day variance
# cancels in the difference and the comparison gets far more sensitive than
# two means would be).
out_path = ROOT / "outputs" / "eval" / f"psg_standard_{steps}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(
    json.dumps(
        {
            "policy": "path_score_greedy_phased",
            "steps": steps,
            "seeds": [r[0] for r in rows],
            "success": [r[1] for r in rows],
            "served": [r[2] for r in rows],
            "failed": [r[3] for r in rows],
            "arrived": [r[4] for r in rows],
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"  wrote {out_path}")
print("PSG_DONE")
