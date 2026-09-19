"""Evaluate the project's expert heuristic on a validation profile.

Why a dedicated script: `configs/baselines.yaml` does not list the project's
expert, so `scripts/baselines/run_baselines.py` never runs it -- and the expert
is the only baseline that matters. Benchmarks that omit it compare against
`greedy_relay`, a far weaker family (0.443 vs 0.708 on the held-out protocol),
and understate the bar by a factor of ~1.6.

The expert is constructed exactly as the BC warm start builds it
(`scripts/train/supervised_train_pg_phased.py`):
    PathScoreGreedy(weights=(1,10,1,0.5,0.2), phased=True, principles=False,
                    router=ServeProbe(env))

The env is driven the way `Evaluator` drives a baseline policy -- `act(obs)`
then `env.step(actions, scores)` -- so the numbers are comparable with
`run_baselines.py` output on the same profile.

Per-seed success is written to JSON so it can be paired on seed against an RL
checkpoint evaluated on the same seeds. Pairing is not optional here: per-seed
success spans 0.29-0.88, so an unpaired 15-seed mean carries a standard error
near 0.06 and cannot separate policies that differ by 0.02.

    python scripts/eval/eval_expert.py --seeds 7-21
    python scripts/eval/eval_expert.py --config <profile.yaml> --seeds 7-14 --steps 1440
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "global.yaml"),
                    help="config whose global.validation block defines the protocol")
    ap.add_argument("--seeds", default="7-21")
    ap.add_argument("--steps", type=int, default=0, help="0 = use the profile's episode_steps")
    ap.add_argument("--weights", nargs=5, type=float, default=[1.0, 10.0, 1.0, 0.5, 0.2])
    ap.add_argument("--out", default="", help="default: outputs/eval/expert_<steps>.json")
    args = ap.parse_args()

    profile = _tp.load_validation_profile(Path(args.config))
    steps = args.steps or int(profile["episode_steps"]) or 240
    seeds = parse_seeds(args.seeds)

    print(f"profile: window {profile['window_start_day']}-{profile['window_end_day']}, "
          f"steps {steps}, start_mode {profile['start_mode']}, {len(seeds)} seeds")

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
            weights=tuple(args.weights), phased=True, principles=False,
            router=ServeProbe(env),
        )
        served = 0.0
        # 2026-09-19 补：把**生成量**与**等待量**也记下来。
        #
        # 为什么：训练侧实测 RL 的密钥生成量单调掉 **41%**（10.93M → 6.47M），
        # 而 `mean_reward_generated = 0`、`success_rate` 与 reward 都不动
        # （见 docs/训练诊断记录.md「奖励看不见的行为漂移」）。也就是说奖励有一个
        # 很大的零空间，策略在里面自由漂移。要判断这个漂移**是好是坏**，
        # 必须有**同一 regime 上专家**的同一组量 —— 而本脚本原先只存
        # success/served/failed/arrived，正好缺了有漂移的那两维。
        #
        # `served_keys`/`generated_keys` 是**每步流量**，累加即为整局总量；
        # `waiting_keys` 是**存量**，累加得到"等待密钥·步"，因此同时存均值
        # （= 总和/步数）以便与 rollout_debug 的 `mean_waiting_keys` 直接比。
        generated = 0.0
        waiting = 0.0
        qkp_util = 0.0
        n_steps = 0
        done = False
        while not done:
            actions, scores = expert.act(obs)
            obs, _reward, terminated, truncated, info = env.step(actions, scores)
            served += float(info.get("served_keys", 0.0))
            generated += float(info.get("generated_keys", 0.0))
            waiting += float(info.get("waiting_keys", 0.0))
            qkp_util += float(info.get("qkp_utilization", 0.0))
            n_steps += 1
            done = terminated or truncated

        summary = env.metrics.episode_summary()
        arrived = float(summary.get("arrived_keys", 0.0))
        failed = float(summary.get("failed_keys", 0.0))
        rows.append((seed, served / arrived if arrived else 0.0, served, failed, arrived,
                     generated, waiting / n_steps if n_steps else 0.0,
                     qkp_util / n_steps if n_steps else 0.0))
        print(f"  seed {seed:>3}  success {rows[-1][1]:.4f}  "
              f"served {served:>13,.0f}  failed {failed:>13,.0f}"
              f"  gene {generated:>13,.0f}")

    n = len(rows)
    mean_sr = sum(r[1] for r in rows) / n
    print()
    print(f"  mean success = {mean_sr:.4f}")
    print(f"  min {min(r[1] for r in rows):.4f}  max {max(r[1] for r in rows):.4f}")
    print(f"  failure rate = {sum(r[3] for r in rows) / sum(r[4] for r in rows) * 100:.1f}%")
    print(f"  elapsed {time.perf_counter() - t0:.1f}s")

    out_path = Path(args.out) if args.out else ROOT / "outputs" / "eval" / f"expert_{steps}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "policy": "path_score_greedy_phased",
        "steps": steps,
        "seeds": [r[0] for r in rows],
        "success": [r[1] for r in rows],
        "served": [r[2] for r in rows],
        "failed": [r[3] for r in rows],
        "arrived": [r[4] for r in rows],
        # 2026-09-19 新增：训练侧发现的那两维（生成/等待）+ 利用率。
        # 旧文件（如 expert_seeds100_240.json）**没有这些键**——缺键不是 0，
        # 读的时候要判存在，别把"没测"读成"为 0"。
        "generated": [r[5] for r in rows],
        "waiting_mean": [r[6] for r in rows],
        "qkp_util_mean": [r[7] for r in rows],
    }, indent=2), encoding="utf-8")
    print(f"  wrote {out_path}")
    if rows and rows[0][2]:
        g = sum(r[5] for r in rows) / n
        s = sum(r[2] for r in rows) / n
        print(f"  ★ 生成/服务 = {g / s:.2f}（训练侧 RL 末轮是 ~97.3）")


if __name__ == "__main__":
    main()
