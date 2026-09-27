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

★ 2026-09-27：与 RL 配对时**必须**用 ``--run <训练目录>``。不加 --run 时本脚本
从 default + env_full 拼配置（legacy_v1 请求分布），而 RL 在自己的训练 config
（如 pairhot_v2 + arrival_rate）上验证 —— 实测 1× 下专家两种读数差 0.013。
--run 走 trainer 验证的同一实现（test_protocol.run_validation_env_config），
输出带 task_fingerprint，与 RL 的 validation.json / eval_*.json 指纹一致才可配对。

    python scripts/eval/eval_expert.py --run outputs/experiments/v8/<run> \\
        [--seeds 1000-1029] [--overlay pair_seed1.yaml] --out outputs/eval/<name>.json
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


def _overlay_path(name: str) -> Path:
    p = Path(name)
    return p if p.exists() else ROOT / "configs" / name


def build_policy(name: str, env, obs, config: dict, weights=(1.0, 10.0, 1.0, 0.5, 0.2),
                 milp_window: int = 30, milp_replan: int = 10, milp_time_limit: float = 30.0,
                 milp_max_requests: int = 64, milp_max_paths: int = 8):
    """每个回合构建一次（两种策略都持有 live env 的引用/状态）。"""
    if name == "rh_milp":
        import copy
        from qkd_rl.baselines.receding_horizon_milp import RecedingHorizonMILPPolicy
        pol = RecedingHorizonMILPPolicy(
            config, slot_seconds=float(env.scenario.slot_seconds),
            window_steps=milp_window, replan_every_steps=milp_replan,
            time_limit_s=milp_time_limit, max_requests=milp_max_requests,
            max_paths_per_request=milp_max_paths, switch_decay=0.5,
        )
        # 生成器有状态：预览会推进 RNG。给 MILP 一份 reset 后的深拷贝 ⟹
        # 预览 == env 实际到达，且不扰动 env 自己的请求流（共用则两边都错）。
        pol._ensure_init(obs, gen=copy.deepcopy(env.request_generator))
        return pol
    if name == "expert":
        return PathScoreGreedy(weights=tuple(weights), phased=True, principles=False,
                               router=ServeProbe(env))
    raise ValueError(f"unknown policy {name!r}")


def build_expert_task(run: str = "", config_path: str = "", seeds_spec: str = "",
                      steps: int = 0, overlays: list[str] | None = None
                      ) -> tuple[dict, list[int], int]:
    """→ (env_config, seeds, start_seed_base)。可测试的纯函数（无 env 构建）。

    run 非空：训练目录的 experiment_config.yaml + overlays → 与 trainer 验证
    同一实现。run 为空：旧的 canonical profile 路径（legacy_v1，仅供复现旧数）。
    """
    overlays = list(overlays or [])
    if run:
        config = _tp.load_config([Path(run) / "experiment_config.yaml"])
        for name in overlays:
            config = _tp.deep_merge(config, _tp.load_config([_overlay_path(name)]))
        vcfg = config.get("validation", {}) or {}
        seeds = parse_seeds(seeds_spec) if seeds_spec else _tp.run_validation_seeds(vcfg)
        env_config = _tp.run_validation_env_config(config, vcfg, steps or None)
        return env_config, seeds, int(vcfg.get("start_seed", 0) or 0)
    profile = _tp.load_validation_profile(
        Path(config_path or ROOT / "configs" / "global.yaml"))
    n_steps = steps or int(profile["episode_steps"]) or 240
    config = _tp.build_validation_env_config(
        profile, include_baselines=False, episode_steps=n_steps,
        start_mode=profile["start_mode"])
    for name in overlays:
        config = _tp.deep_merge(config, _tp.load_config([_overlay_path(name)]))
    return config, parse_seeds(seeds_spec or "7-21"), int(profile.get("start_seed", 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="",
                    help="训练目录：在该 run 的任务上评测（与 RL 配对时必须用）")
    ap.add_argument("--config", default="",
                    help="(无 --run 时) config whose global.validation block defines the protocol")
    ap.add_argument("--seeds", default="",
                    help="默认：--run 时用该 run 的验证种子；否则 7-21")
    ap.add_argument("--steps", type=int, default=0, help="0 = use the profile's episode_steps")
    ap.add_argument("--weights", nargs=5, type=float, default=[1.0, 10.0, 1.0, 0.5, 0.2])
    ap.add_argument("--out", default="", help="default: outputs/eval/expert_<steps>.json")
    ap.add_argument("--overlay", action="append", default=[],
                    help="extra config yaml(s) merged over the task config, "
                         "e.g. load_x20.yaml / pair_seed1.yaml")
    # 2026-09-27：可执行滚动时域 MILP（先知未来请求）作为「离线最优」的下沿。
    # 与流体 LP 上界（scripts/milp/fluid_upper_bound.py）一起夹住真实最优。
    ap.add_argument("--policy", choices=["expert", "rh_milp"], default="expert")
    ap.add_argument("--milp-window", type=int, default=30)
    ap.add_argument("--milp-replan", type=int, default=10)
    ap.add_argument("--milp-time-limit", type=float, default=30.0)
    ap.add_argument("--milp-max-requests", type=int, default=64)
    ap.add_argument("--milp-max-paths", type=int, default=8)
    args = ap.parse_args()

    config, seeds, start_seed_base = build_expert_task(
        args.run, args.config, args.seeds, args.steps, args.overlay)
    steps = int(config["env"]["episode_steps"])
    fingerprint = _tp.task_fingerprint(config)
    print(f"task: {'run ' + args.run if args.run else 'canonical profile (legacy)'}; "
          f"overlays {args.overlay}; window {config['env'].get('activation_window_start_day')}-"
          f"{config['env'].get('activation_window_end_day')}, steps {steps}, "
          f"{len(seeds)} seeds, fingerprint {fingerprint}")

    rows = []
    t0 = time.perf_counter()
    for seed in seeds:
        env = build_env_from_config(config)
        obs = env.reset(seed=seed, start_seed=start_seed_base + seed)
        expert = build_policy(
            args.policy, env, obs, config, weights=tuple(args.weights),
            milp_window=args.milp_window, milp_replan=args.milp_replan,
            milp_time_limit=args.milp_time_limit,
            milp_max_requests=args.milp_max_requests, milp_max_paths=args.milp_max_paths)
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
        "policy": "path_score_greedy_phased" if args.policy == "expert" else
                  f"rh_milp_w{args.milp_window}_r{args.milp_replan}_tl{args.milp_time_limit:g}",
        "steps": steps,
        "task_fingerprint": fingerprint,
        "run": args.run,
        "overlays": list(args.overlay),
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
