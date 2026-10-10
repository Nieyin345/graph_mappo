"""Unified BC, PPO, validation and batch entry point.

Examples:
    python scripts/experiment.py train --model v3 --seed 42 --updates 20 --bc
    python scripts/experiment.py evaluate --run outputs/experiments/v3/s42
    python scripts/experiment.py batch --models v2 v3 --seeds 42 43 --updates 20 --parallel 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.core.config import ConfigValidator, deep_merge, load_config, resolve_config_path
from scripts.train.train_graph_mappo import build_config as build_legacy_config


def resolve_experiment(model: str, seed: int, updates: int | None, overrides: list[str],
                       run_name: str, mode: str = "random_episode") -> dict:
    """Merge the existing environment/profile with the shared and model configs."""
    model_dir = ROOT / "qkd_rl" / "model_zoo" / model
    model_config = model_dir / "config.yaml"
    if not model.isidentifier() or model.startswith("_") or not model_config.is_file():
        raise ValueError(f"Unknown model {model!r}; expected model_zoo/<name>/config.yaml")
    legacy_args = argparse.Namespace(mode=mode, configs=None, seed=seed,
                                     num_updates=updates, run_name=run_name,
                                     device=None, resolved_config=None)
    config = build_legacy_config(legacy_args)
    config = deep_merge(config, load_config([ROOT / "configs" / "experiment_base.yaml", model_config]))
    for name in overrides:
        # Bare names resolve under configs/ (archive-aware); paths with a
        # separator (e.g. model_zoo/v3/config_demand_isolated.yaml) are
        # repo-relative so model dirs can own their variant configs.
        override_path = (ROOT / name) if ("/" in name or "\\" in name) else resolve_config_path(ROOT / "configs", name)
        config = deep_merge(config, load_config([override_path]))
    config.setdefault("experiment", {})["model"] = model
    config["seed"]["global_seed"] = seed
    config["seed"]["env_seed"] = seed
    config["project"]["output_dir"] = "outputs/experiments"
    config["project"]["run_name"] = f"{model}/{run_name}"
    if updates is not None:
        config["train"]["num_updates"] = updates
    ConfigValidator().validate(config)
    _check_training_budget(config)
    return config


def _check_training_budget(config: dict) -> None:
    """Refuse to start a run whose PPO budget is too small to train.

    Gradient steps per update = ceil(N / minibatch_size), where
    N = rollout_steps * episodes_per_update (the trainer chunks the buffer with
    `range(0, N, minibatch_size)`, mappo_trainer.update). Below the floor the
    policy barely moves: measured signature is clip_frac == 0 and kl ~ 1e-5 for
    the whole run -- not "worse results", but *no training*, so the artifact is
    just the BC starting point. Several archived batches are exactly this.

    This is checked here, not only in tests, because the failure is silent:
    nothing errors, metrics.jsonl fills with plausible-looking rows, and the
    mistake surfaces weeks later when someone compares incomparable numbers.
    """
    train = config.get("train") or {}
    ppo = train.get("ppo") or {}
    n_samples = int(train.get("rollout_steps", 0)) * int(train.get("episodes_per_update", 0))
    mb = int(ppo.get("minibatch_size", 0))
    floor = int((config.get("training_budget") or {}).get("min_nb", 1))
    nb = math.ceil(n_samples / mb) if (n_samples and mb) else 0
    if nb >= floor:
        return
    raise ValueError(
        f"训练预算不足：nb = ceil({train.get('rollout_steps')} * "
        f"{train.get('episodes_per_update')} / {ppo.get('minibatch_size')}) = {nb} < "
        f"training_budget.min_nb = {floor}。"
        "这样 PPO 每轮只更新约一次参数（实测 clip_frac=0、kl~1e-5），产出的只是 "
        "BC 起点，不是训练结果。调小 minibatch_size 或调大 episodes_per_update；"
        "训练超参请在 configs/experiment_base.yaml 改（见 实验产物规范.md §3.6）。"
    )


def _run(cmd: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(cmd) + "\n")
        stream.flush()
        completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"command exited {completed.returncode}; see {log}")


def train_one(args, model: str, seed: int) -> Path:
    name = getattr(args, "name", None) or f"s{seed}"
    run_dir = ROOT / "outputs" / "experiments" / model / name
    if run_dir.exists() and (run_dir / "checkpoint_final.pt").exists() and not args.resume:
        raise FileExistsError(f"Run already completed: {run_dir}; use --resume with a checkpoint")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = resolve_experiment(model, seed, args.updates, args.config, name, args.mode)
    config["runtime"]["device"] = args.device
    config_path = run_dir / "experiment_config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    threads = int(config["experiment"]["cpu_threads"])
    if threads < 1:
        raise ValueError("experiment.cpu_threads must be positive")
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")
    model_dir = ROOT / "qkd_rl" / "model_zoo" / model
    sources = {}
    for path in sorted(model_dir.glob("*.py")):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"model": model, "seed": seed, "started_at": datetime.now(timezone.utc).isoformat(),
                "config": str(config_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "cpu_threads": threads, "model_sources_sha256": sources,
                "status": "running"}
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checkpoint = args.resume
    try:
        if args.bc and not checkpoint:
            bc_dir = run_dir / "bc"
            cmd = [sys.executable, "scripts/train/supervised_train_pg_phased.py",
                   "--resolved-config", str(config_path), "--output-dir", str(bc_dir),
                   "--run-name", name, "--device", args.device,
                   "--batch-size", str(args.bc_batch_size),
                   "--continuous", "false", "--cover-all-days", "false"]
            bc_episodes = args.bc_episodes if args.bc_episodes is not None else config["experiment"].get("bc", {}).get("episodes")
            if bc_episodes is not None:
                cmd += ["--episodes", str(bc_episodes)]
            if args.bc_data:
                # Trajectories collected with collect_pg_phased_trajectories.py
                # --extra-config <model>/config.yaml carry the model's own
                # observation fields (v3: pair_path masks), so any model whose
                # collector output matches is fine. rebuild_obs backfills them.
                cmd += ["--data-dir", args.bc_data]
            _run(cmd, bc_dir / "process.log", env)
            checkpoint = str(bc_dir / "supervised_pg_phased.pt")
        if not args.bc_only:
            cmd = [sys.executable, "scripts/train/train_graph_mappo.py",
                   "--resolved-config", str(config_path), "--device", args.device]
            if checkpoint:
                cmd += ["--checkpoint", checkpoint]
            _run(cmd, run_dir / "train.log", env)
        manifest["status"] = "completed"
        manifest["checkpoint"] = str(run_dir / "checkpoint_final.pt") if not args.bc_only else checkpoint
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir


def evaluate_one(run_dir: Path, checkpoint: Path | None = None, seeds: list[int] | None = None,
                 overlays: list[str] | None = None, tag: str = "",
                 threads: int | None = None) -> dict:
    """在 run 自己的任务上评测检查点（trainer.evaluate_validation 同一实现）。

    seeds / overlays / tag（2026-09-27 新增）：换请求种子（留出测试集）或叠加
    配置（如 pair_seed1.yaml 换热门站对）。给了 tag 就写 ``eval_<tag>.json``，
    **不覆盖** validation.json；已存在则报错（outputs 只增不改）。
    """
    import torch
    from qkd_rl.env.factory import build_env_from_config
    from qkd_rl.model_zoo import build_model
    from qkd_rl.rl.algos.checkpoint import load_checkpoint
    from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer
    from qkd_rl.rl.algos.policy import MAPPOPolicy

    config = load_config([run_dir / "experiment_config.yaml"])
    for name in overlays or []:
        p = Path(name)
        config = deep_merge(config, load_config([p if p.exists() else ROOT / "configs" / name]))
    if seeds:
        config.setdefault("validation", {})["request_seeds"] = list(seeds)
        config["validation"]["episodes"] = len(seeds)
    if (seeds or overlays) and not tag:
        raise ValueError("换了种子或叠加了配置就必须给 --tag，避免覆盖 validation.json")
    path = run_dir / (f"eval_{tag}.json" if tag else "validation.json")
    if tag and path.exists():
        raise FileExistsError(f"{path} 已存在（outputs 只增不改）；换一个 tag")
    threads = int(threads or config.get("experiment", {}).get("cpu_threads", 4))
    torch.set_num_threads(threads)
    device = config.get("runtime", {}).get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    checkpoint = checkpoint or run_dir / "checkpoint_final.pt"
    env = build_env_from_config(config)
    model = build_model(env.action_resolver.action_space, config)
    model.load_state_dict(load_checkpoint(checkpoint, device).model_state)
    # validation.decode（改进 1a，opt-in）：greedy（默认，逐位不变）| max_weight
    decode = str((config.get("validation", {}) or {}).get("decode", "greedy"))
    decode_beta = float((config.get("validation", {}) or {}).get("decode_beta", 0.0) or 0.0)
    decode_persist = float((config.get("validation", {}) or {}).get("decode_persist_bonus", 0.0) or 0.0)
    policy = MAPPOPolicy(model, device, deterministic_decode=decode, decode_beta=decode_beta,
                         decode_persist_bonus=decode_persist)
    trainer = MAPPOTrainer(env, policy, config, run_dir, device=device)
    result = trainer.evaluate_validation(
        num_episodes=int(config.get("validation", {}).get("episodes", 1))
    )
    result.update({"checkpoint": str(checkpoint), "model": config["experiment"]["model"],
                   "overlays": list(overlays or []), "tag": tag, "decode": decode,
                   "decode_beta": decode_beta, "decode_persist_bonus": decode_persist})
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "batch"):
        p = sub.add_parser(command)
        if command == "train":
            p.add_argument("--model", default="v2")
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--name")
        else:
            p.add_argument("--models", nargs="+", required=True)
            p.add_argument("--seeds", nargs="+", type=int, required=True)
            p.add_argument("--parallel", type=int, default=1)
        p.add_argument("--updates", type=int)
        p.add_argument("--config", action="append", default=[])
        p.add_argument("--mode", default="random_episode")
        p.add_argument("--device", default="cpu")
        p.add_argument("--bc", action="store_true")
        p.add_argument("--bc-only", action="store_true")
        p.add_argument("--bc-episodes", type=int)
        p.add_argument("--bc-batch-size", type=int, default=64)
        p.add_argument("--bc-data")
        p.add_argument("--resume")
        p.add_argument("--evaluate", action="store_true")
    p = sub.add_parser("evaluate")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--seeds", default="", help="如 1000-1029；默认用 run 的验证种子")
    p.add_argument("--overlay", action="append", default=[], help="如 pair_seed1.yaml")
    p.add_argument("--tag", default="", help="输出 eval_<tag>.json（换种子/叠加时必填）")
    p.add_argument("--threads", type=int, default=0, help="0 = 用 run 的 cpu_threads")
    p = sub.add_parser("batch-evaluate")
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()
    if args.command == "train":
        if args.bc_only and not args.bc:
            parser.error("--bc-only requires --bc")
        run = train_one(args, args.model, args.seed)
        print(run)
        if args.evaluate and not args.bc_only:
            print(json.dumps(evaluate_one(run), ensure_ascii=False))
    elif args.command == "batch":
        if args.bc_only and not args.bc:
            parser.error("--bc-only requires --bc")
        jobs = [(model, seed) for model in args.models for seed in args.seeds]
        parallel = min(args.parallel, len(jobs))
        if parallel < 1:
            raise ValueError("--parallel must be positive")
        def job(model, seed):
            run = train_one(args, model, seed)
            return str(run), evaluate_one(run) if args.evaluate and not args.bc_only else None
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(job, m, s): (m, s) for m, s in jobs}
            for future in as_completed(futures):
                print(futures[future], future.result(), flush=True)
    elif args.command == "evaluate":
        seeds: list[int] = []
        for part in args.seeds.split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                seeds.extend(range(int(a), int(b) + 1))
            elif part.strip():
                seeds.append(int(part))
        print(json.dumps(evaluate_one(args.run, args.checkpoint, seeds, args.overlay,
                                      args.tag, args.threads or None), ensure_ascii=False))
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            for result in pool.map(evaluate_one, args.runs):
                print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
