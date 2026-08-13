"""Train Graph-MAPPO on the QKD scheduling environment.

Usage (from the project root):

    conda run -n pytorch python scripts/train_graph_mappo.py
    conda run -n pytorch python scripts/train_graph_mappo.py --num-updates 100 --run-name exp1
    conda run -n pytorch python scripts/train_graph_mappo.py --seed 7
    conda run -n pytorch python scripts/train_graph_mappo.py --checkpoint outputs/exp1/checkpoint_update_0100.pt
"""

from __future__ import annotations

import os

# Conda/matplotlib on Windows loads libiomp5md.dll twice; this documented
# workaround prevents the OMP Error #15 crash at import/exit.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.algos.mappo_trainer import MAPPOTrainer
from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.evaluation import plot_learning_curve, read_train_history
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Graph-MAPPO on the QKD scheduling environment.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Extra config files under configs/ merged in order, e.g. env_full.yaml train_mappo.yaml",
    )
    parser.add_argument(
        "--mode",
        default="random_episode",
        choices=["random_episode", "continuous", "fixed_day", "curriculum", "demand_edge"],
        help="Training profile in configs/train_profiles.yaml.",
    )
    parser.add_argument("--run-name", default=None, help="Override project.run_name (output sub-folder).")
    parser.add_argument("--num-updates", type=int, default=None, help="Override train.num_updates.")
    parser.add_argument("--seed", type=int, default=None, help="Override global/env seed.")
    parser.add_argument("--checkpoint", default=None, help="Resume from a checkpoint .pt file.")
    parser.add_argument("--device", default=None, help="Override runtime.device (cpu/cuda).")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    config = load_default_config(ROOT)
    # Training reads the H5 dataset only: default to the full-scale scenario.
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    profiles = load_config([ROOT / "configs" / "train_profiles.yaml"]).get("train_profiles", {})
    if args.mode not in profiles:
        raise SystemExit(f"Unknown training mode: {args.mode}. Available: {sorted(profiles)}")
    config = deep_merge(config, profiles[args.mode])
    global_cfg = load_config([ROOT / "configs" / "global.yaml"]).get("global", {})
    train_global = global_cfg.get("training", {}) or {}
    train_window = train_global.get("window", {}) or {}
    if train_window.get("start_day") is not None and train_window.get("end_day") is not None:
        train_start_day = int(train_window["start_day"])
        train_end_day = int(train_window["end_day"])
        config["env"]["activation_window_start_day"] = train_start_day
        config["env"]["activation_window_end_day"] = train_end_day
        config["env"]["activation_window_days"] = max(0, train_end_day - train_start_day)
        day_steps = int(config["env"].get("day_steps", 1440))
        if config["env"].get("continuous"):
            session_days = int(config["train"].get("continuous_session_days", 30) or 30)
            extension_days = session_days
        else:
            rollout_steps = int(config["train"].get("rollout_steps", day_steps) or day_steps)
            extension_days = max(1, (rollout_steps + day_steps - 1) // day_steps)
        config["scenario"]["time_limit"]["days"] = train_end_day + extension_days
    if "request_seed" in train_global:
        config["seed"]["env_seed"] = int(train_global["request_seed"])
    validation_global = global_cfg.get("validation", {}) or {}
    if validation_global:
        config["validation"] = validation_global
    if args.configs:
        for name in args.configs:
            config = deep_merge(config, load_config([ROOT / "configs" / name]))
    if args.seed is not None:
        config["seed"]["global_seed"] = args.seed
        config["seed"]["env_seed"] = args.seed
    if args.num_updates is not None:
        config["train"]["num_updates"] = args.num_updates
    if args.run_name is not None:
        config["project"]["run_name"] = args.run_name
    if args.device is not None:
        config["runtime"]["device"] = args.device
    ConfigValidator().validate(config)
    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)

    seed = int(config["seed"]["global_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Auto-select CUDA when available; --device cpu still forces CPU. Only the
    # compute backend changes, the training architecture/algorithm is untouched.
    if args.device is None and torch.cuda.is_available():
        config["runtime"]["device"] = "cuda"
        print("runtime: CUDA detected, using device=cuda (use --device cpu to force CPU)")
    # TF32 fast matmul on Ampere+ GPUs (RTX 30/40 series), no-op on CPU/older.
    torch.set_float32_matmul_precision("high")
    if config["runtime"].get("deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    device = config["runtime"]["device"]
    print(
        f"runtime: device={device} float32_matmul_precision=high "
        f"cudnn.benchmark={torch.backends.cudnn.benchmark}"
    )
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, device)

    output_dir = ROOT / config["project"]["output_dir"] / config["project"]["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"Output dir: {output_dir}")

    trainer = MAPPOTrainer(env, policy, config, output_dir, device=device)
    if args.checkpoint:
        trainer.load_checkpoint(Path(args.checkpoint))
        print(f"Resumed from {args.checkpoint} at update {trainer.update_count}")

    trainer.train()
    final_path = output_dir / "checkpoint_final.pt"
    trainer.save_checkpoint(final_path, trainer.last_stats)
    print(f"Saved final checkpoint: {final_path}")
    if trainer.last_stats is not None:
        print(f"Final: {trainer.last_stats}")

    history_path = output_dir / "metrics.jsonl"
    if history_path.exists():
        history = read_train_history(history_path)
        if history:
            figures = plot_learning_curve(
                history,
                output_dir / "figures" / f"{config['project']['run_name']}_learning_curve",
                title=f"{config['project']['run_name']} (seed {seed})",
            )
            for figure in figures:
                print(f"figure           : {figure}")


if __name__ == "__main__":
    main()

