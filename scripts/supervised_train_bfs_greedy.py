"""Supervised warm-start training from the BFS + greedy V2 baseline.

The expert is ``GreedyRelayDiffusionPolicyV2``. The script collects expert
trajectories over the activation window defined in ``configs/supervised_train.yaml``
and trains the actor to regress the heuristic score of **every** legal
physical edge, not only the edges selected by the greedy matching. The
resulting checkpoint is saved under ``outputs/<run_name>/``.

Usage:
    python scripts/supervised_train_bfs_greedy.py --run-name supervised_bfs
    python scripts/supervised_train_bfs_greedy.py --checkpoint outputs/old/supervised_bfs_greedy.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.algos.checkpoint import load_checkpoint, save_checkpoint
from qkd_rl.baselines.greedy_relay_diffusion import GreedyRelayDiffusionPolicyV3
from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/supervised_train.yaml")
    parser.add_argument("--run-name", type=str, default="supervised_bfs_greedy")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cover-all-days", type=str, choices=("true", "false"), default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume supervised training from this checkpoint.")
    return parser.parse_args()


def load_profile(config_path: str) -> dict:
    raw = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8")) or {}
    ev = raw.get("evaluation", {}) or {}
    global_raw = yaml.safe_load((ROOT / "configs" / "global.yaml").read_text(encoding="utf-8")) or {}
    train_global = global_raw.get("global", {}).get("training", {}) or {}
    window = train_global.get("window", {}) or {}
    start_day = int(window.get("start_day", 0))
    end_day = int(window.get("end_day", 0))
    seeds = [int(s) for s in (ev.get("seeds", []) or [])]
    if not seeds and "request_seed" in train_global:
        seeds = [int(train_global["request_seed"])]
    episodes = int(ev.get("episodes", 0) or 0) or max(0, end_day - start_day)
    return {
        "window_start_day": start_day,
        "window_end_day": end_day,
        "episode_days": int(ev.get("episode_days", 1)),
        "episode_steps": int(ev.get("episode_steps", 0) or 0),
        "episodes": episodes,
        "seeds": seeds,
        "cover_all_days": bool(ev.get("cover_all_days", False)),
        "min_loss": float(ev.get("min_loss", 0.0) or 0.0),
        "continuous": bool(ev.get("continuous", False)),
    }


def build_config(profile: dict) -> dict:
    day_steps = 1440
    episode_steps = profile["episode_steps"] or profile["episode_days"] * day_steps
    start_day = profile["window_start_day"]
    end_day = profile["window_end_day"]
    if end_day <= start_day:
        raise ValueError(
            f"supervised_train.yaml must define window.end_day > window.start_day, got {start_day} -> {end_day}"
        )

    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config = deep_merge(config, load_config([ROOT / "configs" / "baselines.yaml"]))
    config["rate_provider"]["provider"] = "h5"
    config["env"]["episode_start_mode"] = "random_day"
    config["env"]["episode_steps"] = (
        int(10**9) if profile.get("continuous", False) else episode_steps
    )
    if profile.get("continuous", False):
        config["env"]["continuous"] = True
    config["env"]["activation_window_start_day"] = start_day
    config["env"]["activation_window_days"] = end_day - start_day
    config["scenario"]["time_limit"]["days"] = end_day + max(1, math.ceil(episode_steps / day_steps))
    config["project"]["output_dir"] = "outputs"
    ConfigValidator().validate(config)
    return config


def main() -> None:
    args = parse_args()
    profile = load_profile(args.config)
    if args.episodes is not None:
        profile["episodes"] = args.episodes
        profile["seeds"] = profile["seeds"][: args.episodes]
    if args.cover_all_days is not None:
        profile["cover_all_days"] = args.cover_all_days == "true"
    if profile["cover_all_days"]:
        profile["seeds"] = list(range(profile["window_start_day"], profile["window_end_day"]))
    if profile["continuous"]:
        profile["cover_all_days"] = True
        profile["seeds"] = list(range(profile["window_start_day"], profile["window_end_day"]))
    config = build_config(profile)
    device = torch.device(args.device)
    torch.manual_seed(profile["seeds"][0])

    env = build_env_from_config(config)
    if profile["continuous"]:
        env.config["env"]["episode_steps"] = int(10**9)
        if not getattr(env, "continuous", False):
            raise RuntimeError(
                "supervised continuous mode requires the updated QKDEnv code. "
                "Restart the Python process after reloading the repository."
            )
        print(
            f"[supervised] continuous window {profile['window_start_day']}->"
            f"{profile['window_end_day']}, scenario_end_t={env.scenario.end_t}"
        )
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config).to(device)

    base_cfg = load_config([ROOT / "configs" / "baselines.yaml"]).get("baselines", {})
    grd_cfg = base_cfg.get("greedy_relay_diffusion_v3", {})
    expert = GreedyRelayDiffusionPolicyV3(
        rate_weight=grd_cfg.get("rate_weight", 1.0),
        importance_weight=grd_cfg.get("importance_weight", 10.0),
        completion_weight=grd_cfg.get("completion_weight", 1.0),
        keep_weight=grd_cfg.get("keep_weight", 0.5),
        switch_weight=grd_cfg.get("switch_weight", 0.2),
        hop_decay_factor=grd_cfg.get("hop_decay_factor", 0.25),
        max_path_links=grd_cfg.get("max_path_links", 3),
        wait_urgency_tau_ratio=grd_cfg.get("wait_urgency_tau_ratio", 0.8),
        ignore_consumption=grd_cfg.get("ignore_consumption", False),
        include_stocked_unavailable=grd_cfg.get("include_stocked_unavailable", True),
    )

    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": args.lr},
            {"params": model.actor.parameters(), "lr": args.lr},
            {"params": model.critic.parameters(), "lr": args.lr},
        ]
    )
    if args.checkpoint:
        data = load_checkpoint(args.checkpoint, device)
        model.load_state_dict(data.model_state)
        if data.optimizer_state is not None:
            try:
                optimizer.load_state_dict(data.optimizer_state)
            except ValueError as exc:
                print(f"optimizer state incompatible ({exc}); starting optimizer fresh")
        print(f"resumed supervised checkpoint: {args.checkpoint} (update={data.update})")
    output_dir = ROOT / "outputs" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "supervised_train.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")

    model.train()
    total_weighted_loss = 0.0
    total_steps = 0
    early_stopped = False
    day_steps = profile["episode_steps"] or profile["episode_days"] * 1440
    obs = None
    for episode, seed in enumerate(profile["seeds"], start=1):
        if profile["continuous"]:
            if obs is None:
                config["env"]["episode_start_day"] = seed
                obs = env.reset(seed=seed)
        else:
            if profile["cover_all_days"]:
                config["env"]["episode_start_day"] = seed
            else:
                config["env"].pop("episode_start_day", None)
            obs = env.reset(seed=seed)
        expert_data = []
        for _ in range(day_steps):
            actions, _scores = expert.act(obs)
            target_scores = expert.score_edges(obs)
            expert_data.append((obs, target_scores))
            obs, _reward, terminated, truncated, _info = env.step(actions)
            if terminated or truncated:
                if profile["continuous"] and env.t < env.scenario.end_t:
                    # Defensive reset of the step counter: even if the loaded
                    # environment code still uses per-day termination, keep
                    # the continuous trajectory moving instead of stopping.
                    env.steps = 0
                    continue
                break

        episode_weighted_loss = 0.0
        episode_samples = 0
        for start in range(0, len(expert_data), args.batch_size):
            batch = expert_data[start : start + args.batch_size]
            outputs = model.batched_forward([item[0] for item in batch], device)
            losses = []
            for (obs, target), edge_map in zip(batch, outputs.edge_score_maps):
                edge_ids = sorted(target.keys())
                if not edge_ids:
                    continue
                pred = torch.stack([edge_map[edge_id] for edge_id in edge_ids])
                target_t = torch.tensor(
                    [target[edge_id] for edge_id in edge_ids],
                    dtype=torch.float32,
                    device=device,
                )
                if target_t.numel() > 1 and target_t.std() > 1.0e-6:
                    target_t = (target_t - target_t.mean()) / target_t.std()
                losses.append(F.mse_loss(pred, target_t))
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            batch_loss = float(loss.detach().cpu())
            batch_samples = len(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_weighted_loss += batch_loss * batch_samples
            total_steps += batch_samples
            episode_weighted_loss += batch_loss * batch_samples
            episode_samples += batch_samples
            running_avg = total_weighted_loss / max(1, total_steps)
            log(
                f"  batch {episode_samples // batch_samples}: loss={batch_loss:.5f} "
                f"running_avg={running_avg:.5f}"
            )
        avg_episode_loss = episode_weighted_loss / max(1, episode_samples)
        log(
            f"episode={episode}/{len(profile['seeds'])} seed={seed} "
            f"expert_steps={len(expert_data)} loss={total_weighted_loss / max(1, total_steps):.5f} "
            f"episode_avg={avg_episode_loss:.5f}"
        )
        if profile["min_loss"] > 0.0 and avg_episode_loss < profile["min_loss"]:
            log(f"early stop: episode avg loss {avg_episode_loss:.5f} < min_loss {profile['min_loss']}")
            early_stopped = True
            break
        if profile["continuous"]:
            # Keep QKP/requests/history, but reset the per-day step counter so
            # the next day can run another full 1440 steps without terminating.
            env.steps = 0

    checkpoint_path = output_dir / "supervised_bfs_greedy.pt"
    save_checkpoint(
        checkpoint_path,
        update=0,
        model=model,
        optimizer=optimizer,
        config=config,
        metrics={
            "pretrain_loss": total_weighted_loss / max(1, total_steps),
            "steps": total_steps,
        },
    )
    with (output_dir / "supervised_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "expert": "GreedyRelayDiffusionPolicyV2",
                "supervision": "all_edge_score_regression",
                "window": {
                    "start_day": profile["window_start_day"],
                    "end_day": profile["window_end_day"],
                },
                "episode_steps": day_steps,
                "seeds": profile["seeds"],
                "cover_all_days": profile["cover_all_days"],
                "continuous": profile["continuous"],
                "min_loss": profile["min_loss"],
                "early_stopped": early_stopped,
                "checkpoint": str(checkpoint_path),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    log(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
