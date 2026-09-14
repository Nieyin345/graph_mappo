"""Supervised warm-start training from the PG-Phased heuristic (behavior cloning).

The expert is ``PathScoreGreedy`` with ``phased=True`` plus the ``ServeProbe``
router (the finalized three-stage heuristic: existing-key network -> mixed
stock-assisted paths -> fresh paths). Unlike the old regression scripts that
mimic heuristic *scores*, this script clones the expert's **decisions**:

- per step the expert returns per-node ``(tx_target, rx_source)`` actions;
- the environment resolves them into the **actually executed** directed arc
  matching (``env.last_matched_arcs``), which is the exact matching space the
  policy samples (dual-port rules: Tx-out<=1, Rx-in<=1, pair 对端不同);
- the model's arc scores are evaluated with ``_matching_log_prob_entropy_fast``,
  i.e. the log probability of the expert matching plus the STOP option,
  **averaged over the matching's decisions** (a per-decision mean NLL, the
  standard length-normalized sequence objective -- not the sum over arcs);
- the loss is ``-mean_log_prob``: the actor learns to rank the arcs that form
  the expert's serviceable paths at the top, and to pick STOP when the expert
  activates nothing.

By default the expert is re-run during training (collect + train per day).
Pass ``--data-dir`` to train from trajectories pre-collected by
``collect_pg_phased_trajectories.py`` instead, so the heuristic is never
re-run and repeated experiments reuse the same expert data.

The resulting checkpoint is saved under ``outputs/<run_name>/`` and can warm
start MAPPO (``--checkpoint`` in ``train_graph_mappo.py``).

Usage:
    python scripts/rl/supervised_train_pg_phased.py --run-name supervised_pg_phased
    python scripts/rl/supervised_train_pg_phased.py --data-dir outputs/trajs_pg_phased
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qkd_rl.rl.algos.checkpoint import load_checkpoint, save_checkpoint
from qkd_rl.baselines.path_greedy import PathScoreGreedy
from qkd_rl.baselines.serve_probe import ServeProbe
from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.env.graph_builder import GraphObservation
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic

# numpy 2.x pickle 引用 ``numpy._core`` 命名空间；本环境若装的是 numpy 1.x
# （如 CUDA torch 环境自带 numpy 1.24），该命名空间不存在会导致反序列化失败。
# 将 ``numpy.core`` 别名到 ``numpy._core``，保证预收集轨迹在任意环境都能加载。
try:
    import numpy._core  # noqa: F401
except ModuleNotFoundError:  # numpy < 2.0
    import sys as _sys
    import numpy.core as _np_core
    _sys.modules.setdefault("numpy._core", _np_core)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/supervised_train.yaml")
    parser.add_argument("--run-name", type=str, default="supervised_pg_phased")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cover-all-days", type=str, choices=("true", "false"), default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume supervised training from this checkpoint.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Train from pre-collected PG-Phased trajectories "
        "(collect_pg_phased_trajectories.py output) instead of re-running the heuristic.",
    )
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


def rebuild_obs(rec: dict) -> GraphObservation:
    """Rebuild a GraphObservation from a minimal stored record.

    The model forward only reads the feature/indices/mask fields; ``state`` is
    not needed (it was dropped by the collector) and the history fields stay at
    their empty defaults (history encoder is disabled in this preset).
    """
    return GraphObservation(
        node_features=rec["node_features"],
        edge_index=rec["edge_index"],
        edge_features=rec["edge_features"],
        node_ids=rec["node_ids"],
        edge_ids=rec["edge_ids"],
        physical_edge_ids=rec["physical_edge_ids"],
        demand_edge_ids=rec["demand_edge_ids"],
        action_candidates=rec["action_candidates"],
        action_masks=rec["action_masks"],
        state=None,
        raw_action_masks=rec.get("raw_action_masks"),
        flat_action_masks=rec.get("flat_action_masks"),
    )


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
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config).to(device)
    # Behavior cloning evaluates the expert matching deterministically.
    model.actor.temperature = 1.0
    from qkd_rl.rl.algos.policy import MAPPOPolicy

    policy = MAPPOPolicy(model, device=device)

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
    total_bc_loss = 0.0
    total_steps = 0
    early_stopped = False
    day_steps = profile["episode_steps"] or profile["episode_days"] * 1440

    def snapshot(episode: int) -> None:
        snap = output_dir / "supervised_pg_phased_latest.pt"
        save_checkpoint(
            snap,
            update=0,
            model=model,
            optimizer=optimizer,
            config=config,
            metrics={
                "pretrain_loss": total_bc_loss / max(1, total_steps),
                "steps": total_steps,
                "episode": episode,
            },
        )
        log(f"periodic checkpoint: {snap} (episode={episode})")

    def train_batch(obs_arcs: list[tuple[GraphObservation, list]]) -> tuple[float, int]:
        """One gradient step on a batch of (obs, expert_arcs)."""
        nonlocal total_bc_loss, total_steps
        outputs = model.batched_forward([item[0] for item in obs_arcs], device)
        losses = []
        for (_obs, arcs), edge_map in zip(obs_arcs, outputs.edge_score_maps):
            if not edge_map:
                continue
            # Keep only arcs the model actually scored; empty expert
            # matchings stay as-is so the STOP probability is trained.
            arcs = [arc for arc in arcs if arc in edge_map]
            mean_lp, _mean_entropy = policy._matching_log_prob_entropy_fast(edge_map, arcs)
            losses.append(-mean_lp)
        if not losses:
            return 0.0, 0
        loss = torch.stack(losses).mean()
        batch_loss = float(loss.detach().cpu())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_bc_loss += batch_loss * len(obs_arcs)
        total_steps += len(obs_arcs)
        return batch_loss, len(obs_arcs)

    def train_episode(expert_data: list[tuple[GraphObservation, list]]) -> tuple[float, int]:
        """Train over one day of expert data (obs, arcs), batch by batch."""
        episode_bc_loss = 0.0
        episode_samples = 0
        for start in range(0, len(expert_data), args.batch_size):
            batch = expert_data[start : start + args.batch_size]
            batch_loss, batch_samples = train_batch(batch)
            if batch_samples == 0:
                continue
            episode_bc_loss += batch_loss * batch_samples
            episode_samples += batch_samples
            log(
                f"  batch {episode_samples // batch_samples}: bc_loss={batch_loss:.5f} "
                f"running_avg={total_bc_loss / max(1, total_steps):.5f}"
            )
        return episode_bc_loss, episode_samples

    if args.data_dir:
        files = sorted(Path(args.data_dir).glob("day_*.pkl.gz"))
        if not files:
            raise SystemExit(f"no day_*.pkl.gz under {args.data_dir}")
        log(f"[data] training from {len(files)} day files in {args.data_dir}")
        for episode, path in enumerate(files, start=1):
            with gzip.open(path, "rb") as fh:
                records = pickle.load(fh)
            expert_data = [(rebuild_obs(rec), list(rec["arcs"])) for rec in records]
            episode_bc_loss, episode_samples = train_episode(expert_data)
            avg_episode_loss = episode_bc_loss / max(1, episode_samples)
            log(
                f"episode={episode}/{len(files)} file={path.name} "
                f"expert_steps={len(expert_data)} bc_loss={total_bc_loss / max(1, total_steps):.5f} "
                f"episode_avg={avg_episode_loss:.5f}"
            )
            if profile["min_loss"] > 0.0 and avg_episode_loss < profile["min_loss"]:
                log(f"early stop: episode avg loss {avg_episode_loss:.5f} < min_loss {profile['min_loss']}")
                early_stopped = True
                break
            if episode % 10 == 0:
                snapshot(episode)
    else:
        expert = PathScoreGreedy(
            weights=(1.0, 10.0, 1.0, 0.5, 0.2),
            phased=True,
            principles=False,
            router=ServeProbe(env),
        )
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
                obs_next, _reward, terminated, truncated, _info = env.step(actions)
                # Behavior cloning targets the arcs the environment ACTUALLY
                # executed (the resolver may reject part of the expert's raw
                # proposals on dual-port conflicts), which is exactly the
                # matching space the policy samples.
                expert_data.append((obs, list(env.last_matched_arcs)))
                obs = obs_next
                if terminated or truncated:
                    if profile["continuous"] and env.t < env.scenario.end_t:
                        env.steps = 0
                        continue
                    break
            episode_bc_loss, episode_samples = train_episode(expert_data)
            avg_episode_loss = episode_bc_loss / max(1, episode_samples)
            log(
                f"episode={episode}/{len(profile['seeds'])} seed={seed} "
                f"expert_steps={len(expert_data)} bc_loss={total_bc_loss / max(1, total_steps):.5f} "
                f"episode_avg={avg_episode_loss:.5f}"
            )
            if profile["min_loss"] > 0.0 and avg_episode_loss < profile["min_loss"]:
                log(f"early stop: episode avg loss {avg_episode_loss:.5f} < min_loss {profile['min_loss']}")
                early_stopped = True
                break
            if profile["continuous"]:
                env.steps = 0
            if episode % 10 == 0:
                snapshot(episode)

    checkpoint_path = output_dir / "supervised_pg_phased.pt"
    save_checkpoint(
        checkpoint_path,
        update=0,
        model=model,
        optimizer=optimizer,
        config=config,
        metrics={
            "pretrain_loss": total_bc_loss / max(1, total_steps),
            "steps": total_steps,
        },
    )
    with (output_dir / "supervised_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "expert": "PathScoreGreedy(phased=True)+ServeProbe",
                "supervision": "expert_matching_joint_log_likelihood",
                "data_dir": args.data_dir,
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
