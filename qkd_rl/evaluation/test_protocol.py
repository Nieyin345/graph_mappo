"""Shared validation protocol for every baseline and RL checkpoint.

All testers read the same ``global.validation`` block from
``configs/global.yaml`` and build the same environment config, so RL,
greedy baselines and the receding-horizon MILP upper bound are compared under
identical conditions: window, request seeds, episode length, random-start mode
and scenario time limit.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from qkd_rl.core.config import deep_merge, load_config
from qkd_rl.env.factory import load_default_config

DAY_STEPS = 1440


def load_validation_profile(config_path: str | Path = "configs/global.yaml") -> dict:
    """Read the canonical validation settings from ``global.yaml``."""
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ev = raw.get("global", {}).get("validation", {}) or {}
    window = ev.get("window", {}) or {}
    episode = ev.get("episode", {}) or {}
    seeds = [int(s) for s in (ev.get("seeds", []) or ev.get("request_seeds", []) or [])]
    return {
        "window_start_day": int(window.get("start_day", 0)),
        "window_end_day": int(window.get("end_day", 30)),
        "episode_days": int(episode.get("days", ev.get("episode_days", 1))),
        "episode_steps": int(episode.get("steps", ev.get("episode_steps", 0)) or 0),
        "episodes": int(ev.get("episodes", 3)),
        "seed_start": int(ev.get("seed_start", 7)),
        "seeds": seeds,
        "start_seed": int(ev.get("start_seed", 0)),
        "start_mode": str(ev.get("start_mode", "random_day")),
    }


def build_validation_env_config(
    profile: dict,
    include_baselines: bool = False,
    episode_days: int | None = None,
    episode_steps: int | None = None,
    start_mode: str | None = None,
) -> dict:
    """Build one identical env config for validation runs.

    ``include_baselines=True`` also merges ``configs/baselines.yaml`` (needed
    by baseline policies; harmless for the environment itself).
    """
    root = Path(__file__).resolve().parents[2]
    start_day = int(profile["window_start_day"])
    end_day = int(profile["window_end_day"])
    window_days = max(0, end_day - start_day)
    if episode_days is not None:
        episode_steps = episode_days * DAY_STEPS
    if episode_steps is None:
        episode_steps = int(profile["episode_steps"]) or int(profile["episode_days"]) * DAY_STEPS
    mode = start_mode or profile["start_mode"]

    config = load_default_config(root)
    config = deep_merge(config, load_config([root / "configs" / "env_full.yaml"]))
    if include_baselines:
        config = deep_merge(config, load_config([root / "configs" / "baselines.yaml"]))
    config["rate_provider"]["provider"] = "h5"
    config["env"]["episode_start_mode"] = mode
    config["env"]["episode_steps"] = episode_steps
    if mode == "random_day":
        config["env"]["activation_window_start_day"] = start_day
        config["env"]["activation_window_end_day"] = end_day
        config["env"]["activation_window_days"] = window_days
        config["scenario"]["time_limit"]["days"] = end_day + max(
            1, math.ceil(episode_steps / DAY_STEPS)
        )
    else:
        config["scenario"]["time_limit"]["days"] = end_day
    return config


def resolve_seeds(profile: dict, episodes: int | None = None) -> list[int]:
    """Return the canonical request seeds for the validation protocol."""
    episodes = int(episodes or profile["episodes"])
    if profile["seeds"]:
        return list(profile["seeds"])
    return list(range(int(profile["seed_start"]), int(profile["seed_start"]) + episodes))
