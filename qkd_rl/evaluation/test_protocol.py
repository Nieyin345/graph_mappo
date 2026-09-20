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
    """Read the canonical validation settings from ``global.yaml``.

    ★★ 两个键位都认，且**读不到就抛错**。

    原先只读 ``raw["global"]["validation"]``，用一个 ``.get`` 链兜底到
    ``window_end_day=30`` / ``episode_days=1``。而 ``configs/train_full_rl.yaml``
    的 ``validation:`` 写在**顶层**（没有 ``global:`` 包裹）⟹ 拿它当参数调用时
    **不报错**，静默返回窗口 **0–30 天**——正好落在**训练窗口内**。
    实测踩到过一次：一个判读脚本因此在训练窗口上跑完了全程，还打印了
    「天 0–30」而没人注意，差一点把窗内读数当留出读数写进结论。

    静默的默认值在这里是最坏的失败模式：它不崩，只是让判读**测了另一个东西**。
    所以：两种键位都支持（``global.validation`` 优先，其次顶层 ``validation``），
    两者都没有就**抛 FileNotFoundError 式的 ValueError**，让调用方立刻知道。
    """
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ev = (raw.get("global", {}) or {}).get("validation")
    if ev is None:
        ev = raw.get("validation")
    if ev is None:
        raise ValueError(
            "%s 里找不到 validation 段（既不在 global.validation，也不在顶层 "
            "validation）。**不会用默认值兜底**：默认窗口是 0–30 天，会静默地"
            "把测量跑在训练窗口内。" % path)
    window = ev.get("window", {}) or {}
    episode = ev.get("episode", {}) or {}
    seeds = [int(s) for s in (ev.get("seeds", []) or ev.get("request_seeds", []) or [])]
    if "start_day" not in window or "end_day" not in window:
        raise ValueError("%s 的 validation.window 缺 start_day/end_day" % path)
    return {
        "window_start_day": int(window["start_day"]),
        "window_end_day": int(window["end_day"]),
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
