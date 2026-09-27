"""Shared validation protocol for every baseline and RL checkpoint.

All testers read the same ``global.validation`` block from
``configs/global.yaml`` and build the same environment config, so RL,
greedy baselines and the receding-horizon MILP upper bound are compared under
identical conditions: window, request seeds, episode length, random-start mode
and scenario time limit.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
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


def checkpoint_validation_config(
    checkpoint_config: dict | None, validation_config: dict
) -> dict:
    """Use checkpoint model/observation settings on one canonical validation env.

    Validation must keep physical dynamics identical across policies. Only the
    sections that define the policy input/output contract are copied from the
    checkpoint; request generation, QKP dynamics, routing, scenario, reward and
    dataset/window settings stay canonical. Older checkpoints without a saved
    config simply use the validation config unchanged.
    """
    config = deepcopy(validation_config)
    if checkpoint_config:
        for section in ("features", "model", "action_resolver"):
            if section in checkpoint_config:
                config[section] = deep_merge(
                    config.get(section, {}), checkpoint_config[section]
                )
    # Validation episodes are always finite, even for continuous-training runs.
    config.setdefault("env", {})["continuous"] = False
    return config


def resolve_seeds(profile: dict, episodes: int | None = None) -> list[int]:
    """Return the canonical request seeds for the validation protocol."""
    episodes = int(episodes or profile["episodes"])
    if profile["seeds"]:
        return list(profile["seeds"])
    return list(range(int(profile["seed_start"]), int(profile["seed_start"]) + episodes))


# ---------------------------------------------------------------------------
# 「按训练任务评测」路径（2026-09-27 审计后新增）
#
# 实测问题：trainer 的验证继承训练 config（pairhot_v2 + arrival_rate），而
# eval_expert 从 default + env_full 重新拼（legacy_v1）⟹ 专家与 RL 在**两个
# 不同任务**上打分，1× 下专家读数差 0.013（0.7674 vs 0.7803），RL−专家差被
# 高估约 0.013。下面两个函数是**唯一实现**：trainer 验证、eval_expert --run、
# experiment.py evaluate 都走它；task_fingerprint 写进每份结果，配对脚本
# 指纹不同就拒绝配对。
# ---------------------------------------------------------------------------

# 决定「任务」的配置段：请求流、密钥池、路由、速率数据、场景、环境动力学。
# 不含 model/features/reward/ppo 等策略侧段（它们不改变被测的任务）。
TASK_SECTIONS = ("requests", "qkp", "routing", "rate_provider", "scenario", "env")


def run_validation_seeds(validation_cfg: dict, num_episodes: int | None = None) -> list[int]:
    """训练 config 的 ``validation`` 段 → 请求种子列表（与 trainer 旧逻辑逐字等价）。"""
    seeds = [int(s) for s in (validation_cfg.get("request_seeds", []) or [])]
    if not seeds:
        seeds = list(range(7, 7 + int(validation_cfg.get("episodes", 3) or 3)))
    if num_episodes is not None and num_episodes > 0:
        seeds = [seeds[i % len(seeds)] for i in range(int(num_episodes))]
    return seeds


def run_validation_env_config(
    config: dict, validation_cfg: dict | None = None, step_limit: int | None = None
) -> dict:
    """训练 config → 验证环境 config：任务段原样继承，只换验证窗口/回合长度。

    与 trainer 旧 ``_build_validation_envs`` 逐字等价（由测试钉住）。
    """
    validation_cfg = validation_cfg if validation_cfg is not None else (
        config.get("validation", {}) or {})
    window = validation_cfg.get("window", {}) or {}
    start_day = int(window.get("start_day", 0))
    end_day = int(window.get("end_day", 365))
    episode_steps = int(step_limit) if (step_limit is not None and step_limit > 0) else 0
    day_steps = int(config["env"].get("day_steps", DAY_STEPS))
    if episode_steps <= 0:
        episode_steps = int(validation_cfg.get("episode_steps", 0) or 0)
    if episode_steps <= 0:
        episode_steps = int(validation_cfg.get("episode_days", 1) or 1) * day_steps

    out = deepcopy(config)
    out["env"]["episode_start_mode"] = str(validation_cfg.get("start_mode", "random_day"))
    out["env"]["episode_steps"] = episode_steps
    out["env"]["continuous"] = False
    out["env"]["activation_window_start_day"] = start_day
    out["env"]["activation_window_end_day"] = end_day
    out["env"]["activation_window_days"] = max(0, end_day - start_day)
    out["scenario"]["time_limit"]["days"] = end_day + max(1, math.ceil(episode_steps / day_steps))
    return out


def task_fingerprint(env_config: dict) -> str:
    """任务指纹：TASK_SECTIONS 的规范化 JSON 的 sha1 前 12 位。

    两份结果指纹不同 = 不是同一个任务，禁止配对。``seed.env_seed`` 不计入
    （它只是第一条回合的种子，回合种子另有 ``seeds`` 字段）。
    """
    payload = {k: env_config.get(k) for k in TASK_SECTIONS}
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
