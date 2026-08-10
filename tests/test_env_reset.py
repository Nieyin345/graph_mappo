"""Test env.reset() episode start modes (fixed / random_window / random_day)."""
from __future__ import annotations

from qkd_rl.env.factory import build_env_from_config, load_default_config
from tests.helpers import point_config_to_h5


def _env_with_start_mode(mode: str):
    config = point_config_to_h5(load_default_config("."))
    config["env"]["episode_start_mode"] = mode
    config["env"]["episode_steps"] = 400
    config["env"]["day_steps"] = 1440
    return build_env_from_config(config)


def test_fixed_start_always_uses_scenario_start():
    env = _env_with_start_mode("fixed")
    env.reset(seed=1)
    assert env.t == env.scenario.start_t


def test_random_window_stays_in_bounds():
    env = _env_with_start_mode("random_window")
    start = env.scenario.start_t
    end = env.scenario.end_t
    seen = set()
    for seed in range(10):
        env.reset(seed=seed)
        assert start <= env.t <= end - 400
        seen.add(env.t)
    assert len(seen) > 1  # seeds actually produce different starts


def test_random_day_is_day_aligned_and_in_bounds():
    env = _env_with_start_mode("random_day")
    start = env.scenario.start_t
    end = env.scenario.end_t
    seen = set()
    for seed in range(10):
        env.reset(seed=seed)
        assert start <= env.t <= end - 400
        assert (env.t - start) % 1440 == 0  # aligned to day boundaries
        seen.add(env.t)
    assert len(seen) > 1
