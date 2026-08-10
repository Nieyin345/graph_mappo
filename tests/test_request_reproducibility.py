"""Request generation must be reproducible per episode seed.

``env.reset(seed=...)`` must reseed the request generator too, otherwise
workers that reuse one env across episodes (and reruns of the same seed) see
a request stream that continues from the previous episode instead of
restarting.
"""
from __future__ import annotations

from qkd_rl.env.factory import build_env_from_config
from tests.helpers import build_test_env, point_config_to_h5, load_default_config


def _request_fingerprint(env, steps: int, seed: int) -> list[tuple]:
    env.reset(seed=seed)
    out: list[tuple] = []
    for _ in range(steps):
        for req in env.request_generator.generate(env.t):
            out.append((req.src_gs, req.dst_gs, round(req.amount, 6), req.deadline_t, req.priority))
        env.t += 1
    return out


def test_same_seed_reproduces_stream_across_envs():
    env_a = build_test_env()
    env_b = build_test_env()
    assert _request_fingerprint(env_a, 40, seed=7) == _request_fingerprint(env_b, 40, seed=7)


def test_same_env_reset_twice_reproduces_stream():
    # Regression: the env is reused across episodes in workers; the second
    # episode must restart from the seed instead of continuing the stream.
    env = build_test_env()
    first = _request_fingerprint(env, 40, seed=7)
    second = _request_fingerprint(env, 40, seed=7)
    assert first == second


def test_different_seeds_produce_different_streams():
    env = build_test_env()
    assert _request_fingerprint(env, 40, seed=7) != _request_fingerprint(env, 40, seed=8)


def test_same_seed_reproduces_episode_end_to_end():
    config = point_config_to_h5(load_default_config("."))
    env_a = build_env_from_config(config)
    env_b = build_env_from_config(config)
    rewards_a = _rollout(env_a, seed=3, steps=30)
    rewards_b = _rollout(env_b, seed=3, steps=30)
    assert rewards_a == rewards_b
    assert env_a.metrics.episode_summary() == env_b.metrics.episode_summary()


def _rollout(env, seed: int, steps: int) -> list[float]:
    env.reset(seed=seed)
    rewards: list[float] = []
    idle = {node_id: "idle" for node_id in env.action_resolver.action_space.node_ids}
    for _ in range(steps):
        _obs, reward, _term, _trunc, _info = env.step(idle)
        rewards.append(round(float(reward), 9))
    return rewards