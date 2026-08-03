"""Tests for GAE computation and the rollout buffer."""

from __future__ import annotations

import random

import torch

from qkd_rl.algos.gae import compute_gae
from qkd_rl.algos.rollout_buffer import RolloutBuffer, RolloutStep


def _step(reward: float, value: float, terminated: bool = False) -> RolloutStep:
    return RolloutStep(
        obs=None,
        actions={},
        log_probs={},
        entropies={},
        value=torch.tensor(value, dtype=torch.float32),
        reward=reward,
        terminated=terminated,
        truncated=False,
    )


def test_compute_gae_with_unit_discount():
    rewards = [1.0, 1.0, 1.0]
    values = torch.tensor([0.5, 0.5, 0.5])
    returns, advantages = compute_gae(
        rewards=rewards,
        values=values,
        terminated=[False, False, False],
        last_value=0.5,
        gamma=1.0,
        gae_lambda=1.0,
    )
    torch.testing.assert_close(advantages, torch.tensor([3.0, 2.0, 1.0]))
    torch.testing.assert_close(returns, torch.tensor([3.5, 2.5, 1.5]))


def test_compute_gae_termination_stops_bootstrap():
    rewards = [1.0, 1.0, 1.0]
    values = torch.tensor([0.5, 0.5, 0.5])
    returns, advantages = compute_gae(
        rewards=rewards,
        values=values,
        terminated=[False, False, True],
        last_value=0.5,
        gamma=1.0,
        gae_lambda=1.0,
    )
    torch.testing.assert_close(advantages, torch.tensor([2.5, 1.5, 0.5]))


def test_compute_gae_zero_discount():
    rewards = [1.0, 1.0, 1.0]
    values = torch.tensor([0.5, 0.5, 0.5])
    returns, advantages = compute_gae(
        rewards=rewards,
        values=values,
        terminated=[False, False, False],
        last_value=0.0,
        gamma=0.0,
        gae_lambda=0.95,
    )
    torch.testing.assert_close(returns, torch.tensor([1.0, 1.0, 1.0]))
    torch.testing.assert_close(advantages, torch.tensor([0.5, 0.5, 0.5]))


def test_rollout_buffer_finishes_episodes_and_samples():
    buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
    for reward, value in [(1.0, 0.5), (1.0, 0.5), (1.0, 0.5)]:
        buffer.add(_step(reward, value))
    buffer.finish_episode(last_value=0.5)

    for reward, value in [(0.0, 0.2), (0.0, 0.2)]:
        buffer.add(_step(reward, value))
    buffer.finish_episode(last_value=0.2)

    assert len(buffer) == 5
    for step in buffer.steps:
        assert step.returns is not None and step.advantages is not None

    sampled = [step for batch in buffer.sample(minibatch_size=2, rng=random.Random(0)) for step in batch]
    assert len(sampled) == 5
    assert {id(step) for step in sampled} == {id(step) for step in buffer.steps}

    buffer.clear()
    assert len(buffer) == 0

