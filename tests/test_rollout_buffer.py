"""Tests for GAE computation and the rollout buffer."""

from __future__ import annotations

import random

import torch

from qkd_rl.rl.algos.gae import compute_gae
from qkd_rl.rl.algos.rollout_buffer import RolloutBuffer, RolloutStep


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
        mean_log_prob=torch.tensor(0.0),
        mean_entropy=torch.tensor(0.0),
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


def test_mc_value_target_removes_bootstrap_from_returns():
    # "mc": only the CRITIC target (returns) is a bootstrap-free
    # Monte-Carlo sum; the actor's advantages stay the GAE ones, so the
    # policy gradient keeps GAE's variance reduction and value baseline.
    rewards = [1.0, 2.0, 3.0]
    values = torch.tensor([100.0, 100.0, 100.0])  # wildly biased value
    gae_buffer = RolloutBuffer(gamma=0.9, gae_lambda=0.95, value_target="gae")
    mc_buffer = RolloutBuffer(gamma=0.9, gae_lambda=0.95, value_target="mc")
    for r in rewards:
        gae_buffer.add(_step(r, 100.0))
        mc_buffer.add(_step(r, 100.0))
    gae_buffer.finish_episode(last_value=100.0)
    mc_buffer.finish_episode(last_value=100.0)

    gae_returns = torch.stack([s.returns for s in gae_buffer.steps])
    mc_returns = torch.stack([s.returns for s in mc_buffer.steps])
    gae_adv = torch.stack([s.advantages for s in gae_buffer.steps])
    mc_adv = torch.stack([s.advantages for s in mc_buffer.steps])

    # MC returns = discounted reward sums only (no +100 value offset).
    expected = torch.tensor([1.0 + 0.9 * 2.0 + 0.81 * 3.0, 2.0 + 0.9 * 3.0, 3.0])
    torch.testing.assert_close(mc_returns, expected)
    # Actor advantages are identical in both modes (GAE).
    torch.testing.assert_close(mc_adv, gae_adv)
    # The GAE returns carry the biased value (+100-ish), the MC ones do not.
    assert gae_returns.mean() > mc_returns.mean() + 50.0


def test_mc_value_target_invalid_mode_rejected():
    import pytest

    with pytest.raises(ValueError):
        RolloutBuffer(gamma=0.9, gae_lambda=0.95, value_target="nope")


def _shipped_step(log_probs: dict[str, torch.Tensor]) -> RolloutStep:
    return RolloutStep(
        obs=None,
        actions={"n0": ("a", "b")},
        log_probs=log_probs,
        entropies={k: v * 2 for k, v in log_probs.items()},
        value=torch.tensor(1.0),
        reward=1.0,
        terminated=False,
        truncated=False,
    )


def test_shipping_collapses_repeated_per_node_scalars():
    """Every node carries the same log-prob, so the pickled form collapses it.

    Workers ship one episode back per update and the trainer unpickles all of
    it serially, so the object count in this payload is the rollout's critical
    path. The observation is only half the bytes; the other half is ~180 tiny
    per-node tensors per step that all hold the same value.
    """
    import pickle

    nodes = [f"n{i}" for i in range(50)]
    shared = torch.tensor(-1.25)

    back = pickle.loads(pickle.dumps(_shipped_step({n: shared for n in nodes})))
    assert list(back.log_probs) == nodes
    for node in nodes:
        torch.testing.assert_close(back.log_probs[node], shared)
        torch.testing.assert_close(back.entropies[node], shared * 2)

    # The effect that matters: at a fixed node count the shipped payload no
    # longer carries one tensor per node. Measured against the same step with
    # genuinely distinct per-node values -- which is also the case the collapse
    # must NOT fire on, so the two tests pin each other.
    n = 500
    collapsed = pickle.dumps(_shipped_step({f"n{i}": shared.clone() for i in range(n)}))
    distinct = pickle.dumps(_shipped_step({f"n{i}": torch.tensor(-1.0 - i * 1e-3) for i in range(n)}))
    assert len(collapsed) < len(distinct) / 2


def test_shipping_preserves_distinct_per_node_values():
    """The collapse must not fire when the values actually differ."""
    import pickle

    step = _shipped_step({f"n{i}": torch.tensor(-1.25 * (i + 1)) for i in range(6)})
    back = pickle.loads(pickle.dumps(step))
    for i in range(6):
        torch.testing.assert_close(back.log_probs[f"n{i}"], torch.tensor(-1.25 * (i + 1)))


def test_shipping_handles_empty_and_single_entry_dicts():
    import pickle

    for value in ({}, {"only": torch.tensor(0.5)}):
        back = pickle.loads(pickle.dumps(_shipped_step(value)))
        assert list(back.log_probs) == list(value)
        for node, tensor in value.items():
            torch.testing.assert_close(back.log_probs[node], tensor)

