"""Generalized Advantage Estimation (GAE) for MAPPO.

Computes per-step returns and advantages from rewards, value estimates, and
episode termination flags. The global critic produces one value per time step,
so all agents of one step share the same advantage.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def compute_gae(
    rewards: Sequence[float],
    values: torch.Tensor,
    terminated: Sequence[bool],
    last_value: torch.Tensor | float,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE returns and advantages.

    Args:
        rewards: per-step rewards, length ``T``.
        values: per-step value estimates, shape ``(T,)``.
        terminated: per-step flag; ``True`` means the episode ended naturally
            at this step so the next value is not bootstrapped.
        last_value: value estimate of the state after the final step (``0``
            when the final step was terminated).
        gamma: discount factor.
        gae_lambda: GAE smoothing parameter.

    Returns:
        ``(returns, advantages)``, each of shape ``(T,)``.
    """
    device = values.device
    values = values.to(torch.float32)
    rewards_t = torch.as_tensor(list(rewards), dtype=torch.float32, device=device)
    non_terminal = 1.0 - torch.as_tensor(list(terminated), dtype=torch.float32, device=device)
    last_value_t = torch.as_tensor(last_value, dtype=torch.float32, device=device)

    advantages = torch.zeros_like(values)
    gae = torch.zeros((), dtype=torch.float32, device=device)
    for t in reversed(range(values.shape[0])):
        next_value = values[t + 1] if t + 1 < values.shape[0] else last_value_t
        delta = rewards_t[t] - values[t] + gamma * next_value * non_terminal[t]
        gae = delta + gamma * gae_lambda * non_terminal[t] * gae
        advantages[t] = gae
    returns = advantages + values
    return returns, advantages
