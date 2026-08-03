"""Rollout buffer for MAPPO training.

Stores full-graph observations for every step, computes GAE per episode, and
yields minibatches of steps for PPO updates. Each minibatch item is one time
step (a full graph observation); the graph encoder runs once per item because
node/edge counts vary between observations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import torch

from qkd_rl.algos.gae import compute_gae
from qkd_rl.env.graph_builder import GraphObservation


@dataclass
class RolloutStep:
    obs: GraphObservation
    actions: dict[str, str]
    log_probs: dict[str, torch.Tensor]
    entropies: dict[str, torch.Tensor]
    value: torch.Tensor
    reward: float
    terminated: bool
    truncated: bool
    returns: torch.Tensor | None = None
    advantages: torch.Tensor | None = None


class RolloutBuffer:
    """Stores steps of one or more episodes and computes GAE per episode."""

    def __init__(
        self,
        gamma: float,
        gae_lambda: float,
        device: torch.device | str = "cpu",
    ):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.device = torch.device(device)
        self.steps: list[RolloutStep] = []
        self._episode_start = 0

    def __len__(self) -> int:
        return len(self.steps)

    def add(self, step: RolloutStep) -> None:
        self.steps.append(step)

    def finish_episode(self, last_value: torch.Tensor | float) -> None:
        """Compute GAE for steps collected since the last episode boundary."""
        episode = self.steps[self._episode_start :]
        if not episode:
            return
        values = torch.stack([step.value.detach().to(self.device) for step in episode])
        returns, advantages = compute_gae(
            rewards=[step.reward for step in episode],
            values=values,
            terminated=[step.terminated for step in episode],
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        for step, ret, adv in zip(episode, returns, advantages):
            step.returns = ret.detach()
            step.advantages = adv.detach()
        self._episode_start = len(self.steps)

    def sample(
        self,
        minibatch_size: int,
        rng: random.Random | None = None,
    ) -> Iterator[list[RolloutStep]]:
        """Yield minibatches of rollout steps (one full graph per step)."""
        if not self.steps:
            return
        indices = list(range(len(self.steps)))
        shuffle = rng.shuffle if rng is not None else random.shuffle
        shuffle(indices)
        size = max(1, int(minibatch_size))
        for start in range(0, len(indices), size):
            yield [self.steps[idx] for idx in indices[start : start + size]]

    def clear(self) -> None:
        self.steps.clear()
        self._episode_start = 0
