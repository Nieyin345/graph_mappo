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
    joint_log_prob: torch.Tensor | None = None
    joint_entropy: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    matched_edges: list[str] | None = None


class RolloutBuffer:
    """Stores steps of one or more episodes and computes GAE per episode."""

    def __init__(
        self,
        gamma: float,
        gae_lambda: float,
        device: torch.device | str = "cpu",
        value_target: str = "gae",
    ):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.device = torch.device(device)
        # Critic target: "gae" stores the standard GAE return (advantages +
        # bootstrapped value); "mc" stores bootstrap-free Monte-Carlo returns
        # instead. "mc" breaks the positive-feedback loop where a biased
        # value normalizer biases the bootstrap, which biases the returns that
        # re-train the normalizer (measured drift to ~5x the true return
        # scale). GAE advantages for the actor are unchanged in both modes.
        if value_target not in ("gae", "mc"):
            raise ValueError(f"value_target must be 'gae' or 'mc', got {value_target!r}")
        self.value_target = value_target
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
        if self.value_target == "mc":
            # Bootstrap-free Monte-Carlo returns: R_t = sum_k gamma^k r_{t+k}
            # truncated at the episode end. No value bootstrap -> no feedback
            # from the (possibly biased) critic into its own target.
            mc_returns = torch.zeros_like(returns)
            acc = torch.zeros((), dtype=torch.float32, device=self.device)
            rewards_t = torch.as_tensor(
                [step.reward for step in episode], dtype=torch.float32, device=self.device
            )
            for t in reversed(range(rewards_t.shape[0])):
                acc = rewards_t[t] + self.gamma * acc
                mc_returns[t] = acc
            returns = mc_returns
            # The actor also uses the raw MC returns (whitened per minibatch
            # in PPO), so an imperfect critic cannot bias the policy gradient.
            advantages = mc_returns
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
