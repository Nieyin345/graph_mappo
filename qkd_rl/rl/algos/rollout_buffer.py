"""Rollout buffer for MAPPO training.

Stores full-graph observations for every step, computes GAE per episode, and
yields minibatches of steps for PPO updates. Each minibatch item is one time
step (a full graph observation); the graph encoder runs once per item because
node/edge counts vary between observations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Iterator

import torch

from qkd_rl.rl.algos.gae import compute_gae
from qkd_rl.env.graph_builder import GraphObservation


# ---- rollout telemetry shared by trainer and workers -----------------------
# The per-component reward breakdown (rollout_debug.jsonl) must be accumulated
# in BOTH rollout paths: the trainer's single-process loop and the worker
# processes (which see `info` first and ship episodes back). Keeping the key
# list and the accumulation here means the two paths cannot drift apart --
# historically that drift is what made n_rollout_workers>1 lose the breakdown
# and the pin to 1 stick.
# Scalar info fields copied straight from the env step.
_ROLLOUT_DEBUG_INFO_KEYS: tuple[str, ...] = (
    "generated_keys",
    "served_keys",
    "failed_keys",
    "waiting_keys",
    "qkp_utilization",
    "conflict_count",
    "arrived_keys",
)

# (debug key, RewardDetail attribute) pairs summed when the env provides the
# per-component reward breakdown.
_ROLLOUT_DEBUG_DETAIL_KEYS: tuple[tuple[str, str], ...] = (
    ("reward_total", "total"),
    ("reward_served", "served_reward"),
    ("reward_generated", "generated_reward"),
    ("reward_dense", "dense_reward"),
    ("reward_storage", "storage_reward"),
    ("reward_keep_active", "keep_active_reward"),
    ("reward_failed", "failed_penalty"),
    ("reward_waiting", "waiting_penalty"),
    ("reward_urgency_wait", "urgency_wait_penalty"),
    ("reward_switch", "switch_penalty"),
    ("reward_expired", "expired_key_penalty"),
    ("reward_conflict", "conflict_penalty"),
    ("attributed_served", "attributed_served"),
    ("history_utilized", "history_utilized"),
)

# Derived from the two lists above rather than hand-maintained: a key added to
# either source must not also be added here, or `accumulate_rollout_debug`
# raises KeyError on the first step (this drifted once already).
_ROLLOUT_DEBUG_KEYS: tuple[str, ...] = (
    "steps",
    "activated_edges",
    *_ROLLOUT_DEBUG_INFO_KEYS,
    *(key for key, _attr in _ROLLOUT_DEBUG_DETAIL_KEYS),
)


def new_rollout_debug() -> dict[str, float]:
    """Fresh accumulator with the canonical rollout-debug key set."""
    return dict.fromkeys(_ROLLOUT_DEBUG_KEYS, 0.0)


def accumulate_rollout_debug(debug: dict[str, float], info: dict, activated_count: int) -> None:
    """Fold one env step's telemetry into a rollout-debug accumulator.

    ``debug`` comes from ``new_rollout_debug()`` (trainer) or is created per
    episode in the worker; the trainer sums the per-episode dicts afterwards.
    """
    debug["steps"] += 1.0
    debug["activated_edges"] += float(activated_count)
    for key in _ROLLOUT_DEBUG_INFO_KEYS:
        debug[key] += float(info.get(key, 0.0))
    detail = info.get("reward_detail")
    if detail is not None:
        for key, attr in _ROLLOUT_DEBUG_DETAIL_KEYS:
            debug[key] += float(getattr(detail, attr))


@dataclass
class RolloutStep:
    obs: GraphObservation
    actions: dict[str, tuple[str, str]]
    log_probs: dict[str, torch.Tensor]
    entropies: dict[str, torch.Tensor]
    value: torch.Tensor
    reward: float
    terminated: bool
    truncated: bool
    mean_log_prob: torch.Tensor | None = None
    mean_entropy: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    matched_edges: list[tuple[str, str]] | None = None

    # ---- cross-process shipping -------------------------------------------
    # RolloutStep is what the workers send back to the trainer, and the trainer
    # is on the critical path: it unpickles every episode serially while the
    # workers sit idle, so the payload size IS the rollout's wall time.
    #
    # Measured: 115 KB per step is shipped, but the observation is only 58 KB
    # of that. The difference is `log_probs` / `entropies`, which hold ONE TORCH
    # TENSOR PER NODE even though every node carries the same scalar (see the
    # PPO comment in MAPPOTrainer: "every node shares the same scalar, so any
    # node id yields it"). That is ~180 tiny tensors per step, and pickling
    # runs at ~10 MB/s because of them.
    #
    # Collapsing each to (keys, value) keeps the field's meaning identical --
    # __setstate__ rebuilds the same dict, with the nodes sharing one tensor --
    # while cutting the object count per step by roughly two orders of magnitude.
    _SHARED_FIELDS = ("log_probs", "entropies")

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for name in self._SHARED_FIELDS:
            value = state.get(name)
            if not value or len(value) < 2:
                continue
            first = next(iter(value.values()))
            if all(item is first or torch.equal(item, first) for item in value.values()):
                state[name] = (tuple(value.keys()), first)
        return state

    def __setstate__(self, state: dict) -> None:
        for name in self._SHARED_FIELDS:
            value = state.get(name)
            # A 2-tuple of (keys, value); the expanded form is a dict.
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], tuple):
                keys, shared = value
                state[name] = dict.fromkeys(keys, shared)
        self.__dict__.update(state)


def state_free_obs(obs: GraphObservation | None) -> GraphObservation | None:
    """Shallow copy of an observation with its ``EnvState`` dropped.

    ``GraphObservation.state`` is ~90% of a full-scenario observation: the env
    snapshot carries two 1978-entry dicts and a ``LazyEdgeWindows`` whose cache
    can hold one ``EdgeWindow`` per registered link (~2 MB once the whole link
    table has been touched). Nothing on the RL path reads it -- only the offline
    baselines do, from their own ``env.reset`` observations.

    Call this when the ``RolloutStep`` is *created*, not when it is buffered:
    the collectors accumulate a whole episode's steps before handing them to
    ``RolloutBuffer.add``, so stripping only at ``add`` time still holds every
    step's EnvState alive for the length of the episode (8 envs x 360 steps x
    ~2 MB is ~5.7 GB). The returned copy shares the feature arrays, the id
    lists and the mask arrays, so it costs one object allocation.

    ``RolloutBuffer.add`` calls this again as a safety net; it is idempotent.
    """
    if obs is None or obs.state is None:
        return obs
    return replace(obs, state=None)


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
        # Critic target: "gae" (default) stores the standard GAE return
        # (advantages + bootstrapped value); "mc" stores bootstrap-free
        # Monte-Carlo returns instead. Only the CRITIC target is switched in
        # "mc" mode; the actor's GAE advantages are kept in both modes, so the
        # policy gradient keeps GAE's variance reduction and a value baseline.
        # "mc" was introduced to break a value-normalizer feedback loop that
        # no longer exists in this code (returns are not normalized anywhere),
        # so "gae" is the recommended default; "mc" is kept for ablations.
        if value_target not in ("gae", "mc"):
            raise ValueError(f"value_target must be 'gae' or 'mc', got {value_target!r}")
        self.value_target = value_target
        self.steps: list[RolloutStep] = []
        self._episode_start = 0

    def __len__(self) -> int:
        return len(self.steps)

    def add(self, step: RolloutStep) -> None:
        """Append a step, dropping the observation's EnvState as a safety net.

        The collectors already strip it via :func:`state_free_obs`; this guards
        any caller that builds a ``RolloutStep`` directly.
        """
        step.obs = state_free_obs(step.obs)
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
            # truncated at the episode end. Only the critic target is swapped
            # to MC (no value bootstrap -> no feedback from the possibly
            # biased critic into its own target); the actor keeps the GAE
            # advantages computed above.
            mc_returns = torch.zeros_like(returns)
            acc = torch.zeros((), dtype=torch.float32, device=self.device)
            rewards_t = torch.as_tensor(
                [step.reward for step in episode], dtype=torch.float32, device=self.device
            )
            for t in reversed(range(rewards_t.shape[0])):
                acc = rewards_t[t] + self.gamma * acc
                mc_returns[t] = acc
            returns = mc_returns
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
