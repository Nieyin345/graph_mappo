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
    # v8 auxiliary target: for physical edge e at this step's position in
    # ``obs.physical_edge_ids``, 1.0 if e lies on some request's path within the
    # next `horizon` steps, else 0.0. Aligned to the SAME edge list the forecast
    # head is aligned to, so loss time needs no id mapping. None when the model
    # has no forecast head. Filled by `fill_usage_labels` once the episode is
    # complete (the label is forward-looking, so it needs the future steps).
    usage_label: torch.Tensor | None = None

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


# Label targets for the usage-forecast head. "used"/"new" are the method's
# labels; "persist"/"shuffle" are journal controls (generic availability
# forecast / content-free label with the same class balance).
USAGE_LABEL_MODES = ("used", "new", "persist", "shuffle")


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
        # v8: label horizon in slots (0 = do not compute usage labels). Set by
        # the trainer from actor.usage_forecast.horizon so a v3 run costs nothing.
        self.usage_horizon = 0
        # v8 label semantics:
        #   "used"    -- e is on some request path at any step in [t, t+K).
        #                The first batch used this. Measured problem: it is
        #                dominated by persistence (an edge in use now is in use
        #                100% of the time over the horizon), so the head just
        #                relearns "keep serving what you already serve" and ends
        #                up worse than a copy of the current mask.
        #   "new"     -- e is used in (t, t+K] AND is NOT used at t. Removes the
        #                persistence term, leaving only the new activations --
        #                the only place proactive preparation can add anything.
        self.usage_label_mode = "used"

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
        if self.usage_horizon > 0:
            self._fill_usage_labels(episode)
        self._episode_start = len(self.steps)

    def _fill_usage_labels(self, episode: list[RolloutStep]) -> None:
        """Set ``usage_label`` for every step in one finished episode.

        Label for step t on physical edge e, per ``usage_label_mode``:
            "used": 1.0 if e is on some pending request's path at ANY step in
                    [t, t + horizon), else 0.0
            "new":  1.0 if e is on a path at some step in (t, t + horizon] but
                    NOT at t, else 0.0
            "persist": (control) 1.0 if e is in the graph at EVERY step of
                    (t, t + horizon] -- demand-agnostic availability forecast
            "shuffle": (control) the "new" label randomly permuted within the
                    step -- same positive count, no edge identity

        Source: ``obs.request_path_edge_mask``, which the graph builder already
        fills every step (per physical edge, aligned with physical_edge_ids). It
        is the CURRENT-step version; taking the union over the next `horizon`
        steps turns it into the forward-looking label. Because it is the request
        stream's own path set, the label is independent of the policy's actions
        -- it cannot be farmed by hoarding keys.

        "new" exists because "used" is dominated by persistence: measured, an
        edge on a path now is still on a path 100% of the time across the
        horizon, while an edge not on a path becomes one only 10.6% of the time.
        "used" therefore mostly teaches "keep doing what you are doing", which
        the actor's own current-mask attention already encodes. "new" subtracts
        that term, leaving the new activations -- the only place proactive
        preparation can add anything.

        Steps whose mask is missing (None) are treated as all-zero, matching the
        builder's empty-request case.
        """
        horizon = self.usage_horizon
        mode = getattr(self, "usage_label_mode", "used")
        if mode not in USAGE_LABEL_MODES:
            raise ValueError(
                f"unknown usage label_mode {mode!r}; expected one of {USAGE_LABEL_MODES}"
            )
        n = len(episode)
        # Precompute per-step edge-id lists and on-path id sets.
        ids_all: list[list[str]] = []
        on_path: list[set[str]] = []
        for step in episode:
            mask = getattr(step.obs, "request_path_edge_mask", None)
            ids = list(getattr(step.obs, "physical_edge_ids", None) or [])
            ids_all.append(ids)
            on_path.append(
                {ids[pos] for pos, flag in enumerate(mask or []) if flag and pos < len(ids)}
            )
        present: list[set[str]] | None = (
            [set(ids) for ids in ids_all] if mode == "persist" else None
        )
        for t, step in enumerate(episode):
            ids_t = ids_all[t]
            size_t = len(ids_t)
            if size_t == 0:
                step.usage_label = torch.zeros((0,), dtype=torch.float32)
                continue
            label = torch.zeros(size_t, dtype=torch.float32)
            if mode == "persist":
                # Control (a): demand-agnostic availability forecast -- e is in
                # the graph (visible, rate >= min) at EVERY step of (t, t+K].
                # Truncated at the episode end; an empty window is vacuously 1.
                keep = set(ids_t)
                for u in range(t + 1, min(n, t + horizon + 1)):
                    keep &= present[u]
                for pos, eid in enumerate(ids_t):
                    if eid in keep:
                        label[pos] = 1.0
                step.usage_label = label
                continue
            # A step's edge list can differ in ORDER/length from step t's, so
            # accumulate by edge id over a window. "used" includes step t; "new"
            # starts at t+1 so the current step cannot satisfy the future term.
            # 窗口上界（2026-09-27 修，代码审查 A4）："used" = [t, t+K)（含当前
            # 步共 K 槽）；"new" = (t, t+K]（K 个未来槽）。原实现两种模式都用
            # min(n, t+horizon) ⟹ "new" 实际只覆盖 t+1..t+K-1（K-1 槽），
            # horizon=1 时窗口为空、全零标签。
            future = mode != "used"          # "new" and its shuffled control
            lo = t + 1 if future else t
            hi = min(n, t + horizon + (1 if future else 0))
            window_ids: set[str] = set()
            for u in range(lo, hi):
                window_ids |= on_path[u]
            now_ids = on_path[t] if future else set()
            for pos, eid in enumerate(ids_t):
                if eid in window_ids and eid not in now_ids:
                    label[pos] = 1.0
            if mode == "shuffle":
                # Control (a): same per-step positive count as "new", edge
                # identity destroyed. Private generator -> the global torch RNG
                # (and hence the rollouts) stay identical to the "new" arm.
                gen = torch.Generator()
                gen.manual_seed(1_000_003 * t + 7_919 * size_t + int(label.sum().item()))
                label = label[torch.randperm(size_t, generator=gen)]
            step.usage_label = label

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
