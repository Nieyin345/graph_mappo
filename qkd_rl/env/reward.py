from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

from qkd_rl.core.types import KeyRequest
from qkd_rl.env.action_resolver import ResolvedAction
from qkd_rl.env.qkp import LinkQKPPool
from qkd_rl.env.request import ServeResult
from qkd_rl.env.routing import AllocationResult

_ENABLED_FLAG_KEYS = {
    "success_enabled",
    "success_delta_enabled",
    "served_enabled",
    "attribution_enabled",
    "storage_enabled",
    "raw_generation_enabled",
    "dense_enabled",
    "failed_enabled",
    "waiting_enabled",
    "switch_enabled",
    "keep_active_enabled",
    "overflow_enabled",
    "expired_key_enabled",
    "conflict_enabled",
    "urgency_wait_enabled",
}


@dataclass
class RewardDetail:
    total: float
    served_reward: float
    generated_reward: float
    failed_penalty: float
    waiting_penalty: float
    overflow_penalty: float
    expired_key_penalty: float
    conflict_penalty: float
    dense_reward: float = 0.0
    switch_penalty: float = 0.0
    keep_active_reward: float = 0.0
    baseline_reward: float = 0.0
    # Attribution diagnostics: served keys traceable to this slot's activation
    # (``attributed_served``) and history stock unlocked by a mixed channel
    # (``history_utilized``); storage keys on usable paths (``storage_reward``).
    attributed_served: float = 0.0
    history_utilized: float = 0.0
    storage_reward: float = 0.0
    # Deadline-urgency weighted backlog penalty (see ``urgency_wait_enabled``).
    # Split out from ``waiting_penalty`` so the flat and urgency-shaped backlog
    # terms can be compared in the reward decomposition.
    urgency_wait_penalty: float = 0.0
    # Raw urgency-weighted backlog (key-flow units, pre-normalization), kept for
    # diagnostics: it says how much demand is "about to be lost".
    urgency_weighted_keys: float = 0.0


class RewardFunction:
    def __init__(self, config: dict):
        self.config = config
        self.mode = str(config.get("mode", "shaped"))
        # Sliding-window mean of arrived key demand. Used as the normalization
        # denominator so a single Poisson-empty slot (arrived=0) does not blow
        # up the waiting/failed penalties that are stock-based, not flow-based.
        self.normalize_window = max(1, int(config.get("normalize_window", 60)))
        # Denominator floor: a near-empty demand window (e.g. night hours or
        # the first slots after reset) must not amplify the stock-based
        # penalties (waiting/overflow/expired) into million-scale rewards.
        self.normalize_floor = float(config.get("normalize_floor", 1000.0))
        # Per-step reward clipping (0 = disabled): bounds the critic target so
        # a single overflow/expiry outlier cannot blow up the value function.
        self.clip_abs = float(config.get("clip_abs", 0.0))
        # Global multiplier on every reported component including `total`.
        #
        # Rationale: PPO's critic has to predict the discounted return, and Adam
        # moves a parameter by roughly its learning rate per step. With the raw
        # reward the full-scenario return is ~70, so the value head would need
        # ~70/lr steps just to reach the target's mean -- it never gets there,
        # `value_std` collapses, GAE loses its baseline and the advantages become
        # the raw returns (measured: value_std/return_std = 1e-4, corr(V,R) =
        # 0.20). Scaling the reward so returns are O(1) makes the critic
        # fittable. A positive constant scale does not change the optimal policy,
        # and it is applied to every component so the diagnostic decomposition
        # still sums to `total`. Scale the observed return, not the reward
        # magnitude: return ~= per-step reward / (1 - gamma).
        self.reward_scale = float(config.get("reward_scale", 1.0))
        # Waiting backlog penalty: we keep the flow-based delta term, but add
        # a smaller stock term so an already-growing queue still matters even
        # when it temporarily stops increasing.
        self.waiting_stock_weight = float(config.get("waiting_stock_weight", 0.01))
        self._arrived_total = 0.0
        self._served_total = 0.0
        self._prev_success_rate = 0.0
        self._arrived_window: deque[float] = deque(maxlen=self.normalize_window)

    def reset(self) -> None:
        """Clear the arrival window (called on env reset to avoid cross-episode
        leakage of the demand baseline)."""
        self._arrived_total = 0.0
        self._served_total = 0.0
        self._prev_success_rate = 0.0
        self._arrived_window.clear()

    def _clip(self, value: float) -> float:
        if self.clip_abs <= 0.0:
            return value
        return max(-self.clip_abs, min(self.clip_abs, float(value)))

    # RewardDetail fields that are key COUNTS, not reward amounts, and must not
    # be rescaled with the reward.
    _COUNT_FIELDS = (
        "attributed_served",
        "history_utilized",
        "urgency_weighted_keys",
    )

    def _scaled(self, detail: RewardDetail) -> RewardDetail:
        """Apply the global reward scale to every reward component, keeping the
        decomposition consistent with ``total``. Key-count diagnostics are left
        alone. Identity when scale == 1.0."""
        if self.reward_scale == 1.0:
            return detail
        return RewardDetail(
            **{
                name: value * self.reward_scale
                if name not in self._COUNT_FIELDS
                else value
                for name, value in asdict(detail).items()
            }
        )

    @staticmethod
    def _enabled(config: dict, name: str, default: bool = True) -> bool:
        key = f"{name}_enabled"
        if key not in _ENABLED_FLAG_KEYS:
            raise ValueError(f"Unknown reward component flag: {key}")
        return bool(config.get(key, default))

    def _dense_importance_reward(
        self,
        added_by_edge: dict[str, float] | None,
        relay_importance: dict[str, float] | None,
    ) -> float:
        """Dense shaping reward: stored keys weighted by relay importance.

        Links that actually accept keys on high-importance relay paths get
        positive shaping; an optional penalty reduces the reward of
        high-rate/low-importance activation. All values are normalized by a
        configurable reference so the term stays comparable to the other
        reward components.
        """
        weight = float(self.config.get("dense_generation_importance_weight", 0.0))
        if weight <= 0.0 or not added_by_edge or not relay_importance:
            return 0.0
        penalty_weight = float(self.config.get("dense_low_importance_penalty", 0.0))
        reference = float(self.config.get("dense_reference", 1.0)) or 1.0
        added_total = sum(max(0.0, amount) for amount in added_by_edge.values())
        if added_total <= 0.0:
            return 0.0
        total = 0.0
        for edge_id, added in added_by_edge.items():
            importance = max(0.0, min(1.0, float(relay_importance.get(edge_id, 0.0))))
            value = added * importance
            if penalty_weight > 0.0:
                value -= penalty_weight * added * (1.0 - importance)
            total += value
        if bool(self.config.get("dense_normalize_by_added", False)):
            # Shape by the *share* of generated keys on high-importance paths
            # instead of absolute key volume. This keeps the dense reward in
            # [~0, weight] regardless of how much raw generation happens.
            return weight * (total / added_total)
        # Supply-side baseline: the dense generation term is normalized by its
        # own dense_reference (NOT the arrived-demand window, which is the
        # demand-side baseline for served/waiting/failed). Keeping the two
        # baselines separate prevents the supply inventory scale (rate*slot,
        # millions) from being crushed to ~0 by the much smaller demand scale,
        # and lets the importance-weighted generation act as a dense reward
        # between the sparse service events.
        return weight * total / reference

    def compute(
        self,
        serve_result: ServeResult,
        allocation: AllocationResult,
        expired_requests: list[KeyRequest],
        expired_keys: float,
        resolved_action: ResolvedAction,
        qkp: LinkQKPPool,
        arrived_keys: float = 0.0,
        served_keys: float = 0.0,
        waiting_keys: float = 0.0,
        waiting_delta: float = 0.0,
        switch_count: int = 0,
        keep_active_count: int = 0,
        added_by_edge: dict[str, float] | None = None,
        relay_importance: dict[str, float] | None = None,
        baseline_reward: float = 0.0,
        serve_events: list[tuple[str, float, float]] | None = None,
        from_new_by_edge: dict[str, float] | None = None,
        storage_pathness: dict[str, float] | None = None,
    ) -> RewardDetail:
        dense_reward = (
            self._dense_importance_reward(added_by_edge, relay_importance)
            if self._enabled(self.config, "dense")
            else 0.0
        )
        if self.mode == "baseline_score":
            total = float(baseline_reward)
            if self.clip_abs > 0.0:
                total = self._clip(total)
            return self._scaled(RewardDetail(
                total=total,
                served_reward=total,
                generated_reward=0.0,
                failed_penalty=0.0,
                waiting_penalty=0.0,
                overflow_penalty=0.0,
                expired_key_penalty=0.0,
                conflict_penalty=0.0,
                baseline_reward=total,
            ))
        if self.mode == "success_rate":
            arrived = max(0.0, float(arrived_keys))
            served = max(0.0, float(served_keys))
            self._arrived_total += arrived
            self._served_total += served
            current = self._served_total / self._arrived_total if self._arrived_total > 0.0 else 0.0
            success_delta = (current - self._prev_success_rate) * int(
                self._enabled(self.config, "success")
            )
            self._prev_success_rate = current
            failed_keys_total = serve_result.failed_keys + sum(
                max(0.0, req.amount - req.served_amount) for req in expired_requests
            )
            failed_penalty = (
                float(self.config.get("failed_weight", 0.0)) * failed_keys_total
                if self._enabled(self.config, "failed")
                else 0.0
            )
            waiting_penalty = (
                (
                    float(self.config.get("waiting_weight", 0.0)) * max(0.0, waiting_delta)
                    + float(self.config.get("waiting_stock_weight", 0.01)) * max(0.0, waiting_keys)
                )
                if self._enabled(self.config, "waiting")
                else 0.0
            )
            overflow_penalty = (
                float(self.config.get("overflow_weight", 0.0)) * allocation.overflow_keys
                if self._enabled(self.config, "overflow")
                else 0.0
            )
            expired_key_penalty = (
                float(self.config.get("expired_key_weight", 0.0)) * expired_keys
                if self._enabled(self.config, "expired_key")
                else 0.0
            )
            conflict_penalty = (
                float(self.config.get("conflict_weight", 0.0)) * resolved_action.conflict_count
                if self._enabled(self.config, "conflict")
                else 0.0
            )
            switch_penalty = (
                float(self.config.get("switch_weight", 0.0)) * max(0, int(switch_count))
                if self._enabled(self.config, "switch")
                else 0.0
            )
            # Key-flow penalties are normalized by the arrived-demand baseline;
            # switch is a link-toggle count and keeps its own O(1) scale so it
            # is not crushed to ~1e-6 by a bits-per-slot denominator.
            if self.config.get("normalize_penalties_by_arrived", False):
                self._arrived_window.append(arrived)
                denom = max(sum(self._arrived_window) / len(self._arrived_window), self.normalize_floor)
                failed_penalty /= denom
                waiting_penalty /= denom
                overflow_penalty /= denom
                expired_key_penalty /= denom
                conflict_penalty /= denom
            else:
                reference = float(self.config.get("penalty_reference", 1_000_000.0)) or 1.0
                failed_penalty /= reference
                waiting_penalty /= reference
                overflow_penalty /= reference
                expired_key_penalty /= reference
                conflict_penalty /= reference
            total = (
                success_delta
                + dense_reward
                - failed_penalty
                - waiting_penalty
                - overflow_penalty
                - expired_key_penalty
                - conflict_penalty
                - switch_penalty
            )
            if self.clip_abs > 0.0:
                switch_penalty = self._clip(switch_penalty)
                total = self._clip(total)
            return self._scaled(RewardDetail(
                total=total,
                served_reward=success_delta,
                generated_reward=dense_reward,
                failed_penalty=failed_penalty,
                waiting_penalty=waiting_penalty,
                overflow_penalty=overflow_penalty,
                expired_key_penalty=expired_key_penalty,
                conflict_penalty=conflict_penalty,
                dense_reward=dense_reward,
                switch_penalty=switch_penalty,
            ))

        served_weight = float(self.config.get("served_weight", 0.0))
        attributed_served = 0.0
        history_utilized = 0.0
        served_reward = 0.0
        if self._enabled(self.config, "served"):
            if self._enabled(self.config, "attribution", default=False):
                # Attribute served keys to the current link choice: keys traced
                # to this slot's generation count at full weight; history stock
                # unlocked by a mixed channel (a request served partly by new,
                # partly by old keys) counts at the discounted ratio. Service
                # purely from history stock carries no reward: it happened
                # without this slot's activation.
                ratio = float(self.config.get("history_utilization_ratio", 0.3))
                for _req_id, served, from_new in serve_events or []:
                    attributed_served += from_new
                    if from_new > 1.0e-9 and (served - from_new) > 1.0e-9:
                        history_utilized += served - from_new
                served_reward = served_weight * (attributed_served + ratio * history_utilized)
            else:
                served_reward = served_weight * serve_result.served_keys
        # Storage reward: new keys left unconsumed this slot are rewarded only
        # if stored on an edge that lies on some pending request's usable path
        # (activated or stocked), i.e. where they can be consumed later. This
        # replaces the empirical relay-importance shaping with a structural
        # "stored on a real channel" signal.
        storage_reward = 0.0
        if self._enabled(self.config, "storage", default=False):
            storage_weight = float(self.config.get("storage_reward_weight", 0.0))
            storage_reference = float(self.config.get("storage_reference", 1_000_000.0)) or 1.0
            for edge_id, added in (added_by_edge or {}).items():
                unused = max(0.0, added - (from_new_by_edge or {}).get(edge_id, 0.0))
                storage_reward += storage_weight * unused * float((storage_pathness or {}).get(edge_id, 0.0))
            storage_reward /= storage_reference
        # Useful generation: keys that actually fit into the pools
        # (min(rate x slot, capacity_left)). Rewarding stored keys instead of
        # raw rate makes activation of an already-full link neutral (no
        # reward) instead of a huge raw-rate overflow penalty.
        raw_generation = (
            float(self.config.get("generated_weight", 0.0)) * allocation.added_keys
            if self._enabled(self.config, "raw_generation")
            else 0.0
        )
        # displayed generation groups the demand-normalized raw term and the
        # supply-side dense term; set below after normalization.
        generated_reward = dense_reward
        failed_penalty = (
            float(self.config["failed_weight"])
            * (
                serve_result.failed_keys
                + sum(max(0.0, req.amount - req.served_amount) for req in expired_requests)
            )
            if self._enabled(self.config, "failed")
            else 0.0
        )
        # Backlog-aware: penalize both the *increase* in waiting stock and a
        # smaller amount of the remaining stock itself. This keeps the dense
        # signal from the delta term while still caring about queues that stay
        # large even when they stop growing for a step.
        waiting_penalty = (
            (
                float(self.config["waiting_weight"]) * max(0.0, waiting_delta)
                + self.waiting_stock_weight * max(0.0, waiting_keys)
            )
            if self._enabled(self.config, "waiting")
            else 0.0
        )
        overflow_penalty = (
            float(self.config["overflow_weight"]) * allocation.overflow_keys
            if self._enabled(self.config, "overflow")
            else 0.0
        )
        # Deadline-urgency weighted backlog: penalize *which* backlog is about
        # to expire, not just how much there is. ``serve()`` reports
        # ``urgency_weighted_keys = sum(urgency(req) * remaining)`` where
        # urgency rises from 0 (just arrived) to ~1 (about to expire), so the
        # penalty grows as a request's remaining lifetime shrinks. This is the
        # only reward term that distinguishes "serve the urgent one" from
        # "serve the fresh one" -- served and failed rewards are both flat in
        # urgency, so they shape the level but not the choice.
        urgency_wait_penalty = (
            float(self.config.get("urgency_wait_weight", 0.0))
            * max(0.0, float(getattr(serve_result, "urgency_weighted_keys", 0.0)))
            if self._enabled(self.config, "urgency_wait", default=False)
            else 0.0
        )
        expired_key_penalty = (
            float(self.config["expired_key_weight"]) * expired_keys
            if self._enabled(self.config, "expired_key")
            else 0.0
        )
        conflict_penalty = (
            float(self.config["conflict_weight"]) * resolved_action.conflict_count
            if self._enabled(self.config, "conflict")
            else 0.0
        )
        # Switching cost is a *count of toggled links*, not a key-flow: dividing
        # it by the arrived-key-demand window crushes it toward ~1e-6 (the scale
        # bug that let a raw greedy switch penalty float the episode). It is
        # applied at its own O(1) weight and kept outside the key-flow
        # normalization below.
        switch_penalty = (
            float(self.config.get("switch_weight", 0.0)) * max(0, int(switch_count))
            if self._enabled(self.config, "switch")
            else 0.0
        )
        keep_active_reward = (
            float(self.config.get("keep_active_weight", 0.0)) * max(0, int(keep_active_count))
            if self._enabled(self.config, "keep_active")
            else 0.0
        )
        raw_n = 0.0
        key_total = (
            served_reward + raw_generation
            - failed_penalty - waiting_penalty - urgency_wait_penalty
            - overflow_penalty - expired_key_penalty - conflict_penalty
        )
        total = key_total - switch_penalty + keep_active_reward
        fixed_reference = float(self.config.get("served_reference", 0.0) or 0.0)
        if fixed_reference > 0.0:
            # Fixed demand-side reference: the same served/penalty volume has
            # the same reward everywhere in the episode. The sliding arrived
            # mean (and its cold-start floor) made early served keys worth
            # many times more than later ones, biasing the policy toward
            # serving only the first minutes of the day.
            served_reward /= fixed_reference
            raw_n = raw_generation / fixed_reference
            failed_penalty /= fixed_reference
            waiting_penalty /= fixed_reference
            urgency_wait_penalty /= fixed_reference
            overflow_penalty /= fixed_reference
            expired_key_penalty /= fixed_reference
            conflict_penalty /= fixed_reference
            generated_reward = raw_n + dense_reward
            # keep_active_reward must be added here too: it is an O(1) link-count
            # term that is deliberately outside the key-flow normalization, and
            # omitting it made `keep_active_enabled: true` (env_full) a no-op.
            total = (
                key_total / fixed_reference + dense_reward + storage_reward
                - switch_penalty + keep_active_reward
            )
        elif self.config.get("normalize_by_arrived_demand", False):
            # Normalize the *demand-side* key-flow terms (served, waiting, ...)
            # by the recent mean of arrived key demand so they stay O(1)-ish
            # regardless of network scale. A sliding-window mean (instead of
            # the current slot's arrival) keeps the denominator stable when a
            # Poisson slot generates zero requests while waiting/failed stocks
            # remain large. The dense generation term is *supply-side* and is
            # normalized by its own dense_reference inside
            # _dense_importance_reward, so it does not enter this denominator.
            self._arrived_window.append(float(arrived_keys))
            denom = max(sum(self._arrived_window) / len(self._arrived_window), self.normalize_floor)
            served_reward /= denom
            raw_n = raw_generation / denom
            failed_penalty /= denom
            waiting_penalty /= denom
            urgency_wait_penalty /= denom
            overflow_penalty /= denom
            expired_key_penalty /= denom
            conflict_penalty /= denom
            generated_reward = raw_n + dense_reward
            total = (
                key_total / denom + dense_reward + storage_reward
                - switch_penalty + keep_active_reward
            )
        # Note: the former normalize_served_by_queue branch (served / (served +
        # waiting), saturated at ~1 once the queue drains) was removed. It
        # silently overrode served_reference and made the served reward
        # insensitive to absolute served volume. Served keys are now always
        # normalized by the fixed served_reference, so every bit of service
        # has a constant marginal reward and the reward scale is uniform
        # across the episode.
        # Success-rate delta: reward the *increase* in the cumulative episode
        # success rate (served / arrived) contributed by this step, so the
        # policy is directly incentivised to improve the final metric, not
        # just the per-step served volume. The delta is multiplied by a
        # configurable weight (default 5000) so its scale is comparable to
        # the other shaped reward components.
        if self._enabled(self.config, "success_delta"):
            self._arrived_total += float(arrived_keys)
            self._served_total += float(served_keys)
            current = self._served_total / self._arrived_total if self._arrived_total > 0.0 else 0.0
            delta = current - self._prev_success_rate
            self._prev_success_rate = current
            sw = float(self.config.get("success_delta_weight", 5000.0))
            total += delta * sw
        if self.clip_abs > 0.0:
            served_reward = self._clip(served_reward)
            generated_reward = self._clip(generated_reward)
            failed_penalty = self._clip(failed_penalty)
            waiting_penalty = self._clip(waiting_penalty)
            urgency_wait_penalty = self._clip(urgency_wait_penalty)
            overflow_penalty = self._clip(overflow_penalty)
            expired_key_penalty = self._clip(expired_key_penalty)
            conflict_penalty = self._clip(conflict_penalty)
            total = self._clip(total)
        return self._scaled(RewardDetail(
            total=total,
            served_reward=served_reward,
            generated_reward=generated_reward,
            failed_penalty=failed_penalty,
            waiting_penalty=waiting_penalty,
            overflow_penalty=overflow_penalty,
            expired_key_penalty=expired_key_penalty,
            conflict_penalty=conflict_penalty,
            dense_reward=dense_reward,
            switch_penalty=switch_penalty,
            keep_active_reward=keep_active_reward,
            attributed_served=attributed_served,
            history_utilized=history_utilized,
            storage_reward=storage_reward,
            urgency_wait_penalty=urgency_wait_penalty,
            urgency_weighted_keys=max(
                0.0, float(getattr(serve_result, "urgency_weighted_keys", 0.0))
            ),
        ))
