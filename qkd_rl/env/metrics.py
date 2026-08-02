from __future__ import annotations

from qkd_rl.env.action_resolver import ResolvedAction
from qkd_rl.env.qkp import LinkQKPPool
from qkd_rl.env.request import ServeResult
from qkd_rl.env.reward import RewardDetail


class MetricsTracker:
    def reset(self) -> None:
        self.steps = 0
        self.arrived_keys = 0.0
        self.served_keys = 0.0
        self.failed_keys = 0.0
        self.conflict_count = 0
        self.last: dict = {}

    def add_arrivals(self, amount: float) -> None:
        self.arrived_keys += amount

    def update(
        self,
        resolved_action: ResolvedAction,
        generated_keys: dict[str, float],
        serve_result: ServeResult,
        reward_detail: RewardDetail,
        qkp: LinkQKPPool,
    ) -> None:
        self.steps += 1
        self.served_keys += serve_result.served_keys
        self.failed_keys += serve_result.failed_keys
        self.conflict_count += resolved_action.conflict_count
        capacity = sum(qkp.capacities.values()) or 1.0
        level = sum(qkp.levels.values())
        self.last = {
            "reward": reward_detail.total,
            "served_keys": serve_result.served_keys,
            "failed_keys": serve_result.failed_keys,
            "waiting_keys": serve_result.waiting_keys,
            "generated_keys": sum(generated_keys.values()),
            "conflict_count": resolved_action.conflict_count,
            "qkp_utilization": level / capacity,
        }

    def episode_summary(self) -> dict:
        return {
            "steps": self.steps,
            "arrived_keys": self.arrived_keys,
            "served_keys": self.served_keys,
            "failed_keys": self.failed_keys,
            "success_rate": self.served_keys / self.arrived_keys if self.arrived_keys > 0 else 0.0,
            "conflict_count": self.conflict_count,
        }

    def last_info(self, reward_detail: RewardDetail) -> dict:
        return {**self.last, "reward_detail": reward_detail}

