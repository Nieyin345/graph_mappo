from __future__ import annotations

from dataclasses import dataclass

from qkd_rl.core.types import KeyRequest
from qkd_rl.env.action_resolver import ResolvedAction
from qkd_rl.env.qkp import LinkQKPPool
from qkd_rl.env.request import ServeResult
from qkd_rl.env.routing import AllocationResult


@dataclass
class RewardDetail:
    total: float
    served_reward: float
    failed_penalty: float
    waiting_penalty: float
    overflow_penalty: float
    expired_key_penalty: float
    conflict_penalty: float


class RewardFunction:
    def __init__(self, config: dict):
        self.config = config

    def compute(
        self,
        serve_result: ServeResult,
        allocation: AllocationResult,
        expired_requests: list[KeyRequest],
        expired_keys: float,
        resolved_action: ResolvedAction,
        qkp: LinkQKPPool,
    ) -> RewardDetail:
        served_reward = float(self.config["served_weight"]) * serve_result.served_keys
        failed_penalty = float(self.config["failed_weight"]) * (
            serve_result.failed_keys + sum(req.amount for req in expired_requests)
        )
        waiting_penalty = float(self.config["waiting_weight"]) * serve_result.waiting_keys
        overflow_penalty = float(self.config["overflow_weight"]) * allocation.overflow_keys
        expired_key_penalty = float(self.config["expired_key_weight"]) * expired_keys
        conflict_penalty = float(self.config["conflict_weight"]) * resolved_action.conflict_count
        total = served_reward - failed_penalty - waiting_penalty - overflow_penalty - expired_key_penalty - conflict_penalty
        return RewardDetail(
            total=total,
            served_reward=served_reward,
            failed_penalty=failed_penalty,
            waiting_penalty=waiting_penalty,
            overflow_penalty=overflow_penalty,
            expired_key_penalty=expired_key_penalty,
            conflict_penalty=conflict_penalty,
        )

