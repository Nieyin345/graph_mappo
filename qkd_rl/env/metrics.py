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
        self.arrived_requests = 0
        self.completed_requests = 0
        self.conflict_count = 0
        self._last_arrived_keys = 0.0
        # 2026-09-19 新增：逐局累计生成量与平均等待量。
        #
        # 为什么必须累计而不是只看 `last`：训练侧实测 RL 的密钥生成量在 30 轮里
        # 单调掉 41%，而 success_rate 与 reward 都不动（奖励零空间）。要量化
        # "省了多少密钥"就需要**整局总量**，`last` 只是最后一步的瞬时值。
        # 见 docs/训练诊断记录.md「奖励看不见的行为漂移」。
        self.generated_keys = 0.0
        self.waiting_keys_step_sum = 0.0
        self.last: dict = {}

    def add_arrivals(self, requests) -> None:
        amount = sum(req.amount for req in requests)
        self.arrived_keys += amount
        self.arrived_requests += len(requests)
        # Per-step arrival volume, surfaced through ``last`` so the training
        # diagnostics can report the served/failed/arrived balance.
        self._last_arrived_keys = amount

    def update(
        self,
        resolved_action: ResolvedAction,
        generated_keys: dict[str, float],
        serve_result: ServeResult,
        reward_detail: RewardDetail,
        qkp: LinkQKPPool,
        expired_requests: list = None,
    ) -> None:
        expired_keys = sum(max(0.0, req.amount - req.served_amount) for req in (expired_requests or []))
        self.steps += 1
        self.served_keys += serve_result.served_keys
        self.failed_keys += serve_result.failed_keys + expired_keys
        self.completed_requests += len(serve_result.served_requests)
        self.conflict_count += resolved_action.conflict_count
        # 生成量是**每步流量**，直接累加即为整局总量。
        # 等待量是**存量**，累加得到"等待密钥·步"（除以步数才是均值）。
        self.generated_keys += sum(generated_keys.values())
        self.waiting_keys_step_sum += serve_result.waiting_keys
        capacity = sum(qkp.capacities.values()) or 1.0
        level = sum(qkp.levels.values())
        self.last = {
            "reward": reward_detail.total,
            "served_keys": serve_result.served_keys,
            "failed_keys": serve_result.failed_keys + expired_keys,
            "waiting_keys": serve_result.waiting_keys,
            "arrived_keys": self._last_arrived_keys,
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
            "arrived_requests": self.arrived_requests,
            "completed_requests": self.completed_requests,
            "request_completion_rate": (
                self.completed_requests / self.arrived_requests if self.arrived_requests > 0 else 0.0
            ),
            "conflict_count": self.conflict_count,
            # 2026-09-19 新增：与 success_rate 并列的**密钥效率**维度。
            #
            # `success_rate = served/arrived` 对"为了服务这些需求一共生成了多少密钥"
            # **完全不敏感**。实测：RL 的生成量在 30 轮里掉 41% 而 success_rate
            # 与 reward 都不动 —— 这个真实优势因此从未被记录过
            # （三个训练种子对专家省 22.3/41.9/45.4%，逐个 15/15 同向）。
            #
            # `waiting_keys_mean` 是**存量均值**（累加÷步数），可直接与
            # `rollout_debug.jsonl` 的 `mean_waiting_keys` 对照。旧代码只把它
            # 留在 `last` 里（末步快照），末步的瞬时值不能代表整局。
            "generated_keys": self.generated_keys,
            "waiting_keys_mean": (
                self.waiting_keys_step_sum / self.steps if self.steps > 0 else 0.0
            ),
            "key_efficiency": (
                self.generated_keys / self.served_keys if self.served_keys > 0 else 0.0
            ),
        }

    def last_info(self, reward_detail: RewardDetail) -> dict:
        return {**self.last, "reward_detail": reward_detail}


