from __future__ import annotations

import random
from dataclasses import dataclass

from qkd_rl.core.types import KeyRequest
from qkd_rl.env.qkp import LinkQKPPool


@dataclass
class ServeResult:
    served_requests: list[KeyRequest]
    waiting_requests: list[KeyRequest]
    failed_requests: list[KeyRequest]
    served_keys: float
    waiting_keys: float
    failed_keys: float


@dataclass
class DemandEdgeStats:
    pending_amount: float
    pending_count: int
    min_deadline_left: int
    mean_deadline_left: float
    priority_sum: float
    arrival_history: dict[int, float]
    served_history: dict[int, float]
    failed_history: dict[int, float]


class RequestQueue:
    def __init__(self):
        self.pending: list[KeyRequest] = []

    def reset(self) -> None:
        self.pending = []

    def add_arrivals(self, requests: list[KeyRequest]) -> None:
        self.pending.extend(requests)

    def expire(self, t: int) -> list[KeyRequest]:
        expired = [req for req in self.pending if t > req.deadline_t]
        self.pending = [req for req in self.pending if t <= req.deadline_t]
        return expired

    def get_pending(self) -> list[KeyRequest]:
        return list(self.pending)

    def demand_by_node(self) -> dict[str, float]:
        demand: dict[str, float] = {}
        for req in self.pending:
            demand[req.src_gs] = demand.get(req.src_gs, 0.0) + req.amount
            demand[req.dst_gs] = demand.get(req.dst_gs, 0.0) + req.amount
        return demand

    def demand_by_pair(self) -> dict[tuple[str, str], float]:
        demand: dict[tuple[str, str], float] = {}
        for req in self.pending:
            key = tuple(sorted((req.src_gs, req.dst_gs)))
            demand[key] = demand.get(key, 0.0) + req.amount
        return demand

    def stats_by_pair(
        self,
        t: int,
        history: "RequestHistoryTracker",
        windows: list[int],
    ) -> dict[tuple[str, str], DemandEdgeStats]:
        grouped: dict[tuple[str, str], list[KeyRequest]] = {}
        for req in self.pending:
            key = tuple(sorted((req.src_gs, req.dst_gs)))
            grouped.setdefault(key, []).append(req)

        pairs = set(grouped) | history.recent_pairs(t, windows)
        stats: dict[tuple[str, str], DemandEdgeStats] = {}
        for pair in pairs:
            reqs = grouped.get(pair, [])
            deadline_left = [max(0, req.deadline_t - t) for req in reqs]
            stats[pair] = DemandEdgeStats(
                pending_amount=sum(req.amount for req in reqs),
                pending_count=len(reqs),
                min_deadline_left=min(deadline_left) if deadline_left else 0,
                mean_deadline_left=sum(deadline_left) / len(deadline_left) if deadline_left else 0.0,
                priority_sum=sum(req.amount * req.priority for req in reqs),
                arrival_history={window: history.sum(pair, "arrived", t, window) for window in windows},
                served_history={window: history.sum(pair, "served", t, window) for window in windows},
                failed_history={window: history.sum(pair, "failed", t, window) for window in windows},
            )
        return stats

    def serve(self, qkp: LinkQKPPool, routing: "RoutingPolicy", t: int) -> ServeResult:
        served: list[KeyRequest] = []
        waiting: list[KeyRequest] = []
        failed: list[KeyRequest] = []
        next_pending: list[KeyRequest] = []

        for req in sorted(self.pending, key=lambda item: item.deadline_t):
            if routing.consume_for_request(req, qkp, t):
                served.append(req)
            elif t >= req.deadline_t:
                failed.append(req)
            else:
                waiting.append(req)
                next_pending.append(req)

        self.pending = next_pending
        return ServeResult(
            served_requests=served,
            waiting_requests=waiting,
            failed_requests=failed,
            served_keys=sum(req.amount for req in served),
            waiting_keys=sum(req.amount for req in waiting),
            failed_keys=sum(req.amount for req in failed),
        )


class RequestGenerator:
    def __init__(self, gs_ids: list[str], config: dict, seed: int):
        self.gs_ids = gs_ids
        self.config = config
        self.rng = random.Random(seed)
        self.counter = 0

    def generate(self, t: int) -> list[KeyRequest]:
        if len(self.gs_ids) < 2:
            return []
        rate = float(self.config.get("arrival_rate", 0.0))
        if self.rng.random() > rate:
            return []
        src, dst = self.rng.sample(self.gs_ids, 2)
        self.counter += 1
        amount_mean = float(self.config.get("amount_mean", 100.0))
        amount = max(1.0, self.rng.expovariate(1.0 / amount_mean))
        deadline = t + int(self.config.get("deadline_steps", 12))
        return [KeyRequest(f"REQ_{self.counter:08d}", src, dst, amount, t, deadline)]


class RequestHistoryTracker:
    def __init__(self):
        self.events: list[tuple[int, tuple[str, str], str, float]] = []

    def reset(self) -> None:
        self.events = []

    def record_arrivals(self, requests: list[KeyRequest], t: int) -> None:
        self._record("arrived", requests, t)

    def record_served(self, requests: list[KeyRequest], t: int) -> None:
        self._record("served", requests, t)

    def record_failed(self, requests: list[KeyRequest], t: int) -> None:
        self._record("failed", requests, t)

    def recent_pairs(self, t: int, windows: list[int]) -> set[tuple[str, str]]:
        if not windows:
            return set()
        earliest = t - max(windows)
        return {pair for event_t, pair, _kind, _amount in self.events if earliest < event_t <= t}

    def sum(self, pair: tuple[str, str], kind: str, t: int, window: int) -> float:
        start_t = t - window
        return sum(
            amount
            for event_t, event_pair, event_kind, amount in self.events
            if event_pair == pair and event_kind == kind and start_t < event_t <= t
        )

    def _record(self, kind: str, requests: list[KeyRequest], t: int) -> None:
        for req in requests:
            pair = tuple(sorted((req.src_gs, req.dst_gs)))
            self.events.append((t, pair, kind, req.amount))


from qkd_rl.env.routing import RoutingPolicy  # noqa: E402
