from __future__ import annotations

import random
from bisect import bisect_right
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
        """Drop requests whose deadline has passed and return them.

        ``serve()`` deliberately leaves deadline-reached requests in the
        pending queue; this method is the single place that reports them, so
        the expired branch is live instead of dead code.
        """
        expired = [req for req in self.pending if t >= req.deadline_t]
        self.pending = [req for req in self.pending if t < req.deadline_t]
        return expired

    def get_pending(self) -> list[KeyRequest]:
        return list(self.pending)

    def demand_by_node(self) -> dict[str, float]:
        demand: dict[str, float] = {}
        for req in self.pending:
            demand[req.src_gs] = demand.get(req.src_gs, 0.0) + req.amount
            demand[req.dst_gs] = demand.get(req.dst_gs, 0.0) + req.amount
        return demand

    def demand_in_by_node(self) -> dict[str, float]:
        """Pending key demand where the node is the destination."""
        demand: dict[str, float] = {}
        for req in self.pending:
            demand[req.dst_gs] = demand.get(req.dst_gs, 0.0) + req.amount
        return demand

    def demand_out_by_node(self) -> dict[str, float]:
        """Pending key demand where the node is the source."""
        demand: dict[str, float] = {}
        for req in self.pending:
            demand[req.src_gs] = demand.get(req.src_gs, 0.0) + req.amount
        return demand

    def pending_count_by_node(self) -> dict[str, int]:
        count: dict[str, int] = {}
        for req in self.pending:
            count[req.src_gs] = count.get(req.src_gs, 0) + 1
            count[req.dst_gs] = count.get(req.dst_gs, 0) + 1
        return count

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
        next_pending: list[KeyRequest] = []

        for req in sorted(self.pending, key=lambda item: item.deadline_t):
            if routing.consume_for_request(req, qkp, t):
                served.append(req)
            elif t < req.deadline_t:
                waiting.append(req)
                next_pending.append(req)
            else:
                # Deadline reached but not servable: keep in pending so that
                # expire() reports it as expired in the same step.
                next_pending.append(req)

        self.pending = next_pending
        return ServeResult(
            served_requests=served,
            waiting_requests=waiting,
            failed_requests=[],
            served_keys=sum(req.amount for req in served),
            waiting_keys=sum(req.amount for req in waiting),
            failed_keys=0.0,
        )


class RequestGenerator:
    def __init__(self, gs_ids: list[str], config: dict, seed: int):
        self.gs_ids = gs_ids
        self.config = config
        self.rng = random.Random(seed)
        self.counter = 0

    def seed(self, seed: int) -> None:
        """Reseed the request stream (used for per-episode diversity)."""
        self.rng.seed(seed)
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
        return [KeyRequest(f"REQ_{self.counter:08d}", src, dst, amount, t, deadline, self._sample_priority())]

    def _sample_priority(self) -> float:
        mode = self.config.get("priority_mode", "uniform")
        if mode == "uniform":
            return 1.0
        if mode == "random":
            return round(self.rng.uniform(1.0, 3.0), 4)
        raise ValueError(f"Unknown priority_mode: {mode!r}")


class RequestHistoryTracker:
    """Rolling request-event history.

    Events are kept both as a flat list (for debugging) and in an incremental
    per-(pair, kind) index with prefix sums, so window queries are O(log n)
    instead of a full table scan per pair.
    """

    def __init__(self):
        self.events: list[tuple[int, tuple[str, str], str, float]] = []
        self._times: dict[tuple[tuple[str, str], str], list[int]] = {}
        self._prefix: dict[tuple[tuple[str, str], str], list[float]] = {}

    def reset(self) -> None:
        self.events = []
        self._times = {}
        self._prefix = {}

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
        pairs: set[tuple[str, str]] = set()
        for (pair, _kind), times in self._times.items():
            if times and earliest < times[-1] <= t:
                pairs.add(pair)
        return pairs

    def sum(self, pair: tuple[str, str], kind: str, t: int, window: int) -> float:
        key = (pair, kind)
        times = self._times.get(key)
        if not times:
            return 0.0
        left = bisect_right(times, t - window)
        right = bisect_right(times, t)
        prefix = self._prefix[key]
        return prefix[right] - prefix[left]

    def sum_for_node(self, node_id: str, kind: str, t: int, window: int) -> float:
        """Sum event amounts in the window where ``node_id`` is an endpoint."""
        total = 0.0
        start_t = t - window
        for (pair, event_kind), times in self._times.items():
            if event_kind != kind or node_id not in pair:
                continue
            left = bisect_right(times, start_t)
            right = bisect_right(times, t)
            total += self._prefix[(pair, event_kind)][right] - self._prefix[(pair, event_kind)][left]
        return total

    def _record(self, kind: str, requests: list[KeyRequest], t: int) -> None:
        for req in requests:
            pair = tuple(sorted((req.src_gs, req.dst_gs)))
            self.events.append((t, pair, kind, req.amount))
            key = (pair, kind)
            times = self._times.setdefault(key, [])
            prefix = self._prefix.setdefault(key, [0.0])
            times.append(t)
            prefix.append(prefix[-1] + req.amount)


from qkd_rl.env.routing import RoutingPolicy  # noqa: E402
