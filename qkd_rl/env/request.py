from __future__ import annotations

import random
from bisect import bisect_right

import numpy as np
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
    mean_wait_time: float
    priority_sum: float
    wait_bucket_amounts: list[float]


def wait_bucket_edges(deadline_steps: int, bucket_count: int) -> list[int]:
    """Divide ``deadline_steps`` into ``bucket_count`` rounded buckets.

    The bucket width is ``round(deadline_steps / bucket_count)`` (min 1); the
    last boundary is clamped to the deadline, so requests whose age falls past
    the last full bucket still land in the final bucket.
    """
    if bucket_count <= 0:
        raise ValueError("wait_bucket_count must be positive.")
    width = max(1, int(deadline_steps / bucket_count + 0.5))
    edges = [min(deadline_steps, i * width) for i in range(bucket_count)]
    edges.append(deadline_steps)
    return edges


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
            remaining = max(0.0, req.amount - req.served_amount)
            demand[req.src_gs] = demand.get(req.src_gs, 0.0) + remaining
            demand[req.dst_gs] = demand.get(req.dst_gs, 0.0) + remaining
        return demand

    def demand_in_by_node(self) -> dict[str, float]:
        """Pending key demand where the node is the destination."""
        demand: dict[str, float] = {}
        for req in self.pending:
            demand[req.dst_gs] = demand.get(req.dst_gs, 0.0) + max(0.0, req.amount - req.served_amount)
        return demand

    def demand_out_by_node(self) -> dict[str, float]:
        """Pending key demand where the node is the source."""
        demand: dict[str, float] = {}
        for req in self.pending:
            demand[req.src_gs] = demand.get(req.src_gs, 0.0) + max(0.0, req.amount - req.served_amount)
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
            key = (req.src_gs, req.dst_gs) if req.src_gs <= req.dst_gs else (req.dst_gs, req.src_gs)
            demand[key] = demand.get(key, 0.0) + max(0.0, req.amount - req.served_amount)
        return demand

    def stats_by_pair(
        self,
        t: int,
        history: "RequestHistoryTracker" | None = None,
        windows: list[int] | None = None,
        wait_bucket_count: int = 10,
        deadline_steps: int = 960,
    ) -> dict[tuple[str, str], DemandEdgeStats]:
        grouped: dict[tuple[str, str], list[KeyRequest]] = {}
        for req in self.pending:
            key = (req.src_gs, req.dst_gs) if req.src_gs <= req.dst_gs else (req.dst_gs, req.src_gs)
            grouped.setdefault(key, []).append(req)

        pairs = set(grouped)
        if history is not None and windows:
            pairs |= history.recent_pairs(t, windows)
        bucket_edges = wait_bucket_edges(int(deadline_steps), int(wait_bucket_count))
        stats: dict[tuple[str, str], DemandEdgeStats] = {}
        for pair in pairs:
            reqs = grouped.get(pair, [])
            deadline_left = [max(0, req.deadline_t - t) for req in reqs]
            wait_time = [max(0, t - req.arrival_t) for req in reqs]
            bucket_amounts = [0.0] * (len(bucket_edges) - 1)
            for req in reqs:
                remaining = max(0.0, req.amount - req.served_amount)
                age = max(0, t - req.arrival_t)
                idx = min(len(bucket_amounts) - 1, max(0, bisect_right(bucket_edges, age) - 1))
                bucket_amounts[idx] += remaining
            stats[pair] = DemandEdgeStats(
                pending_amount=sum(max(0.0, req.amount - req.served_amount) for req in reqs),
                pending_count=len(reqs),
                min_deadline_left=min(deadline_left) if deadline_left else 0,
                mean_deadline_left=sum(deadline_left) / len(deadline_left) if deadline_left else 0.0,
                mean_wait_time=sum(wait_time) / len(wait_time) if wait_time else 0.0,
                priority_sum=sum(max(0.0, req.amount - req.served_amount) * req.priority for req in reqs),
                wait_bucket_amounts=bucket_amounts,
            )
        return stats

    def serve(self, qkp: LinkQKPPool, routing: "RoutingPolicy", t: int) -> ServeResult:
        """Serve pending requests with per-hop partial service.

        A request is served as much as the bottleneck hop of a usable path
        allows; ``served_keys`` is the amount actually consumed (completed
        plus partial). ``served_requests`` holds only requests completed this
        step; the remainder is re-queued with an updated ``served_amount``.
        Deadline-reached requests stay in the pending queue so that
        ``expire()`` reports them as expired in the same step.
        """
        from dataclasses import replace

        routing.prepare_serve(qkp)
        served: list[KeyRequest] = []
        waiting: list[KeyRequest] = []
        next_pending: list[KeyRequest] = []
        served_keys = 0.0

        def _priority(req: KeyRequest):
            remaining = max(0.0, req.amount - req.served_amount)
            # EDF: earliest deadline first, matching routing.serve_order and the
            # config validation (config.py rejects any other serve_order). With
            # the fixed-deadline generator (deadline = arrival + constant) this
            # ordering is identical to FIFO; it becomes a true EDF as soon as
            # deadlines vary per request. Arrival time only breaks deadline
            # ties, so newly-arrived urgent requests are not starved by old
            # requests whose deadlines are still far away.
            return (
                req.deadline_t,
                req.arrival_t,
                routing.hop_distance(req.src_gs, req.dst_gs),
                remaining,
            )

        for req in sorted(self.pending, key=_priority):
            served_now = routing.partial_consume_for_request(req, qkp, t)
            served_keys += served_now
            if served_now > 0:
                updated = replace(req, served_amount=req.served_amount + served_now)
                if updated.served_amount >= req.amount - 1.0e-9:
                    served.append(updated)
                    continue
                req = updated
            if t < req.deadline_t:
                waiting.append(req)
                next_pending.append(req)
            else:
                # Deadline reached but not fully served: keep in pending so
                # that expire() reports it as expired in the same step.
                next_pending.append(req)

        self.pending = next_pending
        return ServeResult(
            served_requests=served,
            waiting_requests=waiting,
            failed_requests=[],
            served_keys=served_keys,
            waiting_keys=sum(max(0.0, req.amount - req.served_amount) for req in waiting),
            failed_keys=0.0,
        )


class RequestGenerator:
    """Generate GS-GS key requests with realistic, learnable patterns.

    ``arrival_rate`` is the mean requests per slot (Poisson arrivals), so the
    offered load is ``arrival_rate * amount_mean`` key-bits per slot, calibrated
    against the actual generated key rate of the selected links.

    Optional realism patterns (stable across the whole run so the agent can
    learn them; both feed the node time features and GS-GS demand features):

    - ``hourly_weights`` (24 values): intra-day business profile. The arrival
      rate at slot ``t`` is ``arrival_rate * weight[hour] / mean(weights)``
      with ``hour = (t // steps_per_hour) % 24``, so ``arrival_rate`` stays the
      daily mean and peak/off-peak hours are amplified/reduced.
    - ``pair_hotness: zipf``: fixed per-pair weights from a Zipf distribution
      (``pair_zipf_s``, shuffled with ``pair_seed``), mimicking the long tail
      of real city-pair traffic: a few hot GS pairs carry most demand.
    """

    def __init__(self, gs_ids: list[str], config: dict, seed: int):
        self.gs_ids = gs_ids
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.counter = 0
        self._pairs: list[tuple[str, str]] = []
        self._pair_weights: np.ndarray | None = None
        if len(gs_ids) >= 2:
            from itertools import combinations

            self._pairs = list(combinations(sorted(gs_ids), 2))
            mode = self.config.get("pair_hotness", "uniform")
            if mode == "zipf":
                s = float(self.config.get("pair_zipf_s", 1.0))
                ranks = np.arange(1, len(self._pairs) + 1, dtype=float)
                weights = 1.0 / np.power(ranks, s)
                hot_rng = np.random.RandomState(int(self.config.get("pair_seed", 0)))
                hot_rng.shuffle(weights)  # assign hotness to random pairs
                self._pair_weights = weights / weights.sum()
            elif mode != "uniform":
                raise ValueError(f"Unknown pair_hotness: {mode!r}")

    def seed(self, seed: int) -> None:
        """Reseed the request stream (used for per-episode diversity)."""
        self.rng.seed(seed)
        self.counter = 0

    def _lambda_at(self, t: int) -> float:
        base = float(self.config.get("arrival_rate", 0.0))
        weights = self.config.get("hourly_weights")
        if not weights:
            return base
        steps_per_hour = int(self.config.get("steps_per_hour", 60))
        hour = (t // steps_per_hour) % 24
        w = float(weights[hour % len(weights)])
        mean_w = sum(float(x) for x in weights) / len(weights)
        return base * w / mean_w if mean_w > 0 else base

    def _sample_pair(self) -> tuple[str, str]:
        if self._pair_weights is not None and self._pairs:
            idx = int(self.rng.choice(len(self._pairs), p=self._pair_weights))
            return self._pairs[idx]
        src, dst = self.rng.choice(self.gs_ids, 2, replace=False)
        return str(src), str(dst)

    def generate(self, t: int) -> list[KeyRequest]:
        if len(self.gs_ids) < 2:
            return []
        n_requests = int(self.rng.poisson(self._lambda_at(t)))
        if n_requests <= 0:
            return []
        amount_mean = float(self.config.get("amount_mean", 100.0))
        # Optional hard cap on the request size (``amount_max``). The raw
        # exponential tail spawns requests of many times the mean that no
        # relay path can finish inside the deadline, inflating the failure
        # count for every policy; truncated-exponential rejection sampling
        # keeps the same body shape without the unbounded tail.
        amount_max = float(self.config.get("amount_max", 0.0) or 0.0)
        deadline = t + int(self.config.get("deadline_steps", 12))
        out: list[KeyRequest] = []
        for _ in range(n_requests):
            src, dst = self._sample_pair()
            self.counter += 1
            amount = float(self.rng.exponential(amount_mean))
            if amount_max > 0.0:
                while amount >= amount_max:
                    amount = float(self.rng.exponential(amount_mean))
            amount = max(1.0, amount)
            out.append(KeyRequest(f"REQ_{self.counter:08d}", src, dst, amount, t, deadline, self._sample_priority()))
        return out

    def _sample_priority(self) -> float:
        mode = self.config.get("priority_mode", "uniform")
        if mode == "uniform":
            return 1.0
        if mode == "random":
            return round(float(self.rng.uniform(1.0, 3.0)), 4)
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
        # Per-node index: (node_id, kind) -> times/prefix, kept in sync with
        # the pair index so sum_for_node is O(log n) instead of a scan over
        # every (pair, kind) key.
        self._node_times: dict[tuple[str, str], list[int]] = {}
        self._node_prefix: dict[tuple[tuple[str, str], str], list[float]] = {}

    def reset(self) -> None:
        self.events = []
        self._times = {}
        self._prefix = {}
        self._node_times = {}
        self._node_prefix = {}

    def record_arrivals(self, requests: list[KeyRequest], t: int) -> None:
        self._record("arrived", requests, t)

    def record_served(self, requests: list[KeyRequest], t: int) -> None:
        self._record("served", requests, t)

    def record_failed(self, requests: list[KeyRequest], t: int) -> None:
        # Only the unserved remainder counts as failed: the served part was
        # already delivered and counted via ``served``/``served_keys``.
        self._record("failed", requests, t, amount_of=lambda req: max(0.0, req.amount - req.served_amount))

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
        key = (node_id, kind)
        times = self._node_times.get(key)
        if not times:
            return 0.0
        left = bisect_right(times, t - window)
        right = bisect_right(times, t)
        return self._node_prefix[key][right] - self._node_prefix[key][left]

    def _record(self, kind: str, requests: list[KeyRequest], t: int, amount_of=None) -> None:
        for req in requests:
            amount = amount_of(req) if amount_of is not None else req.amount
            pair = tuple(sorted((req.src_gs, req.dst_gs)))
            self.events.append((t, pair, kind, amount))
            key = (pair, kind)
            times = self._times.setdefault(key, [])
            prefix = self._prefix.setdefault(key, [0.0])
            times.append(t)
            prefix.append(prefix[-1] + amount)
            for node_id in (req.src_gs, req.dst_gs):
                nkey = (node_id, kind)
                ntimes = self._node_times.setdefault(nkey, [])
                nprefix = self._node_prefix.setdefault(nkey, [0.0])
                ntimes.append(t)
                nprefix.append(nprefix[-1] + amount)


from qkd_rl.env.routing import RoutingPolicy  # noqa: E402
