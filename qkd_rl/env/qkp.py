from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from qkd_rl.core.types import Edge


@dataclass
class LinkQKPState:
    edge_id: str
    level: float
    capacity: float


class LinkQKPCapacityResolver:
    def __init__(self, config: dict):
        self.config = config["capacity"]

    def capacity_for_edge(self, edge: Edge) -> float:
        per_edge = self.config.get("per_edge", {})
        if edge.edge_id in per_edge:
            return float(per_edge[edge.edge_id])
        per_type = self.config.get("per_link_type", {})
        if edge.link_type.value in per_type:
            return float(per_type[edge.link_type.value])
        return float(self.config["default"])


class LinkQKPPool:
    def __init__(self, edges: list[Edge], config: dict):
        self.edges = {edge.edge_id: edge for edge in edges}
        self.config = config
        self.capacity_resolver = LinkQKPCapacityResolver(config)
        self.levels: dict[str, float] = {}
        self.batches: dict[str, list[tuple[float, int]]] = defaultdict(list)
        self.capacities = {
            edge_id: self.capacity_resolver.capacity_for_edge(edge) for edge_id, edge in self.edges.items()
        }

    def reset(self) -> None:
        initial = float(self.config.get("initial_level", 0.0))
        self.levels = {edge_id: min(initial, capacity) for edge_id, capacity in self.capacities.items()}
        self.batches = defaultdict(list)
        for edge_id, level in self.levels.items():
            if level > 0:
                self.batches[edge_id].append((level, 0))

    def add_keys(self, edge_id: str, amount: float, t: int) -> float:
        amount = max(0.0, amount)
        addable = min(amount, self.get_capacity_left(edge_id))
        if addable > 0:
            self.levels[edge_id] += addable
            self.batches[edge_id].append((addable, t))
        return addable

    def consume_keys(self, edge_id: str, amount: float) -> float:
        amount = max(0.0, amount)
        consumed = min(amount, self.levels[edge_id])
        remaining = consumed
        new_batches: list[tuple[float, int]] = []
        for batch_amount, batch_t in self.batches[edge_id]:
            if remaining <= 0:
                new_batches.append((batch_amount, batch_t))
                continue
            take = min(batch_amount, remaining)
            left = batch_amount - take
            remaining -= take
            if left > 0:
                new_batches.append((left, batch_t))
        self.batches[edge_id] = new_batches
        self.levels[edge_id] -= consumed
        return consumed

    def can_consume_path(self, edge_ids: list[str], amount: float) -> bool:
        return all(self.get_level(edge_id) >= amount for edge_id in edge_ids)

    def consume_path(self, edge_ids: list[str], amount: float) -> bool:
        if not self.can_consume_path(edge_ids, amount):
            return False
        for edge_id in edge_ids:
            self.consume_keys(edge_id, amount)
        return True

    def get_level(self, edge_id: str) -> float:
        return self.levels.get(edge_id, 0.0)

    def get_capacity(self, edge_id: str) -> float:
        return self.capacities[edge_id]

    def get_capacity_left(self, edge_id: str) -> float:
        return self.get_capacity(edge_id) - self.get_level(edge_id)

    def expire(self, t: int) -> float:
        ttl = self.config.get("key_ttl_steps")
        if ttl is None:
            return 0.0
        ttl = int(ttl)
        expired_total = 0.0
        for edge_id, batches in list(self.batches.items()):
            kept: list[tuple[float, int]] = []
            expired = 0.0
            for amount, batch_t in batches:
                if t - batch_t >= ttl:
                    expired += amount
                else:
                    kept.append((amount, batch_t))
            if expired:
                self.levels[edge_id] = max(0.0, self.levels[edge_id] - expired)
                expired_total += expired
            self.batches[edge_id] = kept
        return expired_total

    def snapshot(self) -> dict[str, float]:
        return dict(self.levels)

