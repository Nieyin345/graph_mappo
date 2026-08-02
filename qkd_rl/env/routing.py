from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from qkd_rl.core.types import Edge, KeyRequest
from qkd_rl.env.qkp import LinkQKPPool


@dataclass
class AllocationResult:
    added_keys: float
    overflow_keys: float


class RoutingPolicy:
    def __init__(self, edges: list[Edge], config: dict):
        self.edges = edges
        self.config = config
        self.adj: dict[str, list[tuple[str, str]]] = {}
        for edge in edges:
            self.adj.setdefault(edge.src, []).append((edge.dst, edge.edge_id))
            self.adj.setdefault(edge.dst, []).append((edge.src, edge.edge_id))

    def allocate_generated_keys(
        self,
        activated_edges: list[str],
        generated_keys: dict[str, float],
        qkp: LinkQKPPool,
        t: int,
    ) -> AllocationResult:
        added = 0.0
        overflow = 0.0
        for edge_id in activated_edges:
            amount = generated_keys.get(edge_id, 0.0)
            actual = qkp.add_keys(edge_id, amount, t)
            added += actual
            overflow += max(0.0, amount - actual)
        return AllocationResult(added_keys=added, overflow_keys=overflow)

    def find_request_path(self, request: KeyRequest, qkp: LinkQKPPool | None = None) -> list[str] | None:
        queue: deque[tuple[str, list[str]]] = deque([(request.src_gs, [])])
        visited = {request.src_gs}
        while queue:
            node_id, path = queue.popleft()
            if node_id == request.dst_gs:
                return path
            for neighbor, edge_id in self.adj.get(node_id, []):
                if neighbor in visited:
                    continue
                if qkp is not None and qkp.get_level(edge_id) < request.amount:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path + [edge_id]))
        return None

    def can_serve(self, request: KeyRequest, qkp: LinkQKPPool, t: int) -> bool:
        path = self.find_request_path(request, qkp=qkp)
        return path is not None and qkp.can_consume_path(path, request.amount)

    def consume_for_request(self, request: KeyRequest, qkp: LinkQKPPool, t: int) -> bool:
        path = self.find_request_path(request, qkp=qkp)
        if path is None:
            return False
        return qkp.consume_path(path, request.amount)

