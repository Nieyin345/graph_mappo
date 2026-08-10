from __future__ import annotations

from qkd_rl.core.types import Edge


class NodeActionSpace:
    IDLE = "idle"

    def __init__(self, node_ids: list[str], edges: list[Edge], include_idle: bool = True):
        self.node_ids = node_ids
        self.edges = edges
        self.include_idle = bool(include_idle)
        self.edge_by_pair: dict[tuple[str, str], str] = {}
        self.neighbors: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        for edge in edges:
            key = tuple(sorted((edge.src, edge.dst)))
            self.edge_by_pair[key] = edge.edge_id
            self.neighbors.setdefault(edge.src, []).append(edge.dst)
            self.neighbors.setdefault(edge.dst, []).append(edge.src)

    def candidates_for_node(self, node_id: str) -> list[str]:
        candidates = [] if not self.include_idle else [self.IDLE]
        return candidates + sorted(self.neighbors.get(node_id, []))

    def action_to_edge(self, node_id: str, action: str) -> str | None:
        if action == self.IDLE:
            return None
        return self.edge_by_pair.get(tuple(sorted((node_id, action))))

