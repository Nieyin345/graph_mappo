from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from qkd_rl.core.types import Edge, KeyRequest
from qkd_rl.env.qkp import LinkQKPPool


@dataclass
class AllocationResult:
    added_keys: float
    overflow_keys: float
    added_by_edge: dict[str, float] = field(default_factory=dict)


class RoutingPolicy:
    DISCONNECTED = 10**6

    def __init__(self, edges: list[Edge], config: dict):
        self.edges = edges
        self.config = config
        self.adj: dict[str, list[tuple[str, str]]] = {}
        for edge in edges:
            self.adj.setdefault(edge.src, []).append((edge.dst, edge.edge_id))
            self.adj.setdefault(edge.dst, []).append((edge.src, edge.edge_id))
        self._hop: dict[tuple[str, str], int] = {}
        self._next_hop: dict[tuple[str, str], tuple[str, str]] = {}
        self._precompute_hop_distances()

    def _precompute_hop_distances(self) -> None:
        """Static-topology all-pairs shortest hop counts and next-hop tables.

        Hop count is a topology property (independent of current key stock),
        used as a service-priority key. ``_next_hop[(u, v)]`` is the first edge
        ``(neighbor, edge_id)`` on a shortest ``u -> v`` path, so a full path
        can be reconstructed in O(hops) without a per-step BFS. The cached
        path is only usable while every hop keeps a positive key stock; if a
        hop is empty the caller falls back to a live BFS.
        """
        for src in self.adj:
            dist = {src: 0}
            first_step: dict[str, tuple[str, str]] = {}
            queue = deque([src])
            while queue:
                node = queue.popleft()
                for neighbor, edge_id in self.adj[node]:
                    if neighbor in dist:
                        continue
                    dist[neighbor] = dist[node] + 1
                    first_step[neighbor] = (neighbor, edge_id) if node == src else first_step[node]
                    queue.append(neighbor)
            for dst, d in dist.items():
                self._hop[(src, dst)] = d
                if dst != src:
                    self._next_hop[(src, dst)] = first_step[dst]

    def hop_distance(self, src: str, dst: str) -> int:
        """Shortest hop count between two nodes (large sentinel if disconnected)."""
        return self._hop.get((src, dst), self.DISCONNECTED)

    def shortest_path(self, src: str, dst: str) -> list[str] | None:
        """Cached shortest topological path as an edge list (None if disconnected)."""
        if src == dst:
            return []
        path: list[str] = []
        node = src
        while node != dst:
            nxt = self._next_hop.get((node, dst))
            if nxt is None:
                return None
            node, edge_id = nxt
            path.append(edge_id)
        return path

    def allocate_generated_keys(
        self,
        activated_edges: list[str],
        generated_keys: dict[str, float],
        qkp: LinkQKPPool,
        t: int,
    ) -> AllocationResult:
        added = 0.0
        overflow = 0.0
        added_by_edge: dict[str, float] = {}
        for edge_id in activated_edges:
            amount = generated_keys.get(edge_id, 0.0)
            actual = qkp.add_keys(edge_id, amount, t)
            added += actual
            overflow += max(0.0, amount - actual)
            added_by_edge[edge_id] = actual
        return AllocationResult(added_keys=added, overflow_keys=overflow, added_by_edge=added_by_edge)

    def find_request_path(
        self,
        request: KeyRequest,
        qkp: LinkQKPPool | None = None,
        required_level: float | None = None,
    ) -> list[str] | None:
        """BFS shortest path from ``request.src_gs`` to ``request.dst_gs``.

        ``required_level`` is the per-hop key level a link must hold to be
        usable: pass ``request.amount`` for all-or-nothing serving, or ``0.0``
        for partial serving (any link with a positive key stock is usable).
        Defaults to ``request.amount`` for backward compatibility.
        """
        min_level = max(required_level if required_level is not None else request.amount, 1.0e-9)
        # Parent-pointer BFS: reconstructs the path only once at the end
        # instead of copying a growing path list per explored edge.
        parent: dict[str, tuple[str | None, str | None]] = {request.src_gs: (None, None)}
        queue: deque[str] = deque([request.src_gs])
        while queue:
            node_id = queue.popleft()
            if node_id == request.dst_gs:
                path: list[str] = []
                cur = node_id
                while parent[cur][0] is not None:
                    path.append(parent[cur][1])
                    cur = parent[cur][0]
                path.reverse()
                return path
            for neighbor, edge_id in self.adj.get(node_id, []):
                if neighbor in parent:
                    continue
                if qkp is not None and qkp.get_level(edge_id) < min_level:
                    continue
                parent[neighbor] = (node_id, edge_id)
                queue.append(neighbor)
        return None

    def can_serve(self, request: KeyRequest, qkp: LinkQKPPool, t: int) -> bool:
        path = self.find_request_path(request, qkp=qkp, required_level=request.amount)
        return path is not None and qkp.can_consume_path(path, request.amount)

    def consume_for_request(self, request: KeyRequest, qkp: LinkQKPPool, t: int) -> bool:
        path = self.find_request_path(request, qkp=qkp, required_level=request.amount)
        if path is None:
            return False
        return qkp.consume_path(path, request.amount)

    def prepare_serve(self, qkp: LinkQKPPool) -> None:
        """Precompute the positive-key subgraph for the upcoming serve phase.

        Key levels only decrease while requests are being served (generation
        happened before), so a ``(src, dst)`` pair that is disconnected in the
        positive subgraph at serve start can never become connected during the
        phase. The component map is a sound fast-reject for partial serving,
        and the precomputed adjacency lets the BFS iterate only positive edges
        without a per-edge ``get_level`` dict lookup.
        """
        positive = qkp.positive
        pos_adj: dict[str, list[tuple[str, str]]] = {}
        component: dict[str, int] = {}
        comp_id = 0
        for node_id in self.adj:
            if node_id in component:
                continue
            queue: deque[str] = deque([node_id])
            component[node_id] = comp_id
            while queue:
                node = queue.popleft()
                for neighbor, edge_id in self.adj[node]:
                    if edge_id not in positive:
                        continue
                    pos_adj.setdefault(node, []).append((neighbor, edge_id))
                    if neighbor not in component:
                        component[neighbor] = comp_id
                        queue.append(neighbor)
            comp_id += 1
        self._pos_adj = pos_adj
        self._pos_component = component

    def _find_positive_path(self, request: KeyRequest, qkp: LinkQKPPool) -> list[str] | None:
        """BFS over links with a positive key stock (partial-service search).

        Uses the serve-phase adjacency (already filtered to positive edges) and
        re-checks the live positive set, so edges drained earlier in the same
        serve phase are skipped without traversing the full static topology.
        """
        adj = getattr(self, "_pos_adj", None) or self.adj
        positive = qkp.positive
        parent: dict[str, tuple[str | None, str | None]] = {request.src_gs: (None, None)}
        queue: deque[str] = deque([request.src_gs])
        while queue:
            node_id = queue.popleft()
            if node_id == request.dst_gs:
                path: list[str] = []
                cur = node_id
                while parent[cur][0] is not None:
                    path.append(parent[cur][1])
                    cur = parent[cur][0]
                path.reverse()
                return path
            for neighbor, edge_id in adj.get(node_id, ()):
                if neighbor in parent:
                    continue
                if edge_id not in positive:
                    continue
                parent[neighbor] = (node_id, edge_id)
                queue.append(neighbor)
        return None

    def _usable_cached_path(self, request: KeyRequest, qkp: LinkQKPPool) -> list[str] | None:
        """The cached shortest path if every hop still holds positive keys."""
        path = self.shortest_path(request.src_gs, request.dst_gs)
        if path is None:
            return None
        if all(edge_id in qkp.positive for edge_id in path):
            return path
        return None

    def partial_consume_for_request(self, request: KeyRequest, qkp: LinkQKPPool, t: int) -> float:
        """Serve as much of ``request`` as the bottleneck hop allows.

        Finds any path whose hops all hold a positive key stock, then consumes
        ``min(hop levels, remaining demand)`` from every hop (per-hop key
        relay semantics). Returns the amount actually consumed this step; the
        caller accumulates it on ``request.served_amount`` and re-queues the
        remainder.
        """
        path = self._usable_cached_path(request, qkp)
        if path is None:
            component = getattr(self, "_pos_component", None)
            if component is not None and component.get(request.src_gs, -1) != component.get(request.dst_gs, -1):
                return 0.0
            path = self._find_positive_path(request, qkp)
        if path is None:
            return 0.0
        remaining = max(0.0, request.amount - request.served_amount)
        if remaining <= 1.0e-9:
            return 0.0
        hop_levels = [qkp.get_level(edge_id) for edge_id in path]
        serve_now = min(hop_levels + [remaining])
        if serve_now <= 1.0e-9:
            return 0.0
        qkp.consume_path(path, serve_now)
        return serve_now

