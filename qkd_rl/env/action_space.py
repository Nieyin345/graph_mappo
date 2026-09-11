from __future__ import annotations

from qkd_rl.core.types import Edge


class NodeActionSpace:
    """Static per-node action space under the dual-port directional model.

    Each node has a fixed transmitter (Tx) and receiver (Rx). A node's action is
    a ``(tx_target, rx_source)`` tuple:

    - ``tx_target``: the neighbor whose **Rx** this node's Tx points at, yielding
      the directed arc ``me -> tx_target`` (or ``IDLE`` for "no transmit");
    - ``rx_source``: the neighbor whose **Tx** points at this node's **Rx**,
      yielding the directed arc ``rx_source -> me`` (or ``IDLE``).

    The key-pool still keys on the **undirected pair** (``u, v``), so key
    generation / routing / QKP are unchanged; the direction only decides which
    physical ports are used this slot. Constraints (enforced by the resolver /
    policy / MILP, not here):
    ``Tx-out <= 1``, ``Rx-in <= 1``, and ``u->v`` with ``v->u`` never coexist.
    """

    IDLE = "idle"

    def __init__(self, node_ids: list[str], edges: list[Edge], include_idle: bool = True):
        self.node_ids = node_ids
        self.edges = edges
        self.include_idle = bool(include_idle)
        self.edge_by_pair: dict[tuple[str, str], str] = {}
        self.neighbors: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        self._node_set = set(node_ids)
        for edge in edges:
            key = tuple(sorted((edge.src, edge.dst)))
            self.edge_by_pair[key] = edge.edge_id
            self.neighbors.setdefault(edge.src, []).append(edge.dst)
            self.neighbors.setdefault(edge.dst, []).append(edge.src)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def pair_key(u: str, v: str) -> tuple[str, str]:
        """Canonical undirected pair key shared by ``u,v`` and ``v,u``."""
        return tuple(sorted((u, v)))

    def arc_to_edge(self, src: str, dst: str) -> str | None:
        """Undirected pair edge id for the directed arc ``src -> dst``."""
        if src == dst:
            return None
        return self.edge_by_pair.get(self.pair_key(src, dst))

    def arcs_to_edges(self, arcs: list[tuple[str, str]]) -> list[str]:
        """Deduplicate a list of directed arcs down to undirected pair edge ids.

        At most one entry per pair (对端不同), preserving first-seen order.
        """
        seen: set[str] = set()
        out: list[str] = []
        for u, v in arcs:
            eid = self.edge_by_pair.get(self.pair_key(u, v))
            if eid is not None and eid not in seen:
                seen.add(eid)
                out.append(eid)
        return out

    def is_neighbor(self, u: str, v: str) -> bool:
        return u != v and v in self.neighbors.get(u, ())

    # ------------------------------------------------------------------ actions
    def _with_idle(self, values: list[str]) -> list[str]:
        return ([self.IDLE] if self.include_idle else []) + values

    def tx_candidates_for_node(self, node_id: str) -> list[str]:
        """Nodes whose Rx this node's Tx may point at (neighbours + idle)."""
        return self._with_idle(sorted(self.neighbors.get(node_id, [])))

    def rx_candidates_for_node(self, node_id: str) -> list[str]:
        """Nodes whose Tx may point at this node's Rx (neighbours + idle)."""
        return self._with_idle(sorted(self.neighbors.get(node_id, [])))

    def candidates_for_node(self, node_id: str) -> list[str]:
        """Legacy flat view (single categorical), kept for introspection/tests."""
        return self._with_idle(sorted(self.neighbors.get(node_id, [])))

    def action_to_edge(self, node_id: str, action: str) -> str | None:
        """Legacy undirected-pair edge id for a (node, neighbour) proposal."""
        if action == self.IDLE:
            return None
        return self.arc_to_edge(node_id, action)