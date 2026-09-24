"""GS-pair aggregated Path-Greedy heuristic (aligned with the env's serve BFS).

Core rules (matches the user's intent):
  1. SHORTEST is paramount -- found via a BFS (hop count), never a longer path
     even if the longer one is fully stockable.
  2. Among all *equal-length* shortest paths, prefer the one that reuses
     existing key stock (is_free hops need no activation) and avoids slow links.
  3. Demand is aggregated by ground-station pair, weighted by queue age; pairs
     are satisfied from most- to least-urgent.

For each pair we build a graph from the union of:
    * currently available physical edges (may be activated this slot), and
    * every edge already holding a positive key stock (a "free relay": it is
      already connectable in a future serve-phase positive-subgraph BFS, so it
      costs nothing to traverse this slot).

Then:
  - Stage 1: BFS over that graph to fix the shortest hop count between the two
    GS nodes. This guarantees criterion (1).
  - Stage 2: among all shortest-hop paths, pick the one maximizing
    ``(w_stock if is_free else 0) + w_rate * rate_norm`` per hop -- i.e. the
    path that reuses the most stock while staying away from low-rate links.
    The selected path's non-stocked edges are the "gap edges" we activate, which
    turns the whole path positive-stocked so the env's serve BFS routes over it
    next slot (the exact reverse of the env's positive-subgraph serve search).

Port budget follows the dual-port rule: each node has one Tx and one Rx, so a
node may be a transmitter for one path and a receiver for another while
``u->v`` never coexists with ``v->u`` (free relays do not consume ports).
"""
from __future__ import annotations

from collections import deque

from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.graph_builder import GraphObservation


def edge_endpoints(edge_id: str) -> tuple[str, str]:
    body = edge_id[2:] if edge_id.startswith("E_") else edge_id
    if "__" in body:
        return tuple(body.split("__", 1))
    return (edge_id, edge_id)


class PathGreedyPairPolicy:
    STOCK_TOL = 1.0e-9

    def __init__(
        self,
        wait_urgency_tau_ratio: float = 0.8,
        w_stock: float = 0.5,
        w_rate: float = 0.5,
    ):
        self.wait_urgency_tau_ratio = max(0.0, float(wait_urgency_tau_ratio))
        self.w_stock = float(w_stock)
        self.w_rate = float(w_rate)

    # -- GS-pair aggregation & urgency weighting ------------------------------
    def _pair_demand(self, obs: GraphObservation) -> dict[tuple[str, str], float]:
        """Aggregate pending requests by GS pair, weighted by queue age."""
        t = obs.state.t
        pair_weight: dict[tuple[str, str], float] = {}
        for req in obs.state.pending_requests:
            remaining = max(0.0, req.amount - req.served_amount)
            if remaining <= self.STOCK_TOL:
                continue
            age = max(0, int(t) - int(req.arrival_t))
            deadline_len = max(1, int(req.deadline_t) - int(req.arrival_t))
            if self.wait_urgency_tau_ratio > 0.0:
                tau = max(1.0, float(deadline_len) * float(self.wait_urgency_tau_ratio))
                urgency = _exp(age / tau)
            else:
                urgency = 1.0 + age / float(deadline_len)
            pair = tuple(sorted((req.src_gs, req.dst_gs)))
            pair_weight[pair] = pair_weight.get(pair, 0.0) + remaining * urgency
        return {p: w for p, w in pair_weight.items() if w > self.STOCK_TOL}

    # -- graph & rate normalization ------------------------------------------
    def _build_graph(
        self, obs: GraphObservation
    ) -> tuple[
        dict[str, list[tuple[str, str, bool]]],
        dict[tuple[str, str], str],
        dict[str, float],
    ]:
        """Adjacency over ``available U stocked`` edges plus per-edge rate score.

        Returns ``(adj, pair_to_edge, edge_rate)`` where each adjacency entry is
        ``(neighbor, edge_id, is_free)`` and ``is_free`` marks an already-stocked
        edge (traversable without activation). ``edge_rate`` maps edge_id to a
        normalized rate in [0, 1] for the shortest-path scoring (0 when unknown).
        """
        snapshot = getattr(obs.state, "qkp_snapshot", None) or {}
        stocked: set[str] = {
            eid for eid, lvl in snapshot.items() if lvl > self.STOCK_TOL
        }

        raw_rates: dict[str, float] = {}
        for eid in obs.generation_edge_ids:
            w = obs.state.edge_windows.get(eid)
            if w is not None:
                raw_rates[eid] = float(w.rates[0]) if w.rates else 0.0
        max_rate = max(raw_rates.values(), default=1.0) or 1.0
        edge_rate: dict[str, float] = {
            eid: (v / max_rate if max_rate > 0 else 0.0)
            for eid, v in raw_rates.items()
        }

        nodes = set(obs.node_ids)
        adj: dict[str, list[tuple[str, str, bool]]] = {n: [] for n in obs.node_ids}
        pair_to_edge: dict[tuple[str, str], str] = {}
        seen: set[tuple[str, str]] = set()

        def _add(src: str, dst: str, eid: str, is_free: bool) -> None:
            if src == dst or src not in nodes or dst not in nodes:
                return
            pair = tuple(sorted((src, dst)))
            if pair in seen:
                return
            seen.add(pair)
            pair_to_edge[pair] = eid
            adj[src].append((dst, eid, is_free))
            adj[dst].append((src, eid, is_free))

        for eid in obs.generation_edge_ids:
            u, v = edge_endpoints(eid)
            _add(u, v, eid, eid in stocked)
        for eid in stocked:  # stocked-but-currently-unavailable free relays
            u, v = edge_endpoints(eid)
            _add(u, v, eid, True)
        return adj, pair_to_edge, edge_rate

    # -- Stage 1: BFS shortest hop count ---------------------------------------
    @staticmethod
    def _shortest_dist(
        adj: dict[str, list[tuple[str, str, bool]]], src: str
    ) -> dict[str, int]:
        """Unweighted BFS hop distances from ``src`` (shortest is absolute)."""
        dist = {src: 0}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            for nxt, _eid, _isfree in adj.get(node, ()):
                if nxt in dist:
                    continue
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
        return dist

    # -- Stage 2: best shortest path (stock + rate) ----------------------------
    def _best_shortest_path(
        self,
        obs: GraphObservation,
        adj: dict[str, list[tuple[str, str, bool]]],
        dist: dict[str, int],
        edge_rate: dict[str, float],
        src: str,
        dst: str,
        tx_busy: set[str],
        rx_busy: set[str],
        chosen_pairs: set[tuple[str, str]],
    ) -> list[tuple[str, str, bool]] | None:
        """Greedily walk a shortest path maximizing ``stock + rate`` per hop.

        Starts from ``dst`` and walks backwards, at each step keeping only
        transitions ``prev -> cur`` with ``dist[prev] == dist[cur] - 1`` (so the
        path length is fixed to the BFS minimum). Among those, and only those
        whose gap-edge ports are free under the dual-port budget, we pick the
        highest per-hop ``(w_stock if is_free else 0.0) + w_rate * edge_rate``.
        """
        d = dist[dst]
        # path_cand[level] -> best (gain, prev, edge_id, is_free) reaching a node
        best: list[dict[str, tuple[float, str, str, bool]]] = [
            {} for _ in range(d + 1)
        ]
        best[0][src] = (0.0, "", "", False)
        for level in range(d):
            for cur, seen in best[level].items():  # nodes at this level
                prev_gain = seen[0]
                for nxt, eid, is_free in adj.get(cur, ()):
                    if dist.get(nxt, -1) != level + 1:
                        continue  # not a shortest-step edge
                    pair = tuple(sorted((cur, nxt)))
                    if not is_free:
                        if cur in tx_busy or nxt in rx_busy:
                            continue
                        if pair in chosen_pairs:
                            continue
                    gain = prev_gain + (
                        self.w_stock if is_free else 0.0
                    ) + self.w_rate * edge_rate.get(eid, 0.0)
                    prev = best[level + 1].get(nxt)
                    if prev is None or gain > prev[0]:
                        best[level + 1][nxt] = (gain, cur, eid, is_free)
        if dst not in best[d]:
            return None
        # reconstruct hops in forward order
        hops: list[tuple[str, str, bool]] = []
        cur = dst
        for level in range(d, 0, -1):
            _gain, prev, eid, is_free = best[level][cur]
            hops.append((eid, is_free))
            cur = prev
        hops.reverse()
        return hops

    # -- main -----------------------------------------------------------------
    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        pair_demand = self._pair_demand(obs)
        adj, pair_to_edge, edge_rate = self._build_graph(obs)

        tx_busy: set[str] = set()
        rx_busy: set[str] = set()
        chosen_pairs: set[tuple[str, str]] = set()
        dual: dict[str, list[str]] = {
            node: [NodeActionSpace.IDLE, NodeActionSpace.IDLE] for node in obs.node_ids
        }

        for pair, _w in sorted(pair_demand.items(), key=lambda kv: -kv[1]):
            src, dst = pair
            if src == dst or src not in adj or dst not in adj:
                continue
            if src in tx_busy or dst in rx_busy:
                continue
            dist = self._shortest_dist(adj, src)
            if dst not in dist:
                continue
            hops = self._best_shortest_path(
                obs, adj, dist, edge_rate, src, dst, tx_busy, rx_busy, chosen_pairs
            )
            if hops is None:
                continue
            # activate only the gap edges on the chosen path
            cur = src
            for eid, is_free in hops:
                if is_free:
                    a, b = edge_endpoints(eid)
                    cur = b if a == cur else a
                    continue
                a, b = edge_endpoints(eid)
                if a != cur:
                    a, b = b, a  # orient along path direction
                if dual[a][0] != NodeActionSpace.IDLE or dual[b][1] != NodeActionSpace.IDLE:
                    break  # port conflict (guarded by tx_busy/rx_busy, defensive)
                dual[a][0] = b
                dual[b][1] = a
                chosen_pairs.add(tuple(sorted((a, b))))
                tx_busy.add(a)
                rx_busy.add(b)
                cur = b

        actions = {n: tuple(dual[n]) for n in obs.node_ids}
        return actions, {}


def _exp(x: float) -> float:
    import math

    return math.exp(x)


def path_greedy_pair_actions(obs: GraphObservation) -> dict[str, tuple]:
    pol = PathGreedyPairPolicy()
    actions, _ = pol.act(obs)
    return actions