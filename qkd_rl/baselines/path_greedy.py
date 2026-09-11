"""Path-Greedy heuristic: activate links at the granularity of *end-to-end
paths*, not individual edges.

Rationale
---------
V3 (and any edge-scoring policy) picks edges independently; those edges often
fail to form a complete src->dst path, so the env's serve phase (which only
consumes a path whose every hop has positive stock) serves very little. This
policy instead, per request, walks an actual BFS shortest path and activates
every hop on it — so the activated edge set is *by construction* a union of
complete relay paths.

Key insight (matched to env): generation and consumption are separated in
time. Activation this slot stocks each hop; later slots' serve phase BFS's the
positive-stock subgraph and will naturally route over whichever stocked path we
built. So we do NOT need to select a specific path id — we just need to make
*some* complete path positive-stocked, prioritized by request deadline.

Algorithm (per slot)
--------------------
1. Order pending requests by (deadline_t, arrival_t, remaining) — same EDF the
   env service uses.
2. For each request in order:
   a. Run a BFS over the *physical-edge subgraph* (obstacles: edges already
      used by an earlier request, node Tx/Rx budget, 对端不同).
   b. Take the shortest path src->dst.
   c. Activate every unused edge on it; mark their endpoints so later requests
      re-route around them.
3. Emit dual-port (tx, rx) actions for the chosen edges.
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


class PathGreedyPolicy:
    def __init__(
        self,
        use_stock_bonus: bool = True,
        stock_tol: float = 1.0e-6,
        prefer_shortest: bool = True,
        rate_weight: float = 1.0,
        dense_fill: bool = True,
        persist_kept: bool = True,
        urgent_edges: bool = True,
    ):
        self.use_stock_bonus = bool(use_stock_bonus)
        self.stock_tol = float(stock_tol)
        self.prefer_shortest = bool(prefer_shortest)
        self.rate_weight = float(rate_weight)
        self.dense_fill = bool(dense_fill)
        self.persist_kept = bool(persist_kept)
        self.urgent_edges = bool(urgent_edges)

    def _request_key(self, req, t: int):
        remaining = max(0.0, req.amount - req.served_amount)
        return (req.deadline_t - t, -remaining, req.dst_gs)
        # We sort by deadline (EDF): soonest deadline first.

    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        """Return dual-port actions dict for env.step.

        Returns ``(actions, scores)`` where actions maps node->(tx, rx).
        """
        active_ids = list(obs.physical_edge_ids)
        # adjacency over physical edges (node -> neighbor)
        adj: dict[str, list[str]] = {n: [] for n in obs.node_ids}
        edge_by_pair: dict[tuple[str, str], str] = {}
        for eid in active_ids:
            u, v = edge_endpoints(eid)
            if u == v or u not in adj or v not in adj:
                continue
            adj[u].append(v)
            adj[v].append(u)
            edge_by_pair[tuple(sorted((u, v)))] = eid

        # normalized per-edge rate for second-stage tie-break (avoid slow links)
        edges = obs.state.edge_windows
        raw_rates = {
            eid: float(edges[eid].rates[0]) for eid in active_ids if edges.get(eid) and edges[eid].rates
        }
        max_rate = max(raw_rates.values(), default=1.0) or 1.0
        edge_rate = {eid: v / max_rate for eid, v in raw_rates.items()}

        # sort requests EDF
        reqs = sorted(obs.state.pending_requests, key=lambda r: self._request_key(r, obs.state.t))

        # Dual-port occupancy, committed as each request's path is chosen so
        # later requests re-route around already-used ports. Under the env
        # model each node has ONE transmitter and ONE receiver, so a node may
        # be the transmitting end of one path AND the receiving end of another
        # in the same slot. We therefore track the two roles independently.
        tx_busy: set[str] = set()   # node already used as a transmitter
        rx_busy: set[str] = set()   # node already used as a receiver
        dual: dict[str, list[str]] = {
            node: [NodeActionSpace.IDLE, NodeActionSpace.IDLE] for node in obs.node_ids
        }
        chosen_pairs: set[tuple[str, str]] = set()

        for req in reqs:
            src, dst = req.src_gs, req.dst_gs
            if src not in adj or dst not in adj:
                continue
            remaining = max(0.0, req.amount - req.served_amount)
            if remaining <= 1.0e-9:
                continue
            # The source must still have a free transmitter and the destination
            # a free receiver, otherwise this path is impossible this slot.
            if src in tx_busy or dst in rx_busy:
                continue
            path = self._best_path(adj, src, dst, tx_busy, rx_busy, edge_by_pair, edge_rate)
            if path is None:
                continue
            # every node except the destination transmits, every node except
            # the source receives. Since each node can only transmit once and
            # receive once per slot, assign the ports directly per directed hop.
            for a, b in zip(path[:-1], path[1:]):
                if dual[a][0] != NodeActionSpace.IDLE or dual[b][1] != NodeActionSpace.IDLE:
                    # port conflict (should not happen given tx_busy/rx_busy)
                    path = None
                    break
                dual[a][0] = b          # a transmits to b
                dual[b][1] = a          # b receives from a
                tx_busy.add(a)
                rx_busy.add(b)
                chosen_pairs.add(tuple(sorted((a, b))))
            if path is None:
                continue

        if self.dense_fill:
            # Second pass: pack any still-idle ports with additional links so
            # the positive-key network the env's serve/routing BFS can use is
            # dense (more connected stock => more requests routable). Edges that
            # were active last slot are preferred (no switch -> full rate, keeps
            # paths continuously charged); among the rest, highest rate first.
            kept = set(obs.state.last_activated_edges)
            def _dense_key(eid: str):
                return (1 if self.persist_kept and eid in kept else 0, edge_rate.get(eid, 0.0))
            for eid in sorted(active_ids, key=_dense_key, reverse=True):
                u, v = edge_endpoints(eid)
                if u == v or u not in adj or v not in adj:
                    continue
                pair = tuple(sorted((u, v)))
                if pair not in edge_by_pair:
                    continue
                if pair in chosen_pairs:
                    continue
                if u not in tx_busy and v not in rx_busy:
                    dual[u][0] = v
                    dual[v][1] = u
                    tx_busy.add(u)
                    rx_busy.add(v)
                    chosen_pairs.add(pair)
                elif v not in tx_busy and u not in rx_busy:
                    dual[v][0] = u
                    dual[u][1] = v
                    tx_busy.add(v)
                    rx_busy.add(u)
                    chosen_pairs.add(pair)

        actions = {n: tuple(dual[n]) for n in obs.node_ids}
        return actions, {}

    def _best_path(
        self,
        adj: dict[str, list[str]],
        src: str,
        dst: str,
        tx_busy: set[str],
        rx_busy: set[str],
        edge_by_pair: dict[tuple[str, str], str],
        edge_rate: dict[str, float],
    ) -> list[str] | None:
        """Shortest src->dst path; among equal-length ones pick highest total rate.

        Stage 1 is a BFS that fixes the *minimum hop count* (shortest is always
        honoured). Stage 2 walks only the equal-length shortest edges and, among
        them, selects the path that maximises the sum of normalized edge rates
        (so slow links are avoided whenever a same-length faster path exists).
        Port budgets (one Tx / one Rx per node) and the ``u->v``/``v->u`` rule
        are enforced throughout.
        """
        if src in tx_busy:      # source needs a free Tx
            return None
        pnode = {src: (None, "")}
        q = deque([src])
        while q:
            node = q.popleft()
            for nxt in adj.get(node, ()):
                if nxt in pnode:
                    continue
                pair = tuple(sorted((node, nxt)))
                if pair not in edge_by_pair:
                    continue
                if node in tx_busy or nxt in rx_busy:
                    continue
                pnode[nxt] = (node, pair)
                q.append(nxt)
        if dst not in pnode:
            return None
        if self.rate_weight <= 0.0:
            # plain shortest path (default BFS if rate tie-break disabled)
            path: list[str] = [dst]
            cur = dst
            while pnode[cur][0] is not None:
                cur = pnode[cur][0]
                path.append(cur)
            path.reverse()
            return path
        # Reconstruct full shortest topology distances to ensure we never extend
        # beyond the minimum hop count (BFS parent pointers already give a valid
        # shortest path; recompute dist for the second-stage DP).
        dist = {src: 0}
        q = deque([src])
        while q:
            node = q.popleft()
            for nxt in adj.get(node, ()):
                if nxt in dist:
                    continue
                pair = tuple(sorted((node, nxt)))
                if pair not in edge_by_pair:
                    continue
                if node in tx_busy or nxt in rx_busy:
                    continue
                dist[nxt] = dist[node] + 1
                q.append(nxt)
        d = dist[dst]
        best: list[dict[str, list]] = [{} for _ in range(d + 1)]
        best[0][src] = [0.0, None, None]  # [cum_rate, prev, prev_edge]
        for level in range(d):
            for cur, (cr, _prev, _pe) in best[level].items():
                for nxt in adj.get(cur, ()):
                    if dist.get(nxt, -1) != level + 1:
                        continue
                    pair = tuple(sorted((cur, nxt)))
                    eid = edge_by_pair.get(pair)
                    if eid is None:
                        continue
                    if cur in tx_busy or nxt in rx_busy:
                        continue
                    new_score = cr + self.rate_weight * edge_rate.get(eid, 0.0)
                    cand = best[level + 1].get(nxt)
                    if cand is None or new_score > cand[0]:
                        best[level + 1][nxt] = [new_score, cur, pair]
        if dst not in best[d]:
            return None
        path_rev: list[str] = [dst]
        cur = dst
        for level in range(d, 0, -1):
            _s, cur, _e = best[level][cur]
            path_rev.append(cur)
        path = list(reversed(path_rev))
        return path


def path_greedy_actions(obs: GraphObservation) -> dict[str, tuple]:
    pol = PathGreedyPolicy()
    actions, _ = pol.act(obs)
    return actions