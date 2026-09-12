"""Full-topology, stock-aware service checks for the phased scheduler (PG-Phased).

``ServeProbe`` answers the two routing questions the three-stage phased
heuristic needs without touching the environment's own routing logic:

- ``has_path``: can the request be served **entirely from existing key stock**
  over the full positive-key network (including currently invisible links)?
- ``mixed_path`` / ``best_paths``: enumerate candidate ``src -> dst`` paths and
  score them by *serviceability* — bottleneck available keys, where a visible
  edge may be activated to generate fresh keys (stock + expected generation)
  and an invisible edge is usable only if its stock already covers the amount.
"""

from __future__ import annotations

from collections import deque


class ServeProbe:
    """Full-topology, stock-aware service checks for the phased scheduler.

    Phase 1 (has_path): a request is already fully servable in the positive-key
    network over ALL edges (including currently invisible links that still hold
    key stock).
    Phase 2 (mixed_path): find the shortest path where *invisible* edges are
    usable only if their stock >= amount, while *visible* edges are usable (they
    will be activated to generate fresh keys). Returns the edge list of the path.
    """

    def __init__(self, env):
        self.env = env

    def has_path(self, req, amount):
        path = self.env.routing.find_request_path(
            req, qkp=self.env.qkp, required_level=amount)
        return path is not None

    def mixed_path(self, req, amount, visible):
        adj = self.env.routing.adj
        qkp = self.env.qkp
        start, goal = req.src_gs, req.dst_gs
        if start not in adj or goal not in adj:
            return None
        parent = {start: (None, None)}
        q = deque([start])
        while q:
            node = q.popleft()
            if node == goal:
                break
            for nxt, eid in adj.get(node, ()):
                if nxt in parent:
                    continue
                if eid not in visible and qkp.get_level(eid) < amount:
                    continue
                parent[nxt] = (node, eid)
                q.append(nxt)
        if goal not in parent:
            return None
        edges = []
        cur = goal
        while parent[cur][0] is not None:
            edges.append(parent[cur][1])
            cur = parent[cur][0]
        return list(reversed(edges))

    def best_paths(self, req, amount, obs, visible, max_hops=2, visible_only=False,
                   k=8, min_invisible=0.0):
        """Enumerate candidate src->dst paths up to ``max_hops`` hops and score
        them by serviceability: bottleneck available keys (visible edge = current
        stock + expected generation incl. switch decay; invisible edge = stock),
        then total surplus. ``min_invisible`` = the stock an invisible edge must
        already hold to be usable as a mixed (stock-assisted) hop; visible-only
        (fresh) paths drop invisible edges entirely. Returns
        [(edge_ids, bottleneck, total)] sorted desc, truncated to k."""
        adj = self.env.routing.adj
        qkp = self.env.qkp
        slot = float(self.env.scenario.slot_seconds)
        windows = obs.state.edge_windows
        last_act = set(obs.state.last_activated_edges)
        start, goal = req.src_gs, req.dst_gs
        if start not in adj or goal not in adj:
            return []
        raw = []
        nbr1 = adj.get(start, ())
        for mid, e1 in nbr1:
            if mid == goal:
                continue
            for nxt, e2 in adj.get(mid, ()):
                if nxt == goal:
                    raw.append([e1, e2])
        if max_hops >= 3:
            for a, e1 in nbr1:
                if a in (start, goal):
                    continue
                for b, e2 in adj.get(a, ()):
                    if b in (start, goal, a):
                        continue
                    for e3 in (e3 for n2, e3 in adj.get(b, ()) if n2 == goal):
                        raw.append([e1, e2, e3])
        scored, seen = [], set()
        for path in raw:
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            avails = []
            ok = True
            for eid in path:
                level = qkp.get_level(eid)
                if eid in visible:
                    rate = float(windows[eid].rates[0])
                    decay = 1.0 if eid in last_act else 0.5
                    av = level + rate * decay * slot
                else:
                    if visible_only or level < min_invisible:
                        ok = False
                        break
                    av = level
                avails.append(av)
            if not ok:
                continue
            bot = min(avails)
            scored.append((path, bot, sum(avails)))
        scored.sort(key=lambda x: (-x[1], -x[2]))
        return scored[:k]
