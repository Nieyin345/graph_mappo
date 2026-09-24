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

from qkd_rl.baselines.greedy_relay_diffusion import (
    GreedyRelayDiffusionPolicyV3,
    compute_dynamic_relay_importance,
)
from qkd_rl.core.types import KeyRequest
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
        v3_mix: float = 0.0,
        fill_mode: str = "rate",
        v3_expert: bool = False,
        v3_weights: tuple = (1.0, 10.0, 1.0, 0.5, 0.2),
        max_hops: int | None = None,
    ):
        self.use_stock_bonus = bool(use_stock_bonus)
        self.stock_tol = float(stock_tol)
        self.prefer_shortest = bool(prefer_shortest)
        self.rate_weight = float(rate_weight)
        self.dense_fill = bool(dense_fill)
        self.persist_kept = bool(persist_kept)
        self.urgent_edges = bool(urgent_edges)
        # v3_mix>0 时把 V3 边分（rate×importance+completion+keep−switch）混入
        # 路径评分；v3_expert=True 时策略内部自动算 V3 分并强制 v3_mix=1.0
        if v3_expert and v3_mix == 0.0:
            v3_mix = 1.0
        self.v3_mix = float(v3_mix)
        self.v3_expert = bool(v3_expert)
        self.v3_weights = tuple(float(x) for x in v3_weights)
        self._v3_scorer: GreedyRelayDiffusionPolicyV3 | None = None
        # None = 严格最短优先（同跳数选分最高）；整数 N = 在 <=N 跳内自由选
        # 总分最高的路径（可绕行更长但更快/更优的链路，max_hops=3 即三跳实验）
        self.max_hops = max_hops
        # dense_fill 第二遍的填充排序：'rate'=persist 优先+边分；
        # 'request'=persist 优先+需求扩散 importance+边分（面向 pending 请求）
        self.fill_mode = fill_mode

    def _request_key(self, req, t: int):
        remaining = max(0.0, req.amount - req.served_amount)
        return (req.deadline_t - t, -remaining, req.dst_gs)
        # We sort by deadline (EDF): soonest deadline first.

    def act(
        self,
        obs: GraphObservation,
        v3_scores: dict[str, float] | None = None,
    ) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        """Return dual-port actions dict for env.step.

        Returns ``(actions, scores)`` where actions maps node->(tx, rx).

        ``v3_scores``: optional per-edge V3 scores (``GreedyRelayDiffusionPolicyV3
        .score_edges`` output). When ``v3_mix > 0`` they are mixed into the path
        scoring so path choice reflects demand/importance/completion signals,
        not just link rate.
        """
        active_ids = list(obs.generation_edge_ids)
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

        # per-edge score for second-stage tie-break: rate-normalized by default,
        # or V3's full score (rate×importance + completion + keep − switch) when
        # v3_expert/v3_mix is on (min-max normalized to [0, 1]).
        if self.v3_expert:
            if self._v3_scorer is None:
                self._v3_scorer = GreedyRelayDiffusionPolicyV3(
                    rate_weight=self.v3_weights[0],
                    importance_weight=self.v3_weights[1],
                    completion_weight=self.v3_weights[2],
                    keep_weight=self.v3_weights[3],
                    switch_weight=self.v3_weights[4])
            v3_scores = self._v3_scorer.score_edges(obs)
        edges = obs.state.edge_windows
        raw_rates = {
            eid: float(edges[eid].rates[0]) for eid in active_ids if edges.get(eid) and edges[eid].rates
        }
        max_rate = max(raw_rates.values(), default=1.0) or 1.0
        edge_rate = {eid: v / max_rate for eid, v in raw_rates.items()}
        edge_score: dict[str, float] = dict(edge_rate)
        if self.v3_mix > 0.0 and v3_scores:
            vs = {eid: float(v3_scores[eid]) for eid in active_ids if eid in v3_scores}
            if vs:
                vmin, vmax = min(vs.values()), max(vs.values())
                span = vmax - vmin
                v3n = {eid: ((v - vmin) / span if span > 0 else 1.0) for eid, v in vs.items()}
                m = self.v3_mix
                edge_score = {
                    eid: (1.0 - m) * edge_rate.get(eid, 0.0) + m * v3n.get(eid, 0.0)
                    for eid in active_ids
                }

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
            path = self._best_path(adj, src, dst, tx_busy, rx_busy, edge_by_pair,
                                   edge_score, self.max_hops)
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
            # paths continuously charged); among the rest, 'request' mode boosts
            # edges near pending demand (relay importance), then edge score.
            imp_norm = None
            if self.fill_mode == "request":
                importance = compute_dynamic_relay_importance(obs)
                imax = max(importance.values(), default=0.0) or 1.0
                imp_norm = {eid: importance.get(eid, 0.0) / imax for eid in active_ids}
            kept = set(obs.state.last_activated_edges)
            def _dense_key(eid: str):
                imp = imp_norm.get(eid, 0.0) if imp_norm is not None else 0.0
                return (1 if self.persist_kept and eid in kept else 0, imp, edge_score.get(eid, 0.0))
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
        edge_score: dict[str, float],
        max_hops: int | None = None,
    ) -> list[str] | None:
        """Shortest src->dst path; among equal-length ones pick highest total score.

        Stage 1 is a BFS that fixes the *minimum hop count* (shortest is always
        honoured). Stage 2 walks only the equal-length shortest edges and, among
        them, selects the path that maximises the sum of per-edge scores
        (default: normalized rate, so slow links are avoided whenever a
        same-length faster path exists; with ``v3_mix>0`` the score is blended
        with V3's demand/importance signal). Port budgets (one Tx / one Rx per
        node) and the ``u->v``/``v->u`` rule are enforced throughout.

        When ``max_hops=N`` is given the shortest-only constraint is relaxed:
        any *simple* path with at most N hops is allowed and the highest total
        score wins, so a longer detour over faster links can beat a short slow
        path (max_hops=3 => allow 3-edge paths).
        """
        if src in tx_busy:      # source needs a free Tx
            return None
        if max_hops is None:
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
            # Reconstruct full shortest topology distances to ensure we never
            # extend beyond the minimum hop count (BFS parent pointers already
            # give a valid shortest path; recompute dist for the 2nd-stage DP).
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
                        new_score = cr + self.rate_weight * edge_score.get(eid, 0.0)
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

        # --- relaxed: any simple path with hop count <= max_hops, max total score ---
        # state per (level, node): (score, prev_node, prev_pair, visited)
        best: list[dict[str, tuple]] = [{} for _ in range(max_hops + 1)]
        best[0][src] = (0.0, None, None, frozenset([src]))
        for level in range(max_hops):
            for cur, (cr, _pr, _pe, vis) in best[level].items():
                for nxt in adj.get(cur, ()):
                    if nxt in vis:
                        continue
                    pair = tuple(sorted((cur, nxt)))
                    eid = edge_by_pair.get(pair)
                    if eid is None:
                        continue
                    if cur in tx_busy or nxt in rx_busy:
                        continue
                    new_score = cr + self.rate_weight * edge_score.get(eid, 0.0)
                    cand = best[level + 1].get(nxt)
                    if cand is None or new_score > cand[0]:
                        best[level + 1][nxt] = (new_score, cur, pair, vis | {nxt})
        b_level, b_score = -1, -1.0
        for level in range(1, max_hops + 1):
            cand = best[level].get(dst)
            if cand is not None and cand[0] > b_score:
                b_level, b_score = level, cand[0]
        if b_level < 0:
            return None
        path_rev: list[str] = [dst]
        cur = dst
        for level in range(b_level, 0, -1):
            _s, cur, _e, _v = best[level][cur]
            path_rev.append(cur)
        path = list(reversed(path_rev))
        return path


def path_greedy_actions(obs: GraphObservation) -> dict[str, tuple]:
    pol = PathGreedyPolicy()
    actions, _ = pol.act(obs)
    return actions


class PathScoreGreedy:
    """User-designed variant: V3 per-edge scores + BFS shortest-path activation
    (whole path claimed as one unit), then greedy fill of the remaining edges.

    Step 1: requests in EDF order; each gets a shortest path via BFS and, among
    equal-length paths, the one with the highest *sum of V3 edge scores*; the
    whole path is activated at once (dual-port, one Tx + one Rx per node).
    Step 2: chosen edges are excluded; the rest are sorted by V3 score and
    greedily claimed while the ports are free.
    """

    def __init__(self, weights: tuple = (1.0, 10.0, 1.0, 0.5, 0.2),
                 phased: bool = False, principles: bool = False, router=None):
        self.weights = tuple(float(x) for x in weights)
        self._v3_scorer: GreedyRelayDiffusionPolicyV3 | None = None
        self._pg = PathGreedyPolicy()  # reuse _best_path (V3-score path DP)
        # phased 三阶段：阶段1 现有密钥全量网络能否完全服务（有则跳过）；
        # 阶段2 混合存量通道（可见边激活 + 不可见边存量≥需求 拼最短路径，拼上则跳过）；
        # 阶段3 全新纯可见通道（原逻辑）。
        self.phased = bool(phased)
        # principles 三原则：第一轮只处理 2 跳（先现有密钥、再混合、再全新，
        # 选路以"服务量优先、余量次之"为准）；第二轮剩余请求放宽到 3 跳+
        #（同样必须当前能立即用才选）；最后剩余边分数排序抢占（为未来积累）。
        self.principles = bool(principles)
        self.router = router  # has_path / best_paths 回调

    def _v3_scores(self, obs: GraphObservation) -> dict[str, float]:
        if self._v3_scorer is None:
            self._v3_scorer = GreedyRelayDiffusionPolicyV3(
                rate_weight=self.weights[0], importance_weight=self.weights[1],
                completion_weight=self.weights[2], keep_weight=self.weights[3],
                switch_weight=self.weights[4])
        return self._v3_scorer.score_edges(obs)

    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        active_ids = list(obs.generation_edge_ids)
        adj: dict[str, list[str]] = {n: [] for n in obs.node_ids}
        edge_by_pair: dict[tuple[str, str], str] = {}
        for eid in active_ids:
            u, v = edge_endpoints(eid)
            if u == v or u not in adj or v not in adj:
                continue
            adj[u].append(v)
            adj[v].append(u)
            edge_by_pair[tuple(sorted((u, v)))] = eid

        edge_scores = self._v3_scores(obs)
        tx_busy: set[str] = set()
        rx_busy: set[str] = set()
        dual: dict[str, list[str]] = {
            n: [NodeActionSpace.IDLE, NodeActionSpace.IDLE] for n in obs.node_ids
        }
        chosen_pairs: set[tuple[str, str]] = set()

        # ---- Step 1: EDF requests, shortest path, V3-score-best, whole-path claim ----
        reqs = sorted(
            obs.state.pending_requests,
            key=lambda r: (r.deadline_t - obs.state.t, -(r.amount - r.served_amount)),
        )
        if self.principles and self.router is not None:
            self._principles_loop(reqs, obs, set(active_ids),
                                  tx_busy, rx_busy, dual, chosen_pairs)
        else:
            for req in reqs:
                src, dst = req.src_gs, req.dst_gs
                if src not in adj or dst not in adj:
                    continue
                remaining = max(0.0, req.amount - req.served_amount)
                if remaining <= 1.0e-9:
                    continue
                # 阶段1：现有密钥全量正库存网络（含不可见边存量）能否完全服务
                if self.phased and self.router is not None:
                    if self.router.has_path(req, remaining):
                        continue
                    # 阶段2/3：2跳优先（混合→全新），再放宽3跳（混合→全新）；
                    # 选路以服务量（瓶颈可用）优先，余量次之
                    claimed = False
                    for hops, vis in ((2, False), (2, True),
                                      (3, False), (3, True)):
                        paths = self.router.best_paths(
                            req, remaining, obs, set(active_ids),
                            max_hops=hops, visible_only=vis,
                            min_invisible=remaining if not vis else 0.0)
                        if self._try_claim(paths, active_ids, tx_busy, rx_busy,
                                           dual, chosen_pairs):
                            claimed = True
                            break
                    if claimed:
                        continue
                # 阶段3：全新纯可见通道（原逻辑）
                if src in tx_busy or dst in rx_busy:
                    continue
                path = self._pg._best_path(adj, src, dst, tx_busy, rx_busy,
                                           edge_by_pair, edge_scores)
                if path is None:
                    continue
                for a, b in zip(path[:-1], path[1:]):
                    if dual[a][0] != NodeActionSpace.IDLE or dual[b][1] != NodeActionSpace.IDLE:
                        path = None
                        break
                    dual[a][0] = b          # a transmits to b
                    dual[b][1] = a          # b receives from a
                    tx_busy.add(a)
                    rx_busy.add(b)
                    chosen_pairs.add(tuple(sorted((a, b))))

        # ---- Step 2: greedy fill of remaining edges by V3 score ----
        for eid in sorted(active_ids, key=lambda e: edge_scores.get(e, 0.0), reverse=True):
            u, v = edge_endpoints(eid)
            if u == v or u not in adj or v not in adj:
                continue
            pair = tuple(sorted((u, v)))
            if pair not in edge_by_pair or pair in chosen_pairs:
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

        return {n: tuple(dual[n]) for n in obs.node_ids}, {}

    def _principles_loop(self, reqs, obs, active_set,
                         tx_busy, rx_busy, dual, chosen_pairs):
        """Principle-based scheduling.

        Round 1 (2-hop, Principle 1 - serve-now + surplus): stage 1 existing
        full stock service -> stage 2 mixed 2-hop -> stage 3 fresh 2-hop; paths
        are compared by servable amount (bottleneck), surplus second.
        Round 2 (3-hop+, Principle 2 - still must serve right now): stages 2/3
        for the remaining requests with paths relaxed to 3 hops.
        """
        def remaining_of(req):
            return max(0.0, req.amount - req.served_amount)

        def claim_best(paths):
            for edge_ids, bot, _ in paths:
                if bot <= 0:
                    continue
                if self._claim_mixed(edge_ids, active_set, tx_busy, rx_busy,
                                     dual, chosen_pairs):
                    return True
            return False

        done: set[KeyRequest] = set()
        for stage in ("stock", "mixed", "fresh"):
            for req in reqs:
                if req in done:
                    continue
                rem = remaining_of(req)
                if rem <= 1e-9:
                    done.add(req)
                    continue
                if stage == "stock":
                    if self.router.has_path(req, rem):
                        done.add(req)
                elif stage == "mixed":
                    paths = self.router.best_paths(req, rem, obs, active_set,
                                                   max_hops=2, visible_only=False,
                                                   min_invisible=rem)
                    if claim_best(paths):
                        done.add(req)
                else:
                    paths = self.router.best_paths(req, rem, obs, active_set,
                                                   max_hops=2, visible_only=True)
                    if claim_best(paths):
                        done.add(req)

        for stage in ("mixed", "fresh"):
            for req in reqs:
                if req in done:
                    continue
                rem = remaining_of(req)
                if rem <= 1e-9:
                    done.add(req)
                    continue
                if stage == "mixed":
                    paths = self.router.best_paths(req, rem, obs, active_set,
                                                   max_hops=3, visible_only=False,
                                                   min_invisible=rem)
                else:
                    paths = self.router.best_paths(req, rem, obs, active_set,
                                                   max_hops=3, visible_only=True)
                if claim_best(paths):
                    done.add(req)

    @staticmethod
    def _try_claim(paths, active_ids, tx_busy, rx_busy,
                   dual, chosen_pairs) -> bool:
        """Try candidate paths in serviceability order; claim the first whose
        ports are free (all-or-nothing per path)."""
        active = set(active_ids)
        for edge_ids, bot, _ in paths:
            if bot <= 0:
                continue
            if PathScoreGreedy._claim_mixed(edge_ids, active, tx_busy, rx_busy,
                                           dual, chosen_pairs):
                return True
        return False

    @staticmethod
    def _claim_mixed(edge_ids: list[str], active_set: set[str],
                     tx_busy: set[str], rx_busy: set[str],
                     dual: dict[str, list[str]],
                     chosen_pairs: set[tuple[str, str]]) -> bool:
        """Activate the visible edges of a mixed (stock-assisted) path. All-or-
        nothing: port conflicts anywhere return False without claiming."""
        to_claim = [e for e in edge_ids if e in active_set]
        for e in to_claim:
            u, v = edge_endpoints(e)
            if u in tx_busy or v in rx_busy:
                return False
        for e in to_claim:
            u, v = edge_endpoints(e)
            dual[u][0] = v
            dual[v][1] = u
            tx_busy.add(u)
            rx_busy.add(v)
            chosen_pairs.add(tuple(sorted((u, v))))
        return True