"""Static relay-importance greedy baseline.

Same global greedy matching as the demand-edge diffusion baseline, but the
relay importance is computed once at construction time from the full
topology (all scenario edges and all GS pairs). It is not re-derived from the
currently visible graph or the current pending requests, so the per-edge
importance is fixed for the whole run.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations

from qkd_rl.baselines._matching import greedy_matching_actions
from qkd_rl.env.graph_builder import GraphObservation


class GreedyStaticRelayPolicy:
    def __init__(
        self,
        rate_weight: float = 1.0,
        importance_weight: float = 2.0,
        hop_decay_factor: float = 0.25,
        max_path_links: int = 3,
        edges: list | None = None,
        gs_ids: list[str] | None = None,
    ):
        self.rate_weight = float(rate_weight)
        self.importance_weight = float(importance_weight)
        self.hop_decay_factor = max(0.0, float(hop_decay_factor))
        self.max_path_links = max(1, int(max_path_links))
        self._importance: dict[str, float] | None = None
        if edges is not None and gs_ids is not None:
            self._build_static_importance(edges, gs_ids)

    def _build_static_importance(self, edges, gs_ids: list[str]) -> None:
        endpoints: dict[str, tuple[str, str]] = {
            edge.edge_id: (edge.src, edge.dst) for edge in edges
        }
        adj: dict[str, list[str]] = {}
        for edge_id, (src, dst) in endpoints.items():
            adj.setdefault(src, []).append(dst)
            adj.setdefault(dst, []).append(src)
        inf = 10**6
        totals: dict[str, float] = {}
        for src, dst in combinations(sorted(gs_ids), 2):
            src_dist = self._bfs(src, adj)
            dst_dist = self._bfs(dst, adj)
            for edge_id, (u, v) in endpoints.items():
                a = min(src_dist.get(u, inf), src_dist.get(v, inf))
                b = min(dst_dist.get(u, inf), dst_dist.get(v, inf))
                if a >= inf or b >= inf:
                    continue
                total_hops = a + b + 1
                if total_hops > self.max_path_links:
                    continue
                decay = self.hop_decay_factor ** max(0, total_hops - 2)
                totals[edge_id] = totals.get(edge_id, 0.0) + decay
        max_value = max(totals.values(), default=1.0) or 1.0
        self._importance = {
            edge_id: value / max_value for edge_id, value in totals.items()
        }

    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        if self._importance is None:
            raise ValueError(
                "GreedyStaticRelayPolicy requires edges and gs_ids at construction "
                "so the static relay importance can be computed once."
            )
        active_ids = list(obs.physical_edge_ids)
        if not active_ids:
            return greedy_matching_actions(obs, {})
        rates = {
            edge_id: float(obs.state.edge_windows[edge_id].rates[0])
            for edge_id in active_ids
        }
        max_rate = max(rates.values(), default=1.0) or 1.0
        edge_scores: dict[str, float] = {}
        for edge_id in active_ids:
            rate_norm = rates[edge_id] / max_rate
            importance_norm = self._importance.get(edge_id, 0.0)
            edge_scores[edge_id] = (
                self.rate_weight * rate_norm
                + self.importance_weight * importance_norm
            )
        return greedy_matching_actions(obs, edge_scores, tie_rates=rates)

    @staticmethod
    def _bfs(start: str, adj: dict[str, list[str]]) -> dict[str, int]:
        dist = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nxt in adj.get(node, ()):
                if nxt in dist:
                    continue
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
        return dist
