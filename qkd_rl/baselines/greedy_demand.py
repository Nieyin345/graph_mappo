"""Greedy demand baseline: activate the link with the largest pending key demand.

Independent of the training stack: consumes only :class:`GraphObservation`
fields and returns ``(actions, scores)`` like every baseline policy.
"""
from __future__ import annotations

from qkd_rl.baselines._matching import edge_demand, edge_map, greedy_matching_actions, pending_demand
from qkd_rl.env.graph_builder import GraphObservation


class GreedyDemandPolicy:
    """Global greedy matching by pending key demand.

    Demand is the total amount of not-yet-served key requests a link can
    plausibly serve: the exact pair demand for a direct GS-GS link, otherwise
    the incident GS demand for GS-HAP / GS-SAT links. Score per legal link is
    ``demand + rate_weight * rate``; edges are matched globally so both
    endpoints agree on the activated set.
    """

    def __init__(self, rate_weight: float = 0.2):
        self.rate_weight = float(rate_weight)

    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        pair_demand, incident = pending_demand(obs)
        endpoints, _pair_to_edge = edge_map(obs)
        edge_rate = {edge_id: obs.state.edge_windows[edge_id].rates[0] for edge_id in obs.physical_edge_ids}
        edge_scores: dict[str, float] = {}
        for edge_id in obs.physical_edge_ids:
            src, dst = endpoints[edge_id]
            demand = edge_demand(src, dst, pair_demand, incident)
            edge_scores[edge_id] = demand + self.rate_weight * edge_rate.get(edge_id, 0.0)
        return greedy_matching_actions(obs, edge_scores, tie_rates=edge_rate)
