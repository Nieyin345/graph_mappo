"""Greedy QKP baseline: activate the link whose QKP pool holds the most keys.

Independent of the training stack: consumes only :class:`GraphObservation`
fields and returns ``(actions, scores)`` like every baseline policy.
"""
from __future__ import annotations

from qkd_rl.baselines._matching import greedy_matching_actions
from qkd_rl.env.graph_builder import GraphObservation


class GreedyQKPPolicy:
    """Global greedy matching by QKP inventory score.

    Score per legal link:
        level + low_inventory_weight * (max_level - level) + rate_weight * rate

    ``max_level`` is the highest inventory among currently visible links, so
    ``low_inventory_weight > 0`` prefers emptier pools (less overflow risk)
    while ``rate_weight`` adds a rate bonus. Edges are matched globally so
    both endpoints agree and the resolver activates the whole match.
    """

    def __init__(self, low_inventory_weight: float = 0.0, rate_weight: float = 0.2):
        self.low_inventory_weight = float(low_inventory_weight)
        self.rate_weight = float(rate_weight)

    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        edge_level = dict(obs.state.qkp_snapshot)
        edge_rate = {edge_id: obs.state.edge_windows[edge_id].rates[0] for edge_id in obs.physical_edge_ids}
        max_level = max(edge_level.values(), default=0.0)
        edge_scores: dict[str, float] = {}
        for edge_id in obs.physical_edge_ids:
            level = edge_level.get(edge_id, 0.0)
            edge_scores[edge_id] = (
                level
                + self.low_inventory_weight * (max_level - level)
                + self.rate_weight * edge_rate.get(edge_id, 0.0)
            )
        return greedy_matching_actions(obs, edge_scores, tie_rates=edge_rate)
