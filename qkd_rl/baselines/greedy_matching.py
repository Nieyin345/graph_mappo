"""Combined greedy matching baseline: rate + inventory + demand.

A single global greedy matching whose edge score mixes the three signals that
matter in this environment: how fast the link generates keys (rate), how much
key inventory it already holds (level), and how much pending demand it can
serve (demand). Each term is normalized to [0, 1] by the per-step maximum so
the weights stay interpretable across small/full network scales, and the
`keep_weight` bonus prefers links that were active in the previous slot
(avoids the switch-cost rate decay).

Consumes only GraphObservation fields like every other baseline.
"""

from __future__ import annotations

from qkd_rl.baselines._matching import edge_demand, edge_map, greedy_matching_actions, pending_demand
from qkd_rl.env.graph_builder import GraphObservation


class GreedyMatchingPolicy:
    """Global greedy matching with a tunable score mix.

    `score(edge) = (rate_weight * rate_norm + level_weight * level_norm
    + demand_weight * demand_norm) * keep_multiplier` where each norm term is
    divided by its per-step maximum and `keep_multiplier = 1 + keep_weight`
    when the edge was active in the previous slot.
    """

    def __init__(
        self,
        rate_weight: float = 1.0,
        level_weight: float = 0.5,
        demand_weight: float = 1.0,
        keep_weight: float = 0.5,
    ):
        self.rate_weight = float(rate_weight)
        self.level_weight = float(level_weight)
        self.demand_weight = float(demand_weight)
        self.keep_weight = float(keep_weight)

    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        endpoints, _pair_to_edge = edge_map(obs)
        pair_demand, incident = pending_demand(obs)
        windows = obs.state.edge_windows
        rates = {edge_id: windows[edge_id].rates[0] for edge_id in obs.physical_edge_ids}
        levels = dict(obs.state.qkp_snapshot)
        max_rate = max(rates.values(), default=1.0) or 1.0
        max_level = max(levels.values(), default=1.0) or 1.0
        max_demand = max(
            (edge_demand(src, dst, pair_demand, incident) for src, dst in endpoints.values()),
            default=1.0,
        ) or 1.0
        last_activated = set(obs.state.last_activated_edges)

        edge_scores: dict[str, float] = {}
        for edge_id in obs.physical_edge_ids:
            src, dst = endpoints[edge_id]
            rate_norm = rates.get(edge_id, 0.0) / max_rate
            level_norm = levels.get(edge_id, 0.0) / max_level
            demand_norm = edge_demand(src, dst, pair_demand, incident) / max_demand
            score = (
                self.rate_weight * rate_norm
                + self.level_weight * level_norm
                + self.demand_weight * demand_norm
            )
            if self.keep_weight > 0.0 and edge_id in last_activated:
                score *= 1.0 + self.keep_weight
            edge_scores[edge_id] = score
        return greedy_matching_actions(obs, edge_scores, tie_rates=rates)
