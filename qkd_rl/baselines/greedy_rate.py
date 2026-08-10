from __future__ import annotations

from qkd_rl.baselines._matching import greedy_matching_actions
from qkd_rl.env.graph_builder import GraphObservation


class GreedyRatePolicy:
    """Greedy rate baseline: global matching by link rate.

    ``use_future_mean_rate=True`` scores a link by the mean rate over the whole
    forward window (rates[1:]) instead of only the current slot. Edges are
    matched globally (each node activates at most one link, both endpoints
    agree) instead of each node picking independently, which leaves the
    resolver nothing to reject.
    """

    def __init__(self, use_future_mean_rate: bool = False):
        self.use_future_mean_rate = bool(use_future_mean_rate)

    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        windows = obs.state.edge_windows
        if self.use_future_mean_rate:
            edge_rate = {
                edge_id: sum(windows[edge_id].rates[1:]) / max(1, len(windows[edge_id].rates) - 1)
                for edge_id in obs.physical_edge_ids
            }
        else:
            edge_rate = {edge_id: windows[edge_id].rates[0] for edge_id in obs.physical_edge_ids}
        return greedy_matching_actions(obs, edge_rate, tie_rates=edge_rate)
