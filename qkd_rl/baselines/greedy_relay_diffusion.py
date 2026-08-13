"""Non-learning version of the demand-edge relay diffusion.

The demand_edge model encodes pending GS-GS demand as logical edges, diffuses
it onto physical edges with bidirectional BFS up to ``max_path_links`` hops,
then lets an MLP score the physical edges. This baseline performs exactly the
same diffusion step and replaces the MLP/RL with a fixed score:

    score = rate_weight * rate_norm + importance_weight * importance_norm

The matching is then selected greedily by score, exactly like the other
global-matching baselines.
"""

from __future__ import annotations

from qkd_rl.baselines._matching import edge_map, greedy_matching_actions
from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.graph_builder import GraphObservation
from qkd_rl.env.relay_importance import compute_relay_importance


def compute_dynamic_relay_importance(
    obs: GraphObservation,
    max_path_links: int = 3,
    hop_decay_factor: float = 0.25,
    capacity_strength: float = 1.0,
    min_scarcity: float = 0.0,
    wait_urgency_tau_ratio: float = 0.8,
    ignore_consumption: bool = False,
    include_stocked_unavailable: bool = True,
) -> dict[str, float]:
    """Dynamic relay importance shared by heuristic and MILP baselines."""
    return compute_relay_importance(
        node_ids=obs.node_ids,
        physical_edge_ids=obs.physical_edge_ids,
        pending_requests=obs.state.pending_requests,
        qkp_snapshot=obs.state.qkp_snapshot,
        qkp_capacity=obs.state.qkp_capacity,
        t=obs.state.t,
        max_path_links=max_path_links,
        hop_decay_factor=hop_decay_factor,
        capacity_strength=capacity_strength,
        min_scarcity=min_scarcity,
        wait_urgency_tau_ratio=wait_urgency_tau_ratio,
        ignore_consumption=ignore_consumption,
        include_stocked_unavailable=include_stocked_unavailable,
        all_edge_ids=list(obs.state.edge_windows.keys()) if include_stocked_unavailable else None,
    )


class GreedyRelayDiffusionPolicy:
    def __init__(
        self,
        rate_weight: float = 1.0,
        importance_weight: float = 2.0,
        hop_decay_factor: float = 0.25,
        max_path_links: int = 3,
        wait_urgency_tau_ratio: float = 0.8,
        ignore_consumption: bool = False,
        include_stocked_unavailable: bool = True,
    ):
        self.rate_weight = float(rate_weight)
        self.importance_weight = float(importance_weight)
        self.hop_decay_factor = max(0.0, float(hop_decay_factor))
        self.max_path_links = max(1, int(max_path_links))
        self.wait_urgency_tau_ratio = float(wait_urgency_tau_ratio)
        self.ignore_consumption = bool(ignore_consumption)
        self.include_stocked_unavailable = bool(include_stocked_unavailable)

    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        active_ids = list(obs.physical_edge_ids)
        if not active_ids:
            return greedy_matching_actions(obs, {})
        importance = compute_dynamic_relay_importance(
            obs,
            max_path_links=self.max_path_links,
            hop_decay_factor=self.hop_decay_factor,
            wait_urgency_tau_ratio=self.wait_urgency_tau_ratio,
            ignore_consumption=self.ignore_consumption,
            include_stocked_unavailable=self.include_stocked_unavailable,
        )
        rates = {
            edge_id: float(obs.state.edge_windows[edge_id].rates[0])
            for edge_id in active_ids
        }
        max_rate = max(rates.values(), default=1.0) or 1.0
        edge_scores: dict[str, float] = {}
        for edge_id in active_ids:
            rate_norm = rates[edge_id] / max_rate
            importance_norm = importance.get(edge_id, 0.0)
            edge_scores[edge_id] = (
                self.rate_weight * rate_norm
                + self.importance_weight * importance_norm
            )
        return greedy_matching_actions(obs, edge_scores, tie_rates=rates)


class GreedyRelayDiffusionPolicyV2:
    """Improved BFS + greedy scoring with keep/switch/completion terms."""

    def __init__(
        self,
        rate_weight: float = 1.0,
        importance_weight: float = 3.0,
        completion_weight: float = 1.0,
        keep_weight: float = 0.5,
        switch_weight: float = 0.2,
        hop_decay_factor: float = 0.25,
        max_path_links: int = 3,
        wait_urgency_tau_ratio: float = 0.8,
        ignore_consumption: bool = False,
        include_stocked_unavailable: bool = True,
    ):
        self.rate_weight = float(rate_weight)
        self.importance_weight = float(importance_weight)
        self.completion_weight = float(completion_weight)
        self.keep_weight = float(keep_weight)
        self.switch_weight = float(switch_weight)
        self.hop_decay_factor = max(0.0, float(hop_decay_factor))
        self.max_path_links = max(1, int(max_path_links))
        self.wait_urgency_tau_ratio = float(wait_urgency_tau_ratio)
        self.ignore_consumption = bool(ignore_consumption)
        self.include_stocked_unavailable = bool(include_stocked_unavailable)

    def score_edges(self, obs: GraphObservation) -> dict[str, float]:
        """Return a score for every legal physical edge (before matching)."""
        active_ids = list(obs.physical_edge_ids)
        if not active_ids:
            return {}
        endpoints, pair_to_edge = edge_map(obs)
        importance = compute_dynamic_relay_importance(
            obs,
            max_path_links=self.max_path_links,
            hop_decay_factor=self.hop_decay_factor,
            wait_urgency_tau_ratio=self.wait_urgency_tau_ratio,
            ignore_consumption=self.ignore_consumption,
            include_stocked_unavailable=self.include_stocked_unavailable,
        )
        rates = {
            edge_id: float(obs.state.edge_windows[edge_id].rates[0])
            for edge_id in active_ids
        }
        max_rate = max(rates.values(), default=1.0) or 1.0
        levels = obs.state.qkp_snapshot

        completion: dict[str, float] = {}
        candidates = {
            node_id: set(obs.action_candidates[node_id]) - {NodeActionSpace.IDLE}
            for node_id in obs.node_ids
        }
        pairs: dict[tuple[str, str], float] = {}
        for req in obs.state.pending_requests:
            remaining = max(0.0, req.amount - req.served_amount)
            if remaining <= 1.0e-9:
                continue
            pair = tuple(sorted((req.src_gs, req.dst_gs)))
            pairs[pair] = pairs.get(pair, 0.0) + remaining
        for pair, amount in pairs.items():
            a, b = pair
            for relay in candidates.get(a, set()) & candidates.get(b, set()):
                e_a = pair_to_edge.get(tuple(sorted((a, relay))))
                e_b = pair_to_edge.get(tuple(sorted((b, relay))))
                if e_a is None or e_b is None:
                    continue
                stocked_a = levels.get(e_a, 0.0) > 0.0
                stocked_b = levels.get(e_b, 0.0) > 0.0
                if stocked_a and stocked_b:
                    continue
                if not stocked_a and not stocked_b:
                    continue
                completion[e_b if stocked_a else e_a] = (
                    completion.get(e_b if stocked_a else e_a, 0.0) + amount
                )
        max_completion = max(completion.values(), default=1.0) or 1.0
        last_activated = set(obs.state.last_activated_edges)

        edge_scores: dict[str, float] = {}
        for edge_id in active_ids:
            rate_norm = rates[edge_id] / max_rate
            importance_norm = importance.get(edge_id, 0.0)
            completion_norm = completion.get(edge_id, 0.0) / max_completion
            keep_norm = 1.0 if (
                edge_id in last_activated
                and (importance_norm > 0.0 or completion_norm > 0.0)
            ) else 0.0
            switch_norm = 0.0 if edge_id in last_activated else 1.0
            edge_scores[edge_id] = (
                self.rate_weight * rate_norm * (1.0 + self.importance_weight * importance_norm)
                + self.completion_weight * completion_norm
                + self.keep_weight * keep_norm
                - self.switch_weight * switch_norm
            )
        return edge_scores

    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        edge_scores = self.score_edges(obs)
        rates = {
            edge_id: float(obs.state.edge_windows[edge_id].rates[0])
            for edge_id in obs.physical_edge_ids
        }
        return greedy_matching_actions(obs, edge_scores, tie_rates=rates)


class GreedyRelayDiffusionPolicyV3(GreedyRelayDiffusionPolicyV2):
    """V2 scoring that also considers stocked but currently-unavailable edges.

    The stocked-unavailable bridge needs a much stronger importance weight
    than the active-only V2 signal: a 2-day comparison showed w=3 roughly ties
    V2 while w=10 lifts success rate by ~2 points.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_stocked_unavailable", True)
        kwargs.setdefault("importance_weight", 10.0)
        super().__init__(*args, **kwargs)
