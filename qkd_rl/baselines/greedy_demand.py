"""Greedy demand baseline: activate the link with the largest pending key demand.

Independent of the training stack: consumes only :class:`GraphObservation`
fields and returns ``(actions, scores)`` like every baseline policy.
"""
from __future__ import annotations

from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.graph_builder import GraphObservation


class GreedyDemandPolicy:
    """Pick, per node, the neighbour with the largest pending demand.

    Demand is the total amount of not-yet-served key requests between the
    node and that neighbour (symmetric pair). Ties are broken by the current
    link rate so equal-demand pairs prefer faster links.
    """

    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        demand = _pending_demand_by_pair(obs)
        pair_to_edge = _pair_to_edge(obs)
        edge_rate = {edge_id: obs.state.edge_windows[edge_id].rates[0] for edge_id in obs.physical_edge_ids}

        actions: dict[str, str] = {}
        scores: dict[str, dict[str, float]] = {}
        for node_id in obs.node_ids:
            node_scores: dict[str, float] = {}
            best_action = NodeActionSpace.IDLE
            best_score = 0.0
            best_rate = -1.0
            for action in obs.action_candidates[node_id]:
                if action == NodeActionSpace.IDLE:
                    score = 0.0
                else:
                    pair = tuple(sorted((node_id, action)))
                    score = demand.get(pair, 0.0)
                    if score <= 0.0:
                        node_scores[action] = 0.0
                        continue
                    edge_id = pair_to_edge.get(pair)
                    rate = edge_rate.get(edge_id, 0.0)
                    if score > best_score or (score == best_score and rate > best_rate):
                        best_score = score
                        best_rate = rate
                        best_action = action
                node_scores[action] = score
            actions[node_id] = best_action
            scores[node_id] = node_scores
        return actions, scores


def _pending_demand_by_pair(obs: GraphObservation) -> dict[tuple[str, str], float]:
    demand: dict[tuple[str, str], float] = {}
    for req in obs.state.pending_requests:
        pair = tuple(sorted((req.src_gs, req.dst_gs)))
        demand[pair] = demand.get(pair, 0.0) + req.amount
    return demand


def _pair_to_edge(obs: GraphObservation) -> dict[tuple[str, str], str]:
    pair_to_edge: dict[tuple[str, str], str] = {}
    for edge_id in obs.physical_edge_ids:
        edge = edge_id[2:] if edge_id.startswith("E_") else edge_id
        if "__" in edge:
            src, dst = edge.split("__", 1)
            pair_to_edge[tuple(sorted((src, dst)))] = edge_id
    return pair_to_edge
