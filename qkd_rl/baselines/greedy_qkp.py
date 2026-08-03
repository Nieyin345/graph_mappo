"""Greedy QKP baseline: activate the link whose QKP pool holds the most keys.

Independent of the training stack: consumes only :class:`GraphObservation`
fields and returns ``(actions, scores)`` like every baseline policy.
"""
from __future__ import annotations

from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.graph_builder import GraphObservation


class GreedyQKPPolicy:
    """Pick, per node, the neighbour whose edge has the highest stored key level.

    Uses the QKP snapshot of the current time step (keys already produced and
    parked in the link pool). Ties are broken by the current link rate so
    equal-level edges prefer faster links.
    """

    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        edge_level = dict(obs.state.qkp_snapshot)
        pair_to_edge = _pair_to_edge(obs)
        edge_rate = {edge_id: obs.state.edge_windows[edge_id].rates[0] for edge_id in obs.physical_edge_ids}

        actions: dict[str, str] = {}
        scores: dict[str, dict[str, float]] = {}
        for node_id in obs.node_ids:
            node_scores: dict[str, float] = {}
            best_action = NodeActionSpace.IDLE
            best_score = -1.0
            best_rate = -1.0
            for action in obs.action_candidates[node_id]:
                if action == NodeActionSpace.IDLE:
                    score = 0.0
                else:
                    pair = tuple(sorted((node_id, action)))
                    edge_id = pair_to_edge.get(pair)
                    score = edge_level.get(edge_id, 0.0)
                    rate = edge_rate.get(edge_id, 0.0)
                    if score > best_score or (score == best_score and rate > best_rate):
                        best_score = score
                        best_rate = rate
                        best_action = action
                node_scores[action] = score
            actions[node_id] = best_action
            scores[node_id] = node_scores
        return actions, scores


def _pair_to_edge(obs: GraphObservation) -> dict[tuple[str, str], str]:
    pair_to_edge: dict[tuple[str, str], str] = {}
    for edge_id in obs.physical_edge_ids:
        edge = edge_id[2:] if edge_id.startswith("E_") else edge_id
        if "__" in edge:
            src, dst = edge.split("__", 1)
            pair_to_edge[tuple(sorted((src, dst)))] = edge_id
    return pair_to_edge
