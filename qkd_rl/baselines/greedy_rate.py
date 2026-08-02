from __future__ import annotations

from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.graph_builder import GraphObservation


class GreedyRatePolicy:
    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        edge_rate = {edge_id: obs.state.edge_windows[edge_id].rates[0] for edge_id in obs.physical_edge_ids}
        pair_to_edge: dict[tuple[str, str], str] = {}
        for edge_id in obs.physical_edge_ids:
            edge = edge_id[2:] if edge_id.startswith("E_") else edge_id
            if "__" in edge:
                src, dst = edge.split("__", 1)
                pair_to_edge[tuple(sorted((src, dst)))] = edge_id

        actions: dict[str, str] = {}
        scores: dict[str, dict[str, float]] = {}
        for node_id in obs.node_ids:
            candidates = obs.action_candidates[node_id]
            mask = obs.action_masks[node_id]
            node_scores: dict[str, float] = {}
            best_action = NodeActionSpace.IDLE
            best_score = -1.0
            for action, ok in zip(candidates, mask):
                if action == NodeActionSpace.IDLE:
                    score = 0.0
                else:
                    edge_id = pair_to_edge.get(tuple(sorted((node_id, action))))
                    score = edge_rate.get(edge_id, 0.0) if ok else -1.0
                node_scores[action] = score
                if ok and score > best_score:
                    best_score = score
                    best_action = action
            actions[node_id] = best_action
            scores[node_id] = node_scores
        return actions, scores
