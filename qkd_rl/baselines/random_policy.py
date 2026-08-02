from __future__ import annotations

import random

from qkd_rl.env.graph_builder import GraphObservation


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        actions: dict[str, str] = {}
        scores: dict[str, dict[str, float]] = {}
        for node_id in obs.node_ids:
            candidates = obs.action_candidates[node_id]
            mask = obs.action_masks[node_id]
            legal = [action for action, ok in zip(candidates, mask) if ok]
            action = self.rng.choice(legal)
            actions[node_id] = action
            scores[node_id] = {candidate: self.rng.random() for candidate in candidates}
        return actions, scores

