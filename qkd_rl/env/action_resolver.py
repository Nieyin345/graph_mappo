from __future__ import annotations

from dataclasses import dataclass

from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.state import EnvState


@dataclass
class ResolvedAction:
    activated_edges: list[str]
    rejected_actions: dict[str, str]
    illegal_actions: dict[str, str]
    conflict_count: int


class ActionResolver:
    def __init__(self, action_space: NodeActionSpace, config: dict):
        self.action_space = action_space
        self.config = config
        self.mode = config["mode"]

    def resolve(
        self,
        actions: dict[str, str],
        env_state: EnvState,
        masks: dict[str, list[bool]],
        action_scores: dict[str, dict[str, float]] | None = None,
    ) -> ResolvedAction:
        illegal = self._find_illegal(actions, masks)
        valid_actions = {node: action for node, action in actions.items() if node not in illegal}
        if self.mode == "mutual_choice":
            activated = self._resolve_mutual_choice(valid_actions)
        elif self.mode == "priority_matching":
            activated = self._resolve_priority_matching(valid_actions, action_scores or {})
        elif self.mode == "greedy_rate_matching":
            activated = self._resolve_greedy_rate_matching(env_state)
        elif self.mode == "max_weight_matching":
            activated = self._resolve_priority_matching(valid_actions, action_scores or {})
        else:
            raise ValueError(f"Unknown action resolver mode: {self.mode}")
        intended_edges = {
            edge_id
            for node, action in valid_actions.items()
            if (edge_id := self.action_space.action_to_edge(node, action)) is not None
        }
        rejected = {
            node: action
            for node, action in valid_actions.items()
            if (edge_id := self.action_space.action_to_edge(node, action)) is not None and edge_id not in activated
        }
        return ResolvedAction(
            activated_edges=activated,
            rejected_actions=rejected,
            illegal_actions=illegal,
            conflict_count=max(0, len(intended_edges) - len(activated)),
        )

    def _find_illegal(self, actions: dict[str, str], masks: dict[str, list[bool]]) -> dict[str, str]:
        illegal: dict[str, str] = {}
        for node_id, action in actions.items():
            candidates = self.action_space.candidates_for_node(node_id)
            if action not in candidates or not masks[node_id][candidates.index(action)]:
                illegal[node_id] = action
        return illegal

    def _resolve_mutual_choice(self, actions: dict[str, str]) -> list[str]:
        activated: list[str] = []
        used: set[str] = set()
        for node_id, action in actions.items():
            if action == NodeActionSpace.IDLE or node_id in used:
                continue
            if actions.get(action) != node_id:
                continue
            edge_id = self.action_space.action_to_edge(node_id, action)
            if edge_id is not None:
                activated.append(edge_id)
                used.update({node_id, action})
        return activated

    def _resolve_priority_matching(
        self,
        actions: dict[str, str],
        action_scores: dict[str, dict[str, float]],
    ) -> list[str]:
        candidates: list[tuple[float, str, str, str]] = []
        for node_id, action in actions.items():
            edge_id = self.action_space.action_to_edge(node_id, action)
            if edge_id is None:
                continue
            score = action_scores.get(node_id, {}).get(action, 0.0)
            candidates.append((score, edge_id, node_id, action))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return self._greedy_match(candidates)

    def _resolve_greedy_rate_matching(self, env_state: EnvState) -> list[str]:
        candidates = [
            (window.rates[0], edge_id, edge.src, edge.dst)
            for edge_id, window in env_state.edge_windows.items()
            if window.available[0]
            for edge in [self._edge_from_id(edge_id)]
        ]
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return self._greedy_match(candidates)

    def _greedy_match(self, candidates: list[tuple[float, str, str, str]]) -> list[str]:
        used_nodes: set[str] = set()
        activated: list[str] = []
        for _score, edge_id, src, dst in candidates:
            if src in used_nodes or dst in used_nodes:
                continue
            activated.append(edge_id)
            used_nodes.update({src, dst})
        return activated

    def _edge_from_id(self, edge_id: str):
        for key, value in self.action_space.edge_by_pair.items():
            if value == edge_id:
                src, dst = key
                return type("EdgeView", (), {"src": src, "dst": dst})()
        raise KeyError(edge_id)

