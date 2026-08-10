from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx

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
        # action -> candidate index per node, so _find_illegal avoids an O(n)
        # list.index() per submitted action every step.
        self._action_index: dict[str, dict[str, int]] = {
            node_id: {action: i for i, action in enumerate(self.action_space.candidates_for_node(node_id))}
            for node_id in self.action_space.node_ids
        }
        self._edge_by_id = {edge.edge_id: edge for edge in self.action_space.edges}

    def resolve(
        self,
        actions: dict[str, str],
        env_state: EnvState,
        masks: dict[str, list[bool]],
        action_scores: dict[str, dict[str, float]] | None = None,
        edge_scores: dict[str, float] | None = None,
    ) -> ResolvedAction:
        illegal = self._find_illegal(actions, masks)
        valid_actions = {node: action for node, action in actions.items() if node not in illegal}
        if self.mode == "mutual_choice":
            activated = self._resolve_mutual_choice(valid_actions)
        elif self.mode == "priority_matching":
            activated = self._resolve_priority_matching(
                valid_actions,
                action_scores or {},
                env_state,
                edge_scores=edge_scores,
            )
        elif self.mode == "greedy_rate_matching":
            activated = self._resolve_greedy_rate_matching(env_state)
        elif self.mode == "max_weight_matching":
            activated = self._resolve_max_weight_matching(
                masks,
                action_scores or {},
                env_state,
                edge_scores=edge_scores,
            )
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
            conflict_count=len(set(intended_edges).symmetric_difference(set(activated))),
        )

    def _find_illegal(self, actions: dict[str, str], masks: dict[str, list[bool]]) -> dict[str, str]:
        illegal: dict[str, str] = {}
        for node_id, action in actions.items():
            action_idx = self._action_index[node_id].get(action)
            if action_idx is None or not masks[node_id][action_idx]:
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
        env_state: EnvState,
        edge_scores: dict[str, float] | None = None,
    ) -> list[str]:
        score_merge = self.config.get("score_merge", "mean")
        tie_break = self.config.get("tie_break", "rate_then_edge_id")
        score_source = self.config.get("score_source", "edge")
        if score_source not in ("edge", "action"):
            raise ValueError(f"Unsupported action_resolver.score_source: {score_source!r}")
        use_edge_scores = score_source == "edge" and edge_scores is not None
        if use_edge_scores:
            score_merge = "edge_score"
        if score_merge not in ("mean", "max", "edge_score"):
            raise ValueError(f"Unsupported action_resolver.score_merge: {score_merge!r}")
        if tie_break not in ("rate_then_edge_id", "edge_id"):
            raise ValueError(f"Unsupported action_resolver.tie_break: {tie_break!r}")
        candidates: list[tuple[float, str, str, str]] = []
        edge_ids: list[str] = []
        for node_id, action in actions.items():
            edge_id = self.action_space.action_to_edge(node_id, action)
            if edge_id is None:
                continue
            if use_edge_scores:
                # Raw actor edge score: one learned scalar per physical edge,
                # shared by both endpoints, so the greedy matching compares
                # edges on the same scale. Falls back to the action score only
                # when no edge scores were supplied (manual policies/tests).
                score = edge_scores.get(edge_id, 0.0)
            else:
                self_score = action_scores.get(node_id, {}).get(action, 0.0)
                peer_score = action_scores.get(action, {}).get(node_id, 0.0)
                if score_merge == "mean":
                    score = 0.5 * (self_score + peer_score)
                else:
                    score = max(self_score, peer_score)
            candidates.append((score, edge_id, node_id, action))
            edge_ids.append(edge_id)
        if tie_break == "rate_then_edge_id":
            windows = env_state.edge_windows
            if hasattr(windows, "rates0"):
                rate_map = dict(zip(edge_ids, (float(r) for r in windows.rates0(edge_ids))))
            else:
                rate_map = {edge_id: windows[edge_id].rates[0] for edge_id in edge_ids}
            candidates.sort(
                key=lambda item: (
                    -item[0],
                    -rate_map[item[1]],
                    item[1],
                )
            )
        else:
            candidates.sort(key=lambda item: (-item[0], item[1]))
        return self._greedy_match(candidates)

    def _resolve_greedy_rate_matching(self, env_state: EnvState) -> list[str]:
        candidates: list[tuple[float, str, str, str]] = []
        for edge_id, window in env_state.edge_windows.items():
            if not window.available[0]:
                continue
            edge = self._edge_by_id[edge_id]
            candidates.append((window.rates[0], edge_id, edge.src, edge.dst))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return self._greedy_match(candidates)

    def _resolve_max_weight_matching(
        self,
        masks: dict[str, list[bool]],
        action_scores: dict[str, dict[str, float]],
        env_state: EnvState,
        edge_scores: dict[str, float] | None = None,
    ) -> list[str]:
        score_merge = self.config.get("score_merge", "mean")
        tie_break = self.config.get("tie_break", "rate_then_edge_id")
        score_source = self.config.get("score_source", "edge")
        if score_source not in ("edge", "action"):
            raise ValueError(f"Unsupported action_resolver.score_source: {score_source!r}")
        use_edge_scores = score_source == "edge" and edge_scores is not None
        if score_merge not in ("mean", "max", "edge_score"):
            raise ValueError(f"Unsupported action_resolver.score_merge: {score_merge!r}")
        if tie_break not in ("rate_then_edge_id", "edge_id"):
            raise ValueError(f"Unsupported action_resolver.tie_break: {tie_break!r}")

        candidates: list[tuple[float, float, str, str, str]] = []
        edge_iter = (
            (self._edge_by_id[edge_id], float(score))
            for edge_id, score in (edge_scores or {}).items()
            if edge_id in self._edge_by_id
        ) if use_edge_scores else ((edge, None) for edge in self.action_space.edges)
        for edge, supplied_score in edge_iter:
            src = edge.src
            dst = edge.dst
            edge_id = edge.edge_id
            if not self._edge_is_legal(src, dst, masks):
                continue
            if use_edge_scores:
                base_score = float(supplied_score)
            else:
                base_score = self._merge_action_scores(src, dst, action_scores, score_merge)
            if tie_break == "rate_then_edge_id":
                window = env_state.edge_windows.get(edge_id)
                rate = float(window.rates[0]) if window is not None else 0.0
            else:
                rate = 0.0
            candidates.append((base_score, rate, edge_id, src, dst))

        candidates = self._prune_matching_candidates(candidates)
        edge_meta: dict[str, tuple[float, float]] = {
            edge_id: (base_score, rate)
            for base_score, rate, edge_id, _src, _dst in candidates
        }

        if not edge_meta:
            return []

        graph = nx.Graph()
        graph.add_nodes_from(self.action_space.node_ids)
        for base_score, _rate, edge_id, src, dst in candidates:
            graph.add_edge(src, dst, edge_id=edge_id, base_score=base_score)

        edge_rank = {edge_id: i for i, edge_id in enumerate(sorted(edge_meta))}
        max_abs = max((abs(score) for score, _rate in edge_meta.values()), default=1.0)
        eps = max(1.0e-12, max_abs * 1.0e-9)
        max_rate = max((abs(rate) for _score, rate in edge_meta.values()), default=1.0)
        max_rate = max(1.0, max_rate)
        for u, v, data in graph.edges(data=True):
            edge_id = data["edge_id"]
            base_weight = self._positive_weight(float(data["base_score"]))
            if tie_break == "rate_then_edge_id":
                rate_tie = edge_meta[edge_id][1] / max_rate
                tie_score = rate_tie + (
                    float(len(edge_rank) - edge_rank[edge_id]) / max(1.0, float(len(edge_rank)))
                ) * 1.0e-9
            else:
                tie_score = float(len(edge_rank) - edge_rank[edge_id]) * 1.0e-9
            data["weight"] = base_weight + eps * tie_score

        matching = nx.max_weight_matching(graph, maxcardinality=False, weight="weight")
        activated = [graph.edges[u, v]["edge_id"] for u, v in matching]
        return self._sorted_matching_edges(activated, edge_meta, tie_break)

    def _prune_matching_candidates(
        self,
        candidates: list[tuple[float, float, str, str, str]],
    ) -> list[tuple[float, float, str, str, str]]:
        max_per_node = int(self.config.get("max_candidates_per_node", 0) or 0)
        max_edges = int(self.config.get("max_candidate_edges", 0) or 0)
        if max_per_node <= 0 and (max_edges <= 0 or len(candidates) <= max_edges):
            return candidates

        ranked = sorted(candidates, key=lambda item: (-item[0], -item[1], item[2]))
        keep: set[str] = set()
        kept_by_node: dict[str, int] = {}
        for _score, _rate, edge_id, src, dst in ranked:
            if max_per_node > 0:
                if kept_by_node.get(src, 0) >= max_per_node:
                    continue
                if kept_by_node.get(dst, 0) >= max_per_node:
                    continue
            if max_edges > 0 and len(keep) >= max_edges:
                break
            keep.add(edge_id)
            kept_by_node[src] = kept_by_node.get(src, 0) + 1
            kept_by_node[dst] = kept_by_node.get(dst, 0) + 1
        return [item for item in candidates if item[2] in keep]

    def _edge_is_legal(self, src: str, dst: str, masks: dict[str, list[bool]]) -> bool:
        src_idx = self._action_index[src].get(dst)
        dst_idx = self._action_index[dst].get(src)
        if src_idx is None or dst_idx is None:
            return False
        return bool(masks[src][src_idx]) and bool(masks[dst][dst_idx])

    @staticmethod
    def _merge_action_scores(
        src: str,
        dst: str,
        action_scores: dict[str, dict[str, float]],
        score_merge: str,
    ) -> float:
        self_score = action_scores.get(src, {}).get(dst, 0.0)
        peer_score = action_scores.get(dst, {}).get(src, 0.0)
        if score_merge == "mean":
            return 0.5 * (self_score + peer_score)
        return max(self_score, peer_score)

    @staticmethod
    def _positive_weight(score: float) -> float:
        """Map raw actor scores to positive matching weights.

        The actor emits unconstrained real-valued scores. NetworkX treats the
        empty matching (weight 0) as a legal solution, so a graph whose raw
        scores are all negative would otherwise collapse to no activated edges.
        A stable softplus keeps the ordering of edges while ensuring every legal
        edge contributes positive weight.
        """
        if score >= 0.0:
            return score + math.log1p(math.exp(-score))
        return math.log1p(math.exp(score))

    @staticmethod
    def _sorted_matching_edges(
        edge_ids: list[str],
        edge_meta: dict[str, tuple[float, float]],
        tie_break: str,
    ) -> list[str]:
        if tie_break == "rate_then_edge_id":
            return sorted(
                edge_ids,
                key=lambda edge_id: (
                    -edge_meta[edge_id][0],
                    -edge_meta[edge_id][1],
                    edge_id,
                ),
            )
        return sorted(
            edge_ids,
            key=lambda edge_id: (
                -edge_meta[edge_id][0],
                edge_id,
            ),
        )

    def _greedy_match(self, candidates: list[tuple[float, str, str, str]]) -> list[str]:
        used_nodes: set[str] = set()
        activated: list[str] = []
        for _score, edge_id, src, dst in candidates:
            if src in used_nodes or dst in used_nodes:
                continue
            activated.append(edge_id)
            used_nodes.update({src, dst})
        return activated

