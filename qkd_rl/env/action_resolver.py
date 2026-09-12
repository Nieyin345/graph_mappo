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
    # The directed arcs ``(src, dst)`` actually matched this slot. The policy
    # verifies / reconstructs its sampled matching from these; key generation
    # uses ``activated_edges`` (undirected pair ids) so routing/QKP are unchanged.
    matched_arcs: list[tuple[str, str]] = None


class ActionResolver:
    """Resolve per-node ``(tx_target, rx_source)`` proposals into a feasible set
    of directed arcs subject to the dual-port physical constraints:

    - ``Tx``-out <= 1 per node (a node transmits to at most one peer);
    - ``Rx``-in <= 1 per node (a node receives from at most one peer);
    - a pair never carries both ``u->v`` and ``v->u`` in the same slot.

    ``activated_edges`` reports the matched arcs as deduplicated **undirected
    pair edge ids** so key generation / routing / QKP read the same ids as
    before; ``matched_arcs`` carries the directional detail for the policy.
    """

    def __init__(self, action_space: NodeActionSpace, config: dict):
        self.action_space = action_space
        self.config = config
        self.mode = config["mode"]
        # action -> candidate index per node, so _find_illegal avoids an O(n)
        # list.index() per submitted action every step. Candidates (neighbors +
        # idle) are shared by the tx and rx options.
        self._action_index: dict[str, dict[str, int]] = {
            node_id: {
                action: i
                for i, action in enumerate(self.action_space.candidates_for_node(node_id))
            }
            for node_id in self.action_space.node_ids
        }
        self._edge_by_id = {edge.edge_id: edge for edge in self.action_space.edges}

    # ------------------------------------------------------------------ resolve
    def resolve(
        self,
        actions: dict[str, tuple[str, str]],
        env_state: EnvState,
        masks: dict[str, list[bool]],
        action_scores: dict[str, dict[str, float]] | None = None,
        edge_scores: dict[tuple[str, str], float] | None = None,
    ) -> ResolvedAction:
        # Legacy single-target actions (``{node: target}``, produced by the
        # greedy baselines) are normalized to the dual-port ``(tx, rx)`` form:
        # a plain target proposes that link for both the Tx and Rx port.
        if actions and isinstance(next(iter(actions.values())), str):
            actions = {node: (target, target) for node, target in actions.items()}
        illegal = self._find_illegal(actions, masks)
        valid_actions = {node: action for node, action in actions.items() if node not in illegal}
        if self.mode == "mutual_choice":
            matched_arcs = self._resolve_mutual_choice(valid_actions)
        elif self.mode == "priority_matching":
            matched_arcs = self._resolve_priority_matching(
                valid_actions,
                action_scores or {},
                env_state,
                edge_scores=edge_scores,
            )
        elif self.mode == "greedy_rate_matching":
            matched_arcs = self._resolve_greedy_rate_matching(env_state)
        elif self.mode == "max_weight_matching":
            matched_arcs = self._resolve_max_weight_matching(
                masks,
                action_scores or {},
                env_state,
                edge_scores=edge_scores,
            )
        else:
            raise ValueError(f"Unknown action resolver mode: {self.mode}")
        matched_arcs = sorted(set(matched_arcs))
        activated = self.action_space.arcs_to_edges(matched_arcs)
        activated_set = set(activated)
        rejected: dict[str, str] = {}
        conflict = 0
        for u, (tx, rx) in valid_actions.items():
            dropped = set()
            for v in {tx, rx}:  # dedup: (B, B) is a single proposed edge
                if v == NodeActionSpace.IDLE or v == u:
                    continue
                eid = self.action_space.arc_to_edge(u, v)
                if eid is not None and eid not in activated_set:
                    conflict += 1
                    dropped.add(v)
            if dropped:
                rejected[u] = (tuple(sorted(dropped)) if len(dropped) > 1 else next(iter(dropped)))
        return ResolvedAction(
            activated_edges=activated,
            rejected_actions=rejected,
            illegal_actions=illegal,
            conflict_count=conflict,
            matched_arcs=matched_arcs,
        )

    def _find_illegal(self, actions: dict[str, tuple[str, str]], masks: dict[str, list[bool]]) -> dict[str, str]:
        illegal: dict[str, str] = {}
        for node_id, action in actions.items():
            tx, rx = action
            if self._option_illegal(node_id, tx, masks) or self._option_illegal(node_id, rx, masks):
                illegal[node_id] = action
        return illegal

    def _option_illegal(self, node_id: str, option: str, masks: dict[str, list[bool]]) -> bool:
        if option == NodeActionSpace.IDLE:
            return False
        action_idx = self._action_index[node_id].get(option)
        if action_idx is None:
            return True
        return not masks[node_id][action_idx]

    # ------------------------------------------------------------------ mutual choice
    def _resolve_mutual_choice(self, actions: dict[str, tuple[str, str]]) -> list[tuple[str, str]]:
        """Directed arcs agreed by both endpoints: ``u->v`` is active iff ``u``'s
        Tx targets ``v`` and ``v``'s Rx accepts from ``u``."""
        tx_of: dict[str, str] = {}
        rx_of: dict[str, str] = {}
        for u, (tx, rx) in actions.items():
            if tx != NodeActionSpace.IDLE:
                tx_of[u] = tx
            if rx != NodeActionSpace.IDLE:
                rx_of[u] = rx
        agreed: list[tuple[str, str]] = []
        for u, v in tx_of.items():
            if rx_of.get(v) == u and self.action_space.is_neighbor(u, v):
                agreed.append((u, v))
        # Tx-out<=1 / Rx-in<=1 hold automatically (one tx / one rx per node).
        # Only a mirrored pair needs disambiguation (对端不同): keep one.
        result: dict[tuple[str, str], tuple[str, str]] = {}
        for arc in agreed:
            key = self.action_space.pair_key(*arc)
            if key not in result or arc < result[key]:
                result[key] = arc
        return sorted(result.values())

    # ------------------------------------------------------------------ priority
    def _resolve_priority_matching(
        self,
        actions: dict[str, tuple[str, str]],
        action_scores: dict[str, dict[str, float]],
        env_state: EnvState,
        edge_scores: dict[tuple[str, str], float] | None = None,
    ) -> list[tuple[str, str]]:
        score_merge = self.config.get("score_merge", "mean")
        score_source = self.config.get("score_source", "edge")
        use_edge_scores = score_source == "edge" and edge_scores is not None
        candidates: list[tuple[float, str, str, str]] = []  # (score, edge_id, src, dst)
        for u, (tx, rx) in actions.items():
            for v in (tx, rx):
                if v == NodeActionSpace.IDLE or v == u or not self.action_space.is_neighbor(u, v):
                    continue
                edge_id = self.action_space.arc_to_edge(u, v)
                if edge_id is None:
                    continue
                if use_edge_scores:
                    # Directed arc scores: u -> v uses its own score, not the
                    # shared undirected pair score.
                    score = edge_scores.get((u, v), 0.0)
                else:
                    if score_merge not in ("mean", "max"):
                        raise ValueError(f"Unsupported action_resolver.score_merge: {score_merge!r}")
                    self_score = action_scores.get(u, {}).get(v, 0.0)
                    peer_score = action_scores.get(v, {}).get(u, 0.0)
                    score = 0.5 * (self_score + peer_score) if score_merge == "mean" else max(self_score, peer_score)
                candidates.append((score, edge_id, u, v))
        candidates = self._sort_candidates(candidates, env_state)
        return self._greedy_match(candidates)

    def _resolve_greedy_rate_matching(self, env_state: EnvState) -> list[tuple[str, str]]:
        candidates: list[tuple[float, str, str]] = []
        for edge_id, window in env_state.edge_windows.items():
            if not window.available[0]:
                continue
            edge = self._edge_by_id[edge_id]
            candidates.append((window.rates[0], edge_id, edge.src, edge.dst))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return self._greedy_match(candidates)

    def _sort_candidates(
        self, candidates: list[tuple[float, str, str, str]], env_state: EnvState
    ) -> list[tuple[float, str, str, str]]:
        tie_break = self.config.get("tie_break", "rate_then_edge_id")
        if tie_break == "rate_then_edge_id":
            rate_map: dict[str, float] = {}
            for item in candidates:
                window = env_state.edge_windows.get(item[1])
                rate_map[item[1]] = float(window.rates[0]) if window is not None else 0.0
            return sorted(candidates, key=lambda item: (-item[0], -rate_map[item[1]], item[1]))
        return sorted(candidates, key=lambda item: (-item[0], item[1]))

    # ------------------------------------------------------------------ max weight
    def _resolve_max_weight_matching(
        self,
        masks: dict[str, list[bool]],
        action_scores: dict[str, dict[str, float]],
        env_state: EnvState,
        edge_scores: dict[tuple[str, str], float] | None = None,
    ) -> list[tuple[str, str]]:
        tie_break = self.config.get("tie_break", "rate_then_edge_id")
        use_edge_scores = self.config.get("score_source", "edge") == "edge" and edge_scores is not None
        score_merge = self.config.get("score_merge", "mean")

        directed: list[tuple[float, float, str, str]] = []
        for edge in self.action_space.edges:
            if not self._edge_is_legal(edge.src, edge.dst, masks):
                continue
            eid = edge.edge_id
            if use_edge_scores:
                base_uv = float(edge_scores.get((edge.src, edge.dst), 0.0))
                base_vu = float(edge_scores.get((edge.dst, edge.src), 0.0))
            else:
                base_uv = base_vu = self._merge_action_scores(edge.src, edge.dst, action_scores, score_merge)
            window = env_state.edge_windows.get(eid)
            rate = float(window.rates[0]) if window is not None else 0.0
            directed.append((base_uv, rate, edge.src, edge.dst))
            directed.append((base_vu, rate, edge.dst, edge.src))
        directed = self._prune_matching_candidates(directed)
        if not directed:
            return []

        graph = nx.Graph()
        for src in self.action_space.node_ids:
            graph.add_node(("L", src))
            graph.add_node(("R", src))
        rank = {f"{s}->{t}": i for i, (_b, _r, s, t) in enumerate(sorted(directed, key=lambda item: item[3]))}
        max_rate = max((float(item[1]) for item in directed), default=1.0) or 1.0
        for base, rate, u, v in directed:
            data = dict(base=float(base), rate=float(rate), rank=rank[f"{u}->{v}"], arc=(u, v))
            w = self._positive_weight(float(base))
            if tie_break == "rate_then_edge_id":
                # Perturbation must be large enough to survive float64 rounding
                # next to ``w`` (else equal-score arcs resolve arbitrarily) but
                # stay orders of magnitude below any real score difference.
                data["weight"] = w + max(1.0e-9, abs(w) * 1.0e-6) * (rate / max_rate + data["rank"] * 1.0e-6)
            else:
                data["weight"] = w + data["rank"] * 1.0e-6
            graph.add_edge(("L", u), ("R", v), **data)
        matching = nx.max_weight_matching(graph, maxcardinality=False, weight="weight")
        # 对端不同: the bipartite graph models A->C and C->A with different left
        # nodes, so both could be matched; a physical pair has one light path per
        # slot, so keep only the higher-weight direction per undirected pair.
        best_by_pair: dict[tuple[str, str], tuple[tuple[str, str], float]] = {}
        for e1, e2 in matching:
            data = graph.edges[e1, e2]
            arc = data["arc"]
            key = self.action_space.pair_key(*arc)
            w = float(data["weight"])
            if key not in best_by_pair or w > best_by_pair[key][1]:
                best_by_pair[key] = (arc, w)
        arcs = [best_by_pair[key][0] for key in best_by_pair]
        return sorted(arcs)

    def _prune_matching_candidates(
        self, directed: list[tuple[float, float, str, str]]
    ) -> list[tuple[float, float, str, str]]:
        max_per_node = int(self.config.get("max_candidates_per_node", 0) or 0)
        max_edges = int(self.config.get("max_candidate_edges", 0) or 0)
        if max_per_node <= 0 and (max_edges <= 0 or len(directed) <= max_edges):
            return directed
        ranked = sorted(directed, key=lambda item: (-item[0], -item[1], item[3]))
        keep: set[tuple[str, str]] = set()
        kept_tx: dict[str, int] = {}
        kept_rx: dict[str, int] = {}
        for _score, _rate, u, v in ranked:
            if max_per_node > 0 and (
                kept_tx.get(u, 0) >= max_per_node or kept_rx.get(v, 0) >= max_per_node
            ):
                continue
            if max_edges > 0 and len(keep) >= max_edges:
                break
            keep.add((u, v))
            kept_tx[u] = kept_tx.get(u, 0) + 1
            kept_rx[v] = kept_rx.get(v, 0) + 1
        return [item for item in directed if (item[2], item[3]) in keep]

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
        if score >= 0.0:
            return score + math.log1p(math.exp(-score))
        return math.log1p(math.exp(score))

    def _greedy_match(self, candidates: list[tuple[float, str, str, str]]) -> list[tuple[str, str]]:
        """Greedy over (score, edge_id, src, dst) enforcing used_tx / used_rx /
        used_pair (对端不同)."""
        used_tx: set[str] = set()
        used_rx: set[str] = set()
        used_pair: set[tuple[str, str]] = set()
        activated: list[tuple[str, str]] = []
        for _score, _edge_id, src, dst in sorted(candidates, key=lambda item: (-item[0], item[3])):
            if src in used_tx or dst in used_rx:
                continue
            pair = self.action_space.pair_key(src, dst)
            if pair in used_pair:
                continue
            activated.append((src, dst))
            used_tx.add(src)
            used_rx.add(dst)
            used_pair.add(pair)
        return activated