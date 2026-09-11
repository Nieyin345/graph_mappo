"""MAPPO policy wrapper: environment-compatible action sampling and PPO evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from qkd_rl.env.graph_builder import GraphObservation
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic


@dataclass
class PolicyStep:
    actions: dict[str, tuple[str, str]]
    action_scores: dict[str, dict[str, float]]
    log_probs: dict[str, torch.Tensor]
    entropies: dict[str, torch.Tensor]
    value: torch.Tensor
    # Log probability / entropy of the whole sampled matching action. These
    # are the PPO policy terms; node-level dictionaries below are kept for
    # compatibility with older tests/callers.
    joint_log_prob: torch.Tensor
    joint_entropy: torch.Tensor
    # Raw edge-scorer outputs (edge_id -> float) used by the priority-matching
    # resolver. The model computes one global score per physical edge, so
    # these are comparable across nodes (unlike per-node log probabilities).
    edge_scores: dict[str, float] | None = None
    # Ordered list of directed arcs ``(src, dst)`` selected by the sequential
    # global-matching sampler. The order is part of the sampled action: PPO
    # recomputes the joint log probability in this exact order during update.
    matched_edges: list[tuple[str, str]] | None = None


class MAPPOPolicy:
    def __init__(self, model: GraphMAPPOActorCritic, device: torch.device | str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self._edge_endpoints: dict[str, tuple[str, str]] | None = None

    def _endpoints(self) -> dict[str, tuple[str, str]]:
        if self._edge_endpoints is None:
            self._edge_endpoints = {
                edge.edge_id: (edge.src, edge.dst)
                for edge in self.model.action_space.edges
            }
        return self._edge_endpoints

    def _sample_matching(
        self,
        edge_scores: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[dict[str, tuple[str, str]], dict[str, dict[str, float]], list[tuple[str, str]], torch.Tensor, torch.Tensor]:
        """Sample one global directed matching by sequentially picking arcs.

        Each undirected pair contributes two directed arcs ``(src->dst)`` and
        ``(dst->src)``, both scored by the pair edge score. At every step the
        remaining feasible arcs (its transmitter not yet used, its receiver not
        yet used, and the pair not already matched in either direction — 对端
        不同) are scored by the actor's edge scorer and one is sampled from
        their softmax. Returns the directed ``matched_edges`` and per-node
        ``(tx_target, rx_source)`` actions.
        """
        endpoints = self._endpoints()
        edge_ids_list = list(edge_scores.keys())
        if not edge_ids_list:
            joint_lp = torch.zeros((), dtype=torch.float32, device=self.device)
            joint_entropy = torch.zeros((), dtype=torch.float32, device=self.device)
            return {}, {}, [], joint_lp, joint_entropy
        temperature = float(self.model.actor.temperature)
        arcs: list[tuple[str, str]] = []
        arc_edge: list[str] = []
        for edge_id in edge_ids_list:
            src, dst = endpoints[edge_id]
            arcs.append((src, dst))
            arc_edge.append(edge_id)
            arcs.append((dst, src))
            arc_edge.append(edge_id)
        raw = torch.stack([edge_scores[edge_id] for edge_id in edge_ids_list])
        if temperature != 1.0:
            raw = raw / temperature
        # One raw score per undirected pair; every pair maps to two arc ids.
        pair_score_np = raw.detach().cpu().numpy()
        n_arcs = len(arcs)
        scores_np = np.empty(n_arcs, dtype=np.float64)
        for i, edge_id in enumerate(arc_edge):
            pair_pos = edge_ids_list.index(edge_id)
            scores_np[i] = pair_score_np[pair_pos]
        if deterministic:
            gumbel_np = None
        else:
            u = torch.rand(n_arcs, dtype=raw.dtype, device=raw.device).clamp_min(torch.finfo(raw.dtype).tiny)
            gumbel_np = (-torch.log(-torch.log(u))).cpu().numpy()
        remaining = set(range(n_arcs))
        used_tx: set[str] = set()
        used_rx: set[str] = set()
        used_pair: set[tuple[str, str]] = set()
        matched_edges: list[tuple[str, str]] = []
        joint_lp = 0.0
        joint_entropy = 0.0
        while remaining:
            avail = []
            for i in remaining:
                src, dst = arcs[i]
                if src in used_tx or dst in used_rx:
                    continue
                if (src, dst) in used_pair or (dst, src) in used_pair:
                    continue
                avail.append(i)
            if not avail:
                break
            s = scores_np[avail]
            s_max = float(s.max())
            logp = s - (s_max + float(np.log(np.exp(s - s_max).sum())))
            if deterministic:
                k = int(np.argmax(s))
            else:
                k = int(np.argmax(s + gumbel_np[avail]))
            joint_lp += float(logp[k])
            p = np.exp(logp)
            joint_entropy -= float((p * logp).sum())
            sel = avail[k]
            src, dst = arcs[sel]
            matched_edges.append((src, dst))
            used_tx.add(src)
            used_rx.add(dst)
            pair = (src, dst) if src < dst else (dst, src)
            used_pair.add(pair)
            remaining.remove(sel)
        n_decisions = max(1, len(matched_edges))
        joint_entropy = joint_entropy / n_decisions
        joint_lp_t = torch.tensor(joint_lp, dtype=torch.float32, device=self.device)
        joint_entropy_t = torch.tensor(joint_entropy, dtype=torch.float32, device=self.device)

        return self._matching_to_actions(matched_edges, edge_scores), matched_edges, joint_lp_t, joint_entropy_t

    def _matching_to_actions(
        self,
        matched_edges: list[tuple[str, str]],
        edge_scores: dict[str, torch.Tensor],
    ) -> tuple[dict[str, tuple[str, str]], dict[str, dict[str, float]]]:
        """Derive per-node ``(tx_target, rx_source)`` actions from matched arcs."""
        actions: dict[str, tuple[str, str]] = {}
        action_scores: dict[str, dict[str, float]] = {}
        tx_of: dict[str, str] = {}
        rx_of: dict[str, str] = {}
        for node_id in self.model.action_space.node_ids:
            actions[node_id] = (self.model.action_space.IDLE, self.model.action_space.IDLE)
            action_scores[node_id] = {self.model.action_space.IDLE: 0.0}
        for src, dst in matched_edges:
            tx_of[src] = dst
            rx_of[dst] = src
            edge_id = self.model.action_space.arc_to_edge(src, dst)
            score = float(edge_scores.get(edge_id, torch.zeros((), dtype=torch.float32, device=self.device)).detach().cpu())
            action_scores[src][dst] = score
            action_scores[dst][src] = score
        for node_id in self.model.action_space.node_ids:
            tx = tx_of.get(node_id, self.model.action_space.IDLE)
            rx = rx_of.get(node_id, self.model.action_space.IDLE)
            actions[node_id] = (tx, rx)
        return actions, action_scores

    @staticmethod
    def _fill_node_tensors(
        node_ids: list[str],
        joint_lp: torch.Tensor,
        joint_entropy: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Repeat the joint matching log prob / entropy per node.

        The trainer averages the per-node PPO ratios; because every node of
        one step shares the same joint scalar, that average is exactly the
        joint-action ratio, which is what a global matching policy needs.
        Gradients are intentionally preserved here: PPO evaluation must back-
        propagate through the joint log probability into the actor. Rollout
        storage detaches these tensors when writing the RolloutStep.
        """
        return (
            {node_id: joint_lp for node_id in node_ids},
            {node_id: joint_entropy for node_id in node_ids},
        )

    def act(self, obs: GraphObservation, deterministic: bool = False) -> PolicyStep:
        """Sample a global matching action for the current observation."""
        output = self.model(obs, self.device, build_logits_dict=False)
        node_ids = output.logits_node_order if output.logits_node_order is not None else list(output.logits.keys())
        if not node_ids:
            return PolicyStep(
                actions={},
                action_scores={},
                log_probs={},
                entropies={},
                value=output.value,
                joint_log_prob=torch.zeros((), dtype=torch.float32, device=self.device),
                joint_entropy=torch.zeros((), dtype=torch.float32, device=self.device),
                edge_scores=None,
                matched_edges=[],
            )
        edge_scores = output.edge_scores or {}
        actions, action_scores, matched_edges, joint_lp, joint_entropy = self._sample_matching(
            edge_scores,
            deterministic=deterministic,
        )
        log_probs, entropies = self._fill_node_tensors(node_ids, joint_lp, joint_entropy)
        return PolicyStep(
            actions=actions,
            action_scores=action_scores,
            log_probs=log_probs,
            entropies=entropies,
            value=output.value,
            joint_log_prob=joint_lp,
            joint_entropy=joint_entropy,
            edge_scores=(
                {edge_id: float(score.detach().cpu()) for edge_id, score in edge_scores.items()}
                if edge_scores
                else None
            ),
            matched_edges=matched_edges,
        )

    def act_batched(
        self,
        obs_list: list[GraphObservation],
        deterministic: bool = False,
    ) -> list[PolicyStep]:
        """Sample global matching actions for many graphs with one model forward."""
        outputs = self.model.batched_forward(obs_list, self.device)
        values = outputs.values
        edge_score_maps = outputs.edge_score_maps

        steps: list[PolicyStep] = []
        for gi, (obs, value) in enumerate(zip(obs_list, values)):
            node_order = list(obs.node_ids)
            n = len(node_order)
            edge_scores = edge_score_maps[gi] or {}
            if n == 0:
                steps.append(
                    PolicyStep(
                        actions={},
                        action_scores={},
                        log_probs={},
                        entropies={},
                        value=value,
                        joint_log_prob=torch.zeros((), dtype=torch.float32, device=self.device),
                        joint_entropy=torch.zeros((), dtype=torch.float32, device=self.device),
                        edge_scores=None,
                        matched_edges=[],
                    )
                )
                continue
            actions, action_scores, matched_edges, joint_lp, joint_entropy = self._sample_matching(
                edge_scores,
                deterministic=deterministic,
            )
            log_probs, entropies = self._fill_node_tensors(node_order, joint_lp, joint_entropy)
            steps.append(
                PolicyStep(
                    actions=actions,
                    action_scores=action_scores,
                    log_probs=log_probs,
                    entropies=entropies,
                    value=value,
                    joint_log_prob=joint_lp,
                    joint_entropy=joint_entropy,
                    edge_scores=(
                        {edge_id: float(score.detach().cpu()) for edge_id, score in edge_scores.items()}
                        if edge_scores
                        else None
                    ),
                    matched_edges=matched_edges,
                )
            )
        return steps

    def evaluate_actions(
        self,
        obs: GraphObservation,
        actions: dict[str, tuple[str, str]],
        matched_edges: list[tuple[str, str]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        """Recompute the joint matching log prob / entropy and global value.

        Used by the PPO update: the returned log probs / value carry gradients
        so actor and critic losses can be back-propagated.

        ``matched_edges`` is REQUIRED: the joint log prob is only defined for
        the exact sampled edge sequence (it depends on the order), which cannot
        be reconstructed from the actions dict alone.
        """
        output = self.model(obs, self.device, build_logits_dict=False)
        node_ids = output.logits_node_order if output.logits_node_order is not None else list(output.logits.keys())
        if not node_ids:
            return {}, {}, output.value
        edge_scores = output.edge_scores or {}
        joint_lp, joint_entropy = self._matching_log_prob_entropy(edge_scores, matched_edges)
        log_probs, entropies = self._fill_node_tensors(node_ids, joint_lp, joint_entropy)
        return log_probs, entropies, output.value


    def evaluate_actions_batched(
        self,
        obs_list: list[GraphObservation],
        actions_list: list[dict[str, tuple[str, str]]],
        matched_edges_list: list[list[tuple[str, str]]],
    ) -> list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]]:
        """Batched PPO evaluation: one block-diagonal model forward for many
        graphs instead of one forward per step. Per-graph math is identical to
        ``evaluate_actions``; returns (log_probs, entropies, value) per obs."""
        outputs = self.model.batched_forward(obs_list, self.device)
        results: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]] = []
        for i, (obs, value) in enumerate(zip(
            obs_list,
            outputs.values,
        )):
            node_order = list(obs.node_ids)
            if not node_order:
                results.append(({}, {}, value))
                continue
            edge_scores = outputs.edge_score_maps[i] or {}
            joint_lp, joint_entropy = self._matching_log_prob_entropy(edge_scores, matched_edges_list[i])
            log_probs, entropies = self._fill_node_tensors(node_order, joint_lp, joint_entropy)
            results.append((log_probs, entropies, value))
        return results

    def log_prob_entropy_for_matching(
        self,
        edge_scores: dict[str, float] | None,
        matched_edges: list[tuple[str, str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a stored executed matching from detached rollout scores."""
        if not edge_scores:
            return (
                torch.zeros((), dtype=torch.float32, device=self.device),
                torch.zeros((), dtype=torch.float32, device=self.device),
            )
        edge_score_tensors = {
            edge_id: torch.tensor(float(score), dtype=torch.float32, device=self.device)
            for edge_id, score in edge_scores.items()
        }
        return self._matching_log_prob_entropy(edge_score_tensors, matched_edges)

    def _matching_log_prob_entropy(
        self,
        edge_scores: dict[str, torch.Tensor],
        matched_edges: list[tuple[str, str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint log prob and per-decision average entropy of a stored directed
        matching, replaying the dual-port feasibility rules ``Tx-out<=1``,
        ``Rx-in<=1`` and 对端不同 (a pair matched once in either direction)."""
        endpoints = self._endpoints()
        # Candidate directed arcs derived from the scored undirected pairs.
        arcs: list[tuple[str, str]] = []
        arc_edge: list[str] = []
        for edge_id in edge_scores:
            src, dst = endpoints[edge_id]
            arcs.append((src, dst))
            arc_edge.append(edge_id)
            arcs.append((dst, src))
            arc_edge.append(edge_id)
        temp_scores = {
            edge_id: score / float(self.model.actor.temperature)
            if float(self.model.actor.temperature) != 1.0
            else score
            for edge_id, score in edge_scores.items()
        }
        used_tx: set[str] = set()
        used_rx: set[str] = set()
        used_pair: set[tuple[str, str]] = set()
        joint_lp = torch.zeros((), dtype=torch.float32, device=self.device)
        joint_entropy = torch.zeros((), dtype=torch.float32, device=self.device)
        for arc in matched_edges:
            cur_src = arc[0]
            cur_dst = arc[1]
            pair = (cur_src, cur_dst) if cur_src < cur_dst else (cur_dst, cur_src)
            avail = sorted(
                i
                for i in range(len(arcs))
                if arcs[i][0] not in used_tx
                and arcs[i][1] not in used_rx
                and (arcs[i][0], arcs[i][1]) not in used_pair
                and (arcs[i][1], arcs[i][0]) not in used_pair
            )
            raw_scores = torch.stack([temp_scores[arc_edge[i]] for i in avail])
            logp = torch.log_softmax(raw_scores, dim=0)
            joint_entropy = joint_entropy - (logp.exp() * logp).sum()
            try:
                pos = avail.index(arc)
            except ValueError as exc:
                raise ValueError(f"Stored matching arc {arc!r} is not available for evaluation.") from exc
            joint_lp = joint_lp + logp[pos]
            used_tx.add(cur_src)
            used_rx.add(cur_dst)
            used_pair.add(pair)
        n_decisions = max(1, len(matched_edges))
        joint_entropy = joint_entropy / n_decisions
        return joint_lp, joint_entropy


def _masked_log_prob_entropy(
    logits: torch.Tensor, idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log probs and entropies for a masked categorical, ``-inf`` padding safe."""
    logp = torch.log_softmax(logits, dim=-1)
    log_probs = logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    safe = torch.where(torch.isfinite(logits), logp, torch.zeros_like(logp))
    entropies = -(safe.exp() * safe).sum(-1)
    return log_probs, entropies
