"""MAPPO policy wrapper: environment-compatible action sampling and PPO evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qkd_rl.env.graph_builder import GraphObservation
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


@dataclass
class PolicyStep:
    actions: dict[str, str]
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
    # Ordered list of edges selected by the sequential global-matching
    # sampler. The order is part of the sampled action: PPO recomputes the
    # joint log probability in this exact order during the update.
    matched_edges: list[str] | None = None


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
    ) -> tuple[dict[str, str], dict[str, dict[str, float]], list[str], torch.Tensor, torch.Tensor]:
        """Sample one global matching by sequentially picking edges.

        At every step the remaining legal edges (those whose endpoints are
        still free) are scored by the actor's edge scorer and one edge is
        sampled from their softmax. The joint log probability is the sum over
        the selected sequence; the returned entropy is the per-decision
        average entropy so the entropy regularizer stays comparable across
        matching sizes instead of growing with every matched edge.
        """
        endpoints = self._endpoints()
        edge_ids = list(edge_scores.keys())
        remaining = set(edge_ids)
        used_nodes: set[str] = set()
        matched_edges: list[str] = []
        joint_lp = torch.zeros((), dtype=torch.float32, device=self.device)
        joint_entropy = torch.zeros((), dtype=torch.float32, device=self.device)
        while remaining:
            avail = sorted(
                edge_id
                for edge_id in remaining
                if endpoints[edge_id][0] not in used_nodes and endpoints[edge_id][1] not in used_nodes
            )
            if not avail:
                break
            raw_scores = torch.stack([edge_scores[edge_id] for edge_id in avail])
            temperature = float(self.model.actor.temperature)
            scores = raw_scores / temperature if temperature != 1.0 else raw_scores
            logp = torch.log_softmax(scores, dim=0)
            ent = -(logp.exp() * logp).sum()
            if deterministic:
                idx = torch.argmax(scores)
                selected = avail[int(idx)]
            else:
                u = torch.rand_like(scores).clamp_min(torch.finfo(scores.dtype).tiny)
                gumbel = -torch.log(-torch.log(u))
                idx = torch.argmax(scores + gumbel)
                selected = avail[int(idx)]
            joint_lp = joint_lp + logp[int(idx)]
            joint_entropy = joint_entropy + ent
            matched_edges.append(selected)
            used_nodes.update(endpoints[selected])
            remaining.remove(selected)
        n_decisions = max(1, len(matched_edges))
        joint_entropy = joint_entropy / n_decisions

        actions: dict[str, str] = {}
        action_scores: dict[str, dict[str, float]] = {}
        for node_id in self.model.action_space.node_ids:
            actions[node_id] = self.model.action_space.IDLE
            action_scores[node_id] = {self.model.action_space.IDLE: 0.0}
        for edge_id in matched_edges:
            src, dst = endpoints[edge_id]
            actions[src] = dst
            actions[dst] = src
            score = float(edge_scores[edge_id].detach().cpu())
            action_scores[src] = {dst: score}
            action_scores[dst] = {src: score}
        return actions, action_scores, matched_edges, joint_lp, joint_entropy

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
        actions: dict[str, str],
        matched_edges: list[str] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        """Recompute the joint matching log prob / entropy and global value.

        Used by the PPO update: the returned log probs / value carry gradients
        so actor and critic losses can be back-propagated.
        """
        output = self.model(obs, self.device, build_logits_dict=False)
        node_ids = output.logits_node_order if output.logits_node_order is not None else list(output.logits.keys())
        if not node_ids:
            return {}, {}, output.value
        edge_scores = output.edge_scores or {}
        if matched_edges is None:
            matched_edges = self._matched_edges_from_actions(actions)
        joint_lp, joint_entropy = self._matching_log_prob_entropy(edge_scores, matched_edges)
        log_probs, entropies = self._fill_node_tensors(node_ids, joint_lp, joint_entropy)
        return log_probs, entropies, output.value


    def evaluate_actions_batched(
        self,
        obs_list: list[GraphObservation],
        actions_list: list[dict[str, str]],
        matched_edges_list: list[list[str]] | None = None,
    ) -> list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]]:
        """Batched PPO evaluation: one block-diagonal model forward for many
        graphs instead of one forward per step. Per-graph math is identical to
        ``evaluate_actions``; returns (log_probs, entropies, value) per obs."""
        outputs = self.model.batched_forward(obs_list, self.device)
        results: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]] = []
        for i, (obs, actions, value) in enumerate(zip(
            obs_list,
            actions_list,
            outputs.values,
        )):
            node_order = list(obs.node_ids)
            if not node_order:
                results.append(({}, {}, value))
                continue
            edge_scores = outputs.edge_score_maps[i] or {}
            matched_edges = matched_edges_list[i] if matched_edges_list is not None else None
            if matched_edges is None:
                matched_edges = self._matched_edges_from_actions(actions)
            joint_lp, joint_entropy = self._matching_log_prob_entropy(edge_scores, matched_edges)
            log_probs, entropies = self._fill_node_tensors(node_order, joint_lp, joint_entropy)
            results.append((log_probs, entropies, value))
        return results

    def log_prob_entropy_for_matching(
        self,
        edge_scores: dict[str, float] | None,
        matched_edges: list[str],
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

    def _matched_edges_from_actions(self, actions: dict[str, str]) -> list[str]:
        """Reconstruct the matched edge sequence from mutual actions."""
        endpoints = self._endpoints()
        matched: list[str] = []
        for node_id, action in actions.items():
            if action == self.model.action_space.IDLE:
                continue
            if actions.get(action) != node_id:
                continue
            edge_id = self.model.action_space.edge_by_pair.get(tuple(sorted((node_id, action))))
            if edge_id is not None and edge_id not in matched:
                matched.append(edge_id)
        return matched

    def _matching_log_prob_entropy(
        self,
        edge_scores: dict[str, torch.Tensor],
        matched_edges: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint log prob and per-decision average entropy of a stored matching.

        The joint log probability is accumulated across every matched edge.
        Entropy is returned per decision so the regularizer does not grow
        with the number of matched edges (see ``_sample_matching``).
        """
        endpoints = self._endpoints()
        remaining = set(edge_scores.keys())
        used_nodes: set[str] = set()
        joint_lp = torch.zeros((), dtype=torch.float32, device=self.device)
        joint_entropy = torch.zeros((), dtype=torch.float32, device=self.device)
        for edge_id in matched_edges:
            avail = sorted(
                candidate
                for candidate in remaining
                if endpoints[candidate][0] not in used_nodes
                and endpoints[candidate][1] not in used_nodes
            )
            raw_scores = torch.stack([edge_scores[candidate] for candidate in avail])
            temperature = float(self.model.actor.temperature)
            scores = raw_scores / temperature if temperature != 1.0 else raw_scores
            logp = torch.log_softmax(scores, dim=0)
            joint_entropy = joint_entropy - (logp.exp() * logp).sum()
            try:
                pos = avail.index(edge_id)
            except ValueError as exc:
                raise ValueError(f"Stored matching edge {edge_id!r} is not available for evaluation.") from exc
            joint_lp = joint_lp + logp[pos]
            used_nodes.update(endpoints[edge_id])
            remaining.remove(edge_id)
        n_decisions = max(1, len(matched_edges))
        joint_entropy = joint_entropy / n_decisions
        return joint_lp, joint_entropy


def _masked_log_prob_entropy(
    logits: torch.Tensor, idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log probs and entropies for a masked categorical, ``-inf`` padding safe.

    ``log_softmax`` maps the ``-inf`` padding columns to ``-inf``, so
    ``exp(logp) * logp`` would be ``0 * -inf = NaN``. The padding positions are
    exactly where ``logits`` is non-finite, so those entries are zeroed before
    the entropy sum; log probs gathered at valid indices are unaffected.
    """
    logp = torch.log_softmax(logits, dim=-1)
    log_probs = logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    safe = torch.where(torch.isfinite(logits), logp, torch.zeros_like(logp))
    entropies = -(safe.exp() * safe).sum(-1)
    return log_probs, entropies


def _pad_logits(logits: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, int]:
    """Pad per-node logits to a common length with ``-inf``.

    ``-inf`` logits produce exact zero probabilities, so sampled indices,
    log probs, and entropies are unaffected by the padding.
    """
    node_ids = list(logits.keys())
    max_n = max(int(logits[node_id].shape[0]) for node_id in node_ids)
    padded = torch.full((len(node_ids), max_n), float("-inf"), dtype=torch.float32, device=device)
    for i, node_id in enumerate(node_ids):
        n = int(logits[node_id].shape[0])
        padded[i, :n] = logits[node_id]
    return padded, max_n
