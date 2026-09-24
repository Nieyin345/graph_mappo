"""MAPPO policy wrapper: environment-compatible action sampling and PPO evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from qkd_rl.env.graph_builder import GraphObservation
from qkd_rl.model_zoo.v2.model_impl import GraphMAPPOActorCritic


@dataclass
class PolicyStep:
    actions: dict[str, tuple[str, str]]
    action_scores: dict[str, dict[str, float]]
    log_probs: dict[str, torch.Tensor]
    entropies: dict[str, torch.Tensor]
    value: torch.Tensor
    # Joint log probability of the ordered matching sequence (SUM of its
    # conditional log probabilities). The historical field name is retained
    # for callers; only entropy is averaged per decision.
    mean_log_prob: torch.Tensor
    mean_entropy: torch.Tensor
    # Raw edge-scorer outputs keyed by directed arc ``(src, dst)``, used by the
    # priority-matching resolver and stored for PPO evaluation. Directed keys
    # keep ``u -> v`` and ``v -> u`` scored separately.
    edge_scores: dict[tuple[str, str], float] | None = None
    # Ordered list of directed arcs ``(src, dst)`` selected by the sequential
    # global-matching sampler. The order is part of the sampled action: PPO
    # recomputes the joint log probability in this exact order during update.
    matched_edges: list[tuple[str, str]] | None = None


#: Resolver modes whose executed matching is a **deterministic function of the
#: edge scores**, not the policy's sampled sequence. For these the rollout must
#: (a) skip the ``expected_matched_edges`` consistency check in ``QKDEnv.step``
#: — the sampled sequence legitimately differs — and (b) recompute the PPO
#: log-prob from ``env.last_matched_arcs`` via
#: :meth:`MAPPOPolicy.log_prob_entropy_for_matching`, because optimising the
#: *sampled* sequence's log-prob would be a different (wrong) objective.
#:
#: ★ Single source of truth on purpose. Both the ``expected_matched_edges``
#:   argument and the downstream log-prob source key off this set; keeping them
#:   as two separate literal tuples is how they silently drift apart — the
#:   argument says "don't check" while the log-prob still comes from the
#:   sampled sequence, and nothing errors.
#:
#: ``mutual_choice`` is deliberately absent: the resolver *executes the sampled
#: sequence as-is*, so sampling and execution are identical by construction and
#: the check is a meaningful invariant.
DETERMINISTIC_RESOLVER_MODES = ("max_weight_matching", "priority_matching")


def _finalize_matching_log_prob(total, *, n_decisions: int, average: bool):
    """Apply the single matching log-prob reduction contract.

    ``total`` may be a Python/NumPy scalar during rollout or a torch scalar
    during differentiable replay. PPO always passes ``average=False`` because
    the ordered matching is one joint action. Behaviour cloning alone may opt
    into the historical per-decision mean weighting.
    """
    if not average:
        return total
    return total / max(1, int(n_decisions))


class MAPPOPolicy:
    def __init__(self, model: GraphMAPPOActorCritic, device: torch.device | str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        # Static topology cache for the hottest matching path. Arc encoding
        # does not change during a rollout on a fixed QKD topology, so avoid
        # rebuilding numpy arrays at every action sample.
        self._matching_cache = None
        # RNG for the Gumbel noise used by the matching samplers. It must be an
        # explicit generator rather than torch's global RNG: the rollout
        # workers are spawned interpreters whose global torch RNG is seeded
        # from OS entropy, never by the trainer's `torch.manual_seed`. With the
        # global RNG, two runs of the same config and seed took *different*
        # actions from the very first rollout -- the env seed still reproduced
        # the request stream, so the divergence looked like measurement noise
        # instead of the reproducibility bug it was. Set via
        # :meth:`sample_seed`; None falls back to the global RNG.
        self._sample_gen: torch.Generator | None = None
        # One generator per row for act_batched; see sample_seed.
        self._sample_gens_batched: list[torch.Generator] | None = None

    def set_sample_seed(self, seed: int | list[int]) -> None:
        """Seed the action-sampling RNG, making a rollout reproducible from it.

        The request stream is already pinned by the per-episode env seed; this
        pins the other half of the episode -- the policy's own exploration
        noise -- so identical (seed, weights) always yields identical episodes.

        Accepts an int for one sampler, or a list for ``act_batched`` (one
        generator per row, so a row's noise cannot depend on how many rows are
        batched or on the order the scheduler finished the others).

        Only the *stochastic* paths need this. Evaluations sample with
        ``deterministic=True``, which draws no Gumbel noise at all, so they are
        bit-reproducible whether or not this was called.
        """
        if isinstance(seed, (list, tuple)):
            gens = []
            for value in seed:
                gen = torch.Generator(device=self.device)
                gen.manual_seed(int(value))
                gens.append(gen)
            self._sample_gens_batched = gens
            return
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        self._sample_gen = gen

    def _get_matching_cache(self, arcs):
        """Build/reuse integer arc indices used by matching feasibility tests."""
        cache_key = tuple(arcs)
        if self._matching_cache is not None and self._matching_cache[0] == cache_key:
            return self._matching_cache[1]

        node_code = {node_id: i for i, node_id in enumerate(sorted({n for arc in arcs for n in arc}))}
        n_arcs = len(arcs)
        src_code = np.fromiter((node_code[arc[0]] for arc in arcs), dtype=np.int64, count=n_arcs)
        dst_code = np.fromiter((node_code[arc[1]] for arc in arcs), dtype=np.int64, count=n_arcs)
        pair_ids = {}
        pair_code = np.empty(n_arcs, dtype=np.int64)
        for i, arc in enumerate(arcs):
            s_c, d_c = node_code[arc[0]], node_code[arc[1]]
            key = (s_c, d_c) if s_c <= d_c else (d_c, s_c)
            pair_code[i] = pair_ids.setdefault(key, len(pair_ids))

        cache = (src_code, dst_code, pair_code)
        self._matching_cache = (cache_key, cache)
        return cache

    def _sample_matching(
        self,
        arc_scores: dict[tuple[str, str], torch.Tensor],
        deterministic: bool = False,
        build_scores: bool = True,
    ) -> tuple[
        tuple[dict[str, tuple[str, str]], dict[str, dict[str, float]]],
        list[tuple[str, str]],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample one global directed matching by sequentially picking arcs.

        The actor scores every legal directed arc ``(src -> dst)`` separately,
        so ``u -> v`` and ``v -> u`` rank differently. At every step the
        remaining feasible arcs (its transmitter not yet used, its receiver not
        yet used, and the pair not already matched in either direction — 对端
        不同) together with a STOP option are scored and one is sampled from
        their softmax; picking STOP terminates the matching, letting the policy
        express "activate nothing this slot" instead of being forced to fill
        every feasible port. Returns the directed ``matched_edges`` and per-node
        ``(tx_target, rx_source)`` actions.
        """
        arcs = list(arc_scores.keys())
        if not arcs:
            mean_lp = torch.zeros((), dtype=torch.float32, device=self.device)
            mean_entropy = torch.zeros((), dtype=torch.float32, device=self.device)
            return ({}, {}), [], mean_lp, mean_entropy
        temperature = float(self.model.actor.temperature)
        raw = torch.stack([arc_scores[arc] for arc in arcs])
        if temperature != 1.0:
            raw = raw / temperature
        stop_score = float(self.model.actor.stop_logit.detach())
        if temperature != 1.0:
            stop_score = stop_score / temperature
        n_arcs = len(arcs)
        scores_np = raw.detach().cpu().numpy()
        if deterministic:
            gumbel_np = None
        else:
            u = torch.rand(
                n_arcs + 1, dtype=raw.dtype, device=raw.device, generator=self._sample_gen
            ).clamp_min(torch.finfo(raw.dtype).tiny)
            gumbel_np = (-torch.log(-torch.log(u))).cpu().numpy()

        # Integer codes let the feasibility test be one vectorized mask update
        # per decision instead of a rescan of every arc. The dual-port rules are
        # "this node's Tx is taken", "this node's Rx is taken", and "this pair is
        # already matched in either direction"; all three are equality tests on
        # precomputed per-arc codes, so the whole check collapses to three
        # comparisons plus one bitwise-and over the arc array.
        src_code, dst_code, pair_code = self._get_matching_cache(arcs)

        # alive[:-1] mirrors ``remaining``/feasibility; the trailing entry is the
        # STOP option, which stays available at every decision.
        alive = np.ones(n_arcs + 1, dtype=bool)
        stop_arr = np.asarray([stop_score])
        matched_edges: list[tuple[str, str]] = []
        mean_lp = 0.0
        mean_entropy = 0.0
        stopped = False
        while True:
            idx = np.flatnonzero(alive[:n_arcs])
            if idx.size == 0:
                break
            cand = np.append(idx, n_arcs)  # STOP is the last candidate
            s = np.concatenate([scores_np[idx], stop_arr])
            s_max = float(s.max())
            logp = s - (s_max + float(np.log(np.exp(s - s_max).sum())))
            if deterministic:
                k = int(np.argmax(s))
            else:
                k = int(np.argmax(s + gumbel_np[cand]))
            mean_lp += float(logp[k])
            p = np.exp(logp)
            mean_entropy -= float((p * logp).sum())
            if k == idx.size:
                stopped = True
                break
            sel = int(idx[k])
            src, dst = arcs[sel]
            matched_edges.append((src, dst))
            alive[:n_arcs] &= ~(
                (src_code == src_code[sel])
                | (dst_code == dst_code[sel])
                | (pair_code == pair_code[sel])
            )
        # PPO needs the probability of the sampled SEQUENCE. Averaging log
        # probabilities would turn its importance ratio into the D-th root
        # of the true ratio and reweight policy gradients by sequence length.
        # Entropy remains a per-decision regularizer with its own coefficient.
        n_decisions = max(1, len(matched_edges) + (1 if stopped else 0))
        mean_entropy = mean_entropy / n_decisions
        mean_lp = _finalize_matching_log_prob(mean_lp, n_decisions=n_decisions, average=False)
        mean_lp_t = torch.tensor(mean_lp, dtype=torch.float32, device=self.device)
        mean_entropy_t = torch.tensor(mean_entropy, dtype=torch.float32, device=self.device)

        return (
            self._matching_to_actions(matched_edges, arc_scores, build_scores=build_scores),
            matched_edges, mean_lp_t, mean_entropy_t,
        )

    def _matching_to_actions(
        self,
        matched_edges: list[tuple[str, str]],
        arc_scores: dict[tuple[str, str], torch.Tensor],
        build_scores: bool = True,
    ) -> tuple[dict[str, tuple[str, str]], dict[str, dict[str, float]]]:
        """Derive per-node ``(tx_target, rx_source)`` actions from matched arcs.

        ``build_scores=False`` leaves the per-edge scores at 0.0 instead of
        reading them off the tensors. Each read is a ``.detach().cpu()``, i.e. a
        full device sync, and ``_matching_to_actions`` does two per matched arc
        (~40 arcs), so a 4-graph rollout was paying ~80 syncs per step. The
        scores are only consumed by the ``priority_matching`` /
        ``max_weight_matching`` resolvers and by ``_sample_matching``'s own
        sampler; the default ``mutual_choice`` resolver ignores them.
        """
        IDLE = self.model.action_space.IDLE
        actions: dict[str, tuple[str, str]] = {}
        action_scores: dict[str, dict[str, float]] = {}
        tx_of: dict[str, str] = {}
        rx_of: dict[str, str] = {}
        for node_id in self.model.action_space.node_ids:
            actions[node_id] = (IDLE, IDLE)
            action_scores[node_id] = {IDLE: 0.0}
        for src, dst in matched_edges:
            tx_of[src] = dst
            rx_of[dst] = src
            if build_scores:
                zero = torch.zeros((), dtype=torch.float32, device=self.device)
                action_scores[src][dst] = float(arc_scores.get((src, dst), zero).detach().cpu())
                action_scores[dst][src] = float(arc_scores.get((dst, src), zero).detach().cpu())
            else:
                action_scores[src][dst] = 0.0
                action_scores[dst][src] = 0.0
        for node_id in self.model.action_space.node_ids:
            tx = tx_of.get(node_id, IDLE)
            rx = rx_of.get(node_id, IDLE)
            actions[node_id] = (tx, rx)
        return actions, action_scores

    @staticmethod
    def _fill_node_tensors(
        node_ids: list[str],
        mean_lp: torch.Tensor,
        mean_entropy: torch.Tensor,
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
            {node_id: mean_lp for node_id in node_ids},
            {node_id: mean_entropy for node_id in node_ids},
        )

    def act(self, obs: GraphObservation, deterministic: bool = False, build_scores: bool = True) -> PolicyStep:
        """Sample a global matching action for the current observation.

        ``build_scores=False`` skips materializing ``PolicyStep.edge_scores``
        as Python floats. That conversion costs one device sync per legal edge
        (~300 per graph) and is only needed by the ``priority_matching`` /
        ``max_weight_matching`` resolvers and by ``log_prob_entropy_for_matching``;
        in the default ``mutual_choice`` mode the resolver ignores it entirely, so
        a rollout was paying ~300 syncs per step for nothing.
        """
        output = self.model(obs, self.device, build_logits_dict=False)
        node_ids = output.logits_node_order if output.logits_node_order is not None else list(output.logits.keys())
        if not node_ids:
            return PolicyStep(
                actions={},
                action_scores={},
                log_probs={},
                entropies={},
                value=output.value,
                mean_log_prob=torch.zeros((), dtype=torch.float32, device=self.device),
                mean_entropy=torch.zeros((), dtype=torch.float32, device=self.device),
                edge_scores=None,
                matched_edges=[],
            )
        edge_scores = output.edge_scores or {}
        (actions, action_scores), matched_edges, mean_lp, mean_entropy = self._sample_matching(
            edge_scores,
            deterministic=deterministic,
            build_scores=build_scores,
        )
        log_probs, entropies = self._fill_node_tensors(node_ids, mean_lp, mean_entropy)
        return PolicyStep(
            actions=actions,
            action_scores=action_scores,
            log_probs=log_probs,
            entropies=entropies,
            value=output.value,
            mean_log_prob=mean_lp,
            mean_entropy=mean_entropy,
            edge_scores=(
                {edge_id: float(score.detach().cpu()) for edge_id, score in edge_scores.items()}
                if (edge_scores and build_scores)
                else None
            ),
            matched_edges=matched_edges,
        )

    def _actions_from_index_pairs(
        self,
        node_ids: list[str],
        pairs: list[tuple[int, int]],
    ) -> tuple[dict[str, tuple[str, str]], dict[str, dict[str, float]]]:
        """Per-node ``(tx_target, rx_source)`` actions from (src, dst) node-index pairs.

        The index-based twin of :meth:`_matching_to_actions`; scores are left at
        0.0 because the callers that need real scores use the arc-dict path.
        """
        IDLE = self.model.action_space.IDLE
        actions: dict[str, tuple[str, str]] = {}
        action_scores: dict[str, dict[str, float]] = {}
        tx_of: dict[str, str] = {}
        rx_of: dict[str, str] = {}
        for src_i, dst_i in pairs:
            src, dst = node_ids[src_i], node_ids[dst_i]
            tx_of[src] = dst
            rx_of[dst] = src
            action_scores.setdefault(src, {IDLE: 0.0})[dst] = 0.0
            action_scores.setdefault(dst, {IDLE: 0.0})[src] = 0.0
        for node_id in node_ids:
            action_scores.setdefault(node_id, {IDLE: 0.0})
            actions[node_id] = (tx_of.get(node_id, IDLE), rx_of.get(node_id, IDLE))
        return actions, action_scores

    def _sample_matching_arrays(
        self,
        src_list: list[np.ndarray],
        dst_list: list[np.ndarray],
        score_list: list[torch.Tensor],
        node_id_list: list[list[str]],
        deterministic: bool = False,
    ) -> list[tuple[tuple[dict[str, tuple[str, str]], dict[str, dict[str, float]]], list[tuple[str, str]], torch.Tensor, torch.Tensor]]:
        """Vectorized twin of :meth:`_sample_matching` for a batch of graphs.

        Identical sequential-sampling semantics -- one decision per matched arc
        plus a final STOP, with the dual-port rules ``Tx-out<=1``, ``Rx-in<=1``
        and 对端不同 -- but the decision loop runs in lockstep over all graphs on
        numpy arrays instead of repeating a per-graph Python loop. Candidates
        arrive as integer node indices, so the three rules become integer
        comparisons and the canonical pair id is ``min*n_nodes + max`` (unique
        without a dict).

        Gumbel noise is drawn per graph in graph order with the same shape as the
        per-graph sampler, so the same seed yields the same matchings; a decision
        is only counted when it actually picks an arc or STOP, exactly matching
        the per-graph loop's early exit when no arc is feasible.
        """
        G = len(src_list)
        empty = torch.zeros((), dtype=torch.float32, device=self.device)
        if G == 0:
            return []
        n_arcs = np.fromiter((len(s) for s in src_list), dtype=np.int64, count=G)
        A = int(n_arcs.max())
        if A == 0:
            return [
                (
                    self._actions_from_index_pairs(node_id_list[g], []),
                    [], empty, empty,
                )
                for g in range(G)
            ]

        temperature = float(self.model.actor.temperature)
        stop_score = float(self.model.actor.stop_logit.detach())
        if temperature != 1.0:
            stop_score = stop_score / temperature
        # S[g, :n_g] are the arc scores, column A is the STOP option.
        S = np.full((G, A + 1), -np.inf, dtype=np.float32)
        for g in range(G):
            n_g = int(n_arcs[g])
            if n_g:
                sc = score_list[g]
                if temperature != 1.0:
                    sc = sc / temperature
                S[g, :n_g] = sc.detach().to("cpu").numpy()
        S[:, A] = stop_score

        src_code = np.zeros((G, A), dtype=np.int64)
        dst_code = np.zeros((G, A), dtype=np.int64)
        pair_code = np.zeros((G, A), dtype=np.int64)
        alive = np.zeros((G, A + 1), dtype=bool)
        alive[:, A] = True  # STOP is available at every decision
        for g in range(G):
            n_g = int(n_arcs[g])
            if not n_g:
                continue
            alive[g, :n_g] = True
            s = np.asarray(src_list[g], dtype=np.int64)
            d = np.asarray(dst_list[g], dtype=np.int64)
            src_code[g, :n_g] = s
            dst_code[g, :n_g] = d
            n_nodes = max(1, len(node_id_list[g]))
            pair_code[g, :n_g] = np.minimum(s, d) * n_nodes + np.maximum(s, d)

        gumbel = None
        if not deterministic:
            gumbel = np.zeros((G, A + 1), dtype=np.float32)
            for g in range(G):
                n_g = int(n_arcs[g])
                gen = (
                    self._sample_gens_batched[g]
                    if self._sample_gens_batched is not None and g < len(self._sample_gens_batched)
                    else self._sample_gen
                )
                u = torch.rand(
                    n_g + 1, dtype=torch.float32, device=self.device, generator=gen
                )
                u = u.clamp_min(torch.finfo(torch.float32).tiny)
                v = (-torch.log(-torch.log(u))).cpu().numpy()
                gumbel[g, :n_g] = v[:n_g]
                gumbel[g, A] = v[n_g]

        matched_idx: list[list[int]] = [[] for _ in range(G)]
        lp_sum = np.zeros(G, dtype=np.float64)
        ent_sum = np.zeros(G, dtype=np.float64)
        n_dec = np.zeros(G, dtype=np.int64)
        stopped = np.zeros(G, dtype=bool)
        finished = np.zeros(G, dtype=bool)
        while True:
            # A graph whose arcs are all infeasible ends deterministically,
            # without a STOP draw (the per-graph loop breaks when avail is empty).
            finished |= ~alive[:, :A].any(axis=1)
            live = ~finished
            if not live.any():
                break
            legal = np.where(alive, S, -np.inf)
            m = legal.max(axis=1, keepdims=True)
            m = np.where(np.isfinite(m), m, 0.0)
            ex = np.where(alive, np.exp(S - m), 0.0)
            logp = np.where(
                alive,
                S - m - np.log(np.maximum(ex.sum(axis=1, keepdims=True), 1e-300)),
                -np.inf,
            )
            pick = legal if deterministic else legal + gumbel
            pick = np.where(live[:, None], pick, -np.inf)
            k = np.where(live, np.argmax(pick, axis=1), A)

            lp_pick = np.take_along_axis(logp, k[:, None], axis=1)[:, 0]
            lp_sum += np.where(live, lp_pick, 0.0)
            safe = np.where(alive, logp, 0.0)
            ent_sum -= np.where(live, (np.exp(safe) * safe).sum(axis=1), 0.0)
            n_dec += live.astype(np.int64)

            picked_arc = live & (k != A)
            stopped |= live & (k == A)
            finished |= live & (k == A)
            if picked_arc.any():
                kk = np.where(picked_arc, k, 0)
                kill = (
                    (src_code == np.take_along_axis(src_code, kk[:, None], axis=1))
                    | (dst_code == np.take_along_axis(dst_code, kk[:, None], axis=1))
                    | (pair_code == np.take_along_axis(pair_code, kk[:, None], axis=1))
                )
                alive[:, :A] &= ~(kill & picked_arc[:, None])
                for g in np.flatnonzero(picked_arc).tolist():
                    matched_idx[g].append(int(k[g]))

        results = []
        for g in range(G):
            node_ids = node_id_list[g]
            pairs = [
                (int(src_list[g][a]), int(dst_list[g][a])) for a in matched_idx[g]
            ]
            actions, action_scores = self._actions_from_index_pairs(node_ids, pairs)
            matched_edges = [(node_ids[s], node_ids[d]) for s, d in pairs]
            divisor = max(1, int(n_dec[g]))
            joint_lp = _finalize_matching_log_prob(
                float(lp_sum[g]), n_decisions=divisor, average=False
            )
            results.append(
                (
                    (actions, action_scores),
                    matched_edges,
                    torch.as_tensor(joint_lp, dtype=torch.float32, device=self.device),
                    torch.as_tensor(float(ent_sum[g]) / divisor, dtype=torch.float32, device=self.device),
                )
            )
        return results

    def act_batched(
        self,
        obs_list: list[GraphObservation],
        deterministic: bool = False,
        build_scores: bool = True,
        use_edge_arrays: bool = False,
    ) -> list[PolicyStep]:
        """Sample global matching actions for many graphs with one model forward.

        See :meth:`act` for ``build_scores``. ``use_edge_arrays`` routes the
        matching through :meth:`_sample_matching_arrays`, which needs the raw
        candidate index arrays from the batched forward; it is only valid when
        ``build_scores`` is False (the arrays path does not materialize per-edge
        score dicts).
        """
        arrays = None
        want_edge_maps = build_scores or not use_edge_arrays
        outputs = self.model.batched_forward(
            obs_list, self.device, want_edge_maps=want_edge_maps
        )
        values = outputs.values
        edge_score_maps = outputs.edge_score_maps
        if use_edge_arrays and not build_scores:
            arrays = outputs.edge_arrays
        fast = None
        if arrays is not None:
            fast = self._sample_matching_arrays(
                [a[0] for a in arrays],
                [a[1] for a in arrays],
                [a[2] for a in arrays],
                [list(obs.node_ids) for obs in obs_list],
                deterministic=deterministic,
            )

        steps: list[PolicyStep] = []
        for gi, (obs, value) in enumerate(zip(obs_list, values)):
            node_order = list(obs.node_ids)
            n = len(node_order)
            # edge_score_maps is empty when want_edge_maps was False (the arrays
            # path); the sampler does not need it in that case.
            edge_scores = (edge_score_maps[gi] if gi < len(edge_score_maps) else {}) or {}
            if n == 0:
                steps.append(
                    PolicyStep(
                        actions={},
                        action_scores={},
                        log_probs={},
                        entropies={},
                        value=value,
                        mean_log_prob=torch.zeros((), dtype=torch.float32, device=self.device),
                        mean_entropy=torch.zeros((), dtype=torch.float32, device=self.device),
                        edge_scores=None,
                        matched_edges=[],
                    )
                )
                continue
            if fast is not None:
                (actions, action_scores), matched_edges, mean_lp, mean_entropy = fast[gi]
            else:
                (actions, action_scores), matched_edges, mean_lp, mean_entropy = self._sample_matching(
                    edge_scores,
                    deterministic=deterministic,
                    build_scores=build_scores,
                )

            log_probs, entropies = self._fill_node_tensors(node_order, mean_lp, mean_entropy)
            steps.append(
                PolicyStep(
                    actions=actions,
                    action_scores=action_scores,
                    log_probs=log_probs,
                    entropies=entropies,
                    value=value,
                    mean_log_prob=mean_lp,
                    mean_entropy=mean_entropy,
                    edge_scores=(
                        {edge_id: float(score.detach().cpu()) for edge_id, score in edge_scores.items()}
                        if (edge_scores and build_scores)
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
        mean_lp, mean_entropy = self._matching_log_prob_entropy_fast(edge_scores, matched_edges)
        log_probs, entropies = self._fill_node_tensors(node_ids, mean_lp, mean_entropy)
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
        outputs = self.model.batched_forward(obs_list, self.device, want_edge_maps=False)
        arrays = outputs.edge_arrays or []
        results: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]] = []
        for i, (obs, value) in enumerate(zip(
            obs_list,
            outputs.values,
        )):
            node_order = list(obs.node_ids)
            if not node_order:
                results.append(({}, {}, value))
                continue
            src_arr, dst_arr, scores = arrays[i]
            node_pos = {node_id: j for j, node_id in enumerate(node_order)}
            mean_lp, mean_entropy = self._matching_log_prob_entropy_arrays(
                src_arr, dst_arr, scores, node_pos, matched_edges_list[i]
            )
            log_probs, entropies = self._fill_node_tensors(node_order, mean_lp, mean_entropy)
            results.append((log_probs, entropies, value))
        return results

    def log_prob_entropy_for_matching(
        self,
        arc_scores: dict[tuple[str, str], float] | None,
        matched_edges: list[tuple[str, str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a stored executed matching from detached rollout scores."""
        if not arc_scores:
            return (
                torch.zeros((), dtype=torch.float32, device=self.device),
                torch.zeros((), dtype=torch.float32, device=self.device),
            )
        arc_score_tensors = {
            arc: torch.tensor(float(score), dtype=torch.float32, device=self.device)
            for arc, score in arc_scores.items()
        }
        return self._matching_log_prob_entropy(arc_score_tensors, matched_edges)

    def _matching_log_prob_entropy(
        self,
        arc_scores: dict[tuple[str, str], torch.Tensor],
        matched_edges: list[tuple[str, str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint log prob and per-decision average entropy of a stored directed
        matching, replaying the dual-port feasibility rules ``Tx-out<=1``,
        ``Rx-in<=1`` and 对端不同 (a pair matched once in either direction),
        plus the STOP option. Each decision is a softmax over the arcs still
        feasible at that step and STOP; the final decision of a terminated
        matching is a STOP draw over the arcs still feasible (if any), which
        is exactly the process ``_sample_matching`` used.
        """
        arcs = list(arc_scores.keys())
        if not arcs:
            return (
                torch.zeros((), dtype=torch.float32, device=self.device),
                torch.zeros((), dtype=torch.float32, device=self.device),
            )
        temperature = float(self.model.actor.temperature)
        temp_scores = {
            arc: score / temperature if temperature != 1.0 else score
            for arc, score in arc_scores.items()
        }
        stop_score = self.model.actor.stop_logit
        if temperature != 1.0:
            stop_score = stop_score / temperature
        used_tx: set[str] = set()
        used_rx: set[str] = set()
        used_pair: set[tuple[str, str]] = set()
        selected_log_probs: list[torch.Tensor] = []
        mean_entropy = torch.zeros((), dtype=torch.float32, device=self.device)

        def _is_feasible(i: int) -> bool:
            src, dst = arcs[i]
            return (
                src not in used_tx
                and dst not in used_rx
                and (src, dst) not in used_pair
                and (dst, src) not in used_pair
            )

        for arc in matched_edges:
            cur_src, cur_dst = arc
            pair = (cur_src, cur_dst) if cur_src < cur_dst else (cur_dst, cur_src)
            avail = sorted(i for i in range(len(arcs)) if _is_feasible(i))
            raw_scores = torch.cat(
                [torch.stack([temp_scores[arcs[i]] for i in avail]), stop_score.reshape(1)]
            )
            logp = torch.log_softmax(raw_scores, dim=0)
            mean_entropy = mean_entropy - (logp.exp() * logp).sum()
            try:
                pos = next(i for i, a in enumerate(avail) if arcs[a] == arc)
            except StopIteration as exc:
                raise ValueError(f"Stored matching arc {arc!r} is not available for evaluation.") from exc
            selected_log_probs.append(logp[pos])
            used_tx.add(cur_src)
            used_rx.add(cur_dst)
            used_pair.add(pair)
        # After the stored arcs the matching ended: either no arc remains
        # feasible (deterministic end, no STOP decision) or STOP was chosen
        # among the still-feasible arcs.
        remaining_avail = sorted(i for i in range(len(arcs)) if _is_feasible(i))
        stopped = bool(remaining_avail)
        if remaining_avail:
            raw_scores = torch.cat(
                [torch.stack([temp_scores[arcs[i]] for i in remaining_avail]), stop_score.reshape(1)]
            )
            logp = torch.log_softmax(raw_scores, dim=0)
            mean_entropy = mean_entropy - (logp.exp() * logp).sum()
            selected_log_probs.append(logp[-1])  # STOP chosen
        n_decisions = max(1, len(matched_edges) + (1 if stopped else 0))
        mean_entropy = mean_entropy / n_decisions
        selected = torch.stack(selected_log_probs) if selected_log_probs else mean_entropy.new_empty(0)
        total_log_prob = selected.sum()
        joint_log_prob = _finalize_matching_log_prob(
            total_log_prob, n_decisions=n_decisions, average=False
        )
        return joint_log_prob, mean_entropy

    def _matching_log_prob_entropy_fast(
        self,
        arc_scores: dict[tuple[str, str], torch.Tensor],
        matched_edges: list[tuple[str, str]],
        *,
        average_log_prob: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized twin of ``_matching_log_prob_entropy`` (identical math).

        The sequential version issues one small softmax per matching decision;
        with ~50 decisions per graph and dozens of graphs per batch its
        autograd graph explodes into thousands of tiny nodes and the BC backward
        takes tens of seconds. Here each graph's feasibility pattern is a
        static 0/1 matrix ``A`` (built in numpy, no autograd), all decision
        logits come from a single ``A @ scores`` matmul, and the loss is one
        ``log_softmax`` + gather per graph. Mathematically the same: every
        decision's softmax is over the arcs still feasible at that step plus
        STOP, and the final decision (if any arc remains feasible) is a STOP
        draw over the remaining arcs.
        """
        arcs = list(arc_scores.keys())
        n_arcs = len(arcs)
        if n_arcs == 0:
            return (
                torch.zeros((), dtype=torch.float32, device=self.device),
                torch.zeros((), dtype=torch.float32, device=self.device),
            )
        temperature = float(self.model.actor.temperature)
        if temperature != 1.0:
            scores = torch.stack(
                [arc_scores[a] / temperature for a in arcs]
            )
            stop_score = self.model.actor.stop_logit / temperature
        else:
            scores = torch.stack([arc_scores[a] for a in arcs])
            stop_score = self.model.actor.stop_logit

        # Static per-decision feasibility rows (deterministic, numpy only).
        #
        # The feasibility test per decision is three equality checks against
        # precomputed per-arc integer codes, so a row is one bitwise-and over the
        # arc array. The previous version rescanned every arc in Python for every
        # decision, which measured 46% of a whole PPO minibatch (0.99 s of 2.15 s
        # for 256 graphs) -- ~59 decisions x ~340 arcs x 256 graphs per
        # minibatch. `node_code` maps node ids to small ints so the comparisons
        # are integer, and `pair_code` folds "the pair is already matched in
        # either direction" into a single id.
        node_code = {node_id: i for i, node_id in enumerate(sorted({n for arc in arcs for n in arc}))}
        src_code = np.fromiter((node_code[a[0]] for a in arcs), dtype=np.int64, count=n_arcs)
        dst_code = np.fromiter((node_code[a[1]] for a in arcs), dtype=np.int64, count=n_arcs)
        pair_ids: dict[tuple[int, int], int] = {}
        pair_code = np.empty(n_arcs, dtype=np.int64)
        arc_pos: dict[tuple[str, str], int] = {}
        for i, arc in enumerate(arcs):
            src_c, dst_c = node_code[arc[0]], node_code[arc[1]]
            key = (src_c, dst_c) if src_c <= dst_c else (dst_c, src_c)
            pair_code[i] = pair_ids.setdefault(key, len(pair_ids))
            arc_pos[arc] = i

        alive = np.ones(n_arcs, dtype=bool)
        rows = np.empty((len(matched_edges) + 1, n_arcs), dtype=np.float32)
        choices: list[int] = []
        n_rows = 0
        for arc in matched_edges:
            pos = arc_pos.get(arc)
            if pos is None or not alive[pos]:
                raise ValueError(f"Stored matching arc {arc!r} is not available for evaluation.")
            rows[n_rows] = alive
            choices.append(pos)
            n_rows += 1
            # The chosen arc is excluded by its own src/dst/pair code, so no
            # separate "remove it" step is needed.
            alive &= ~(
                (src_code == src_code[pos])
                | (dst_code == dst_code[pos])
                | (pair_code == pair_code[pos])
            )
        # After the stored arcs the matching ended: either no arc remains
        # feasible (deterministic end, no STOP decision) or STOP was chosen
        # among the still-feasible arcs.
        if alive.any():
            rows[n_rows] = alive
            choices.append(n_arcs)  # STOP among the still-feasible arcs
            n_rows += 1
        rows = rows[:n_rows]

        if n_rows == 0:
            return (
                torch.zeros((), dtype=torch.float32, device=self.device),
                torch.zeros((), dtype=torch.float32, device=self.device),
            )
        A = torch.from_numpy(rows).to(device=self.device)
        # logits[d, a] = score of arc a if it is feasible at decision d, else
        # -inf, plus a STOP column present in every decision. This mirrors the
        # sequential softmax (feasible arcs + STOP) exactly, but batched.
        logits = (A * scores.unsqueeze(0)).masked_fill(A == 0.0, float("-inf"))
        logits = torch.cat(
            [logits, stop_score.reshape(1).expand(logits.size(0), 1)], dim=-1
        )
        logp = torch.log_softmax(logits, dim=-1)
        choices_t = torch.tensor(choices, dtype=torch.long, device=self.device).unsqueeze(-1)
        # BC may explicitly average its NLL to preserve the historical
        # per-demonstration weighting. PPO must always use the joint sum.
        selected = logp.gather(-1, choices_t)
        mean_lp = _finalize_matching_log_prob(
            selected.sum(), n_decisions=selected.numel(), average=average_log_prob
        )
        # -inf padding would turn into 0 * -inf = nan, so only finite logits
        # contribute to the entropy.
        safe = torch.where(torch.isfinite(logits), logp, torch.zeros_like(logp))
        mean_entropy = -(safe.exp() * safe).sum(dim=-1).mean()
        return mean_lp, mean_entropy

    def _matching_log_prob_entropy_arrays(
        self,
        src_arr: np.ndarray,
        dst_arr: np.ndarray,
        scores: torch.Tensor,
        node_pos: dict[str, int],
        matched_edges: list[tuple[str, str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Array-input twin of ``_matching_log_prob_entropy_fast`` for PPO updates.

        The dict-input version re-derives the arc order from ``arc_scores.keys()``
        and spends most of its time building that dict (``_edge_score_map``) and
        re-pairing arcs. Here the candidate order is already the flat index order
        of ``src_arr``/``dst_arr``/``scores`` (exactly what ``edge_arrays`` handed
        to ``_sample_matching_arrays``), so the per-arc codes fall out of integer
        indexing and the scores come straight off the array. Math is bit-identical
        to ``_matching_log_prob_entropy_fast``: same feasibility rows, same
        ``log_softmax`` + gather over them.
        """
        n_arcs = int(len(src_arr))
        if n_arcs == 0:
            return (
                torch.zeros((), dtype=torch.float32, device=self.device),
                torch.zeros((), dtype=torch.float32, device=self.device),
            )
        temperature = float(self.model.actor.temperature)
        if temperature != 1.0:
            scores = scores / temperature
            stop_score = self.model.actor.stop_logit / temperature
        else:
            stop_score = self.model.actor.stop_logit

        src_code = np.asarray(src_arr, dtype=np.int64)
        dst_code = np.asarray(dst_arr, dtype=np.int64)
        # Undirected pair id via min/max, matching the sampler's canonical key.
        n_nodes = max(1, len(node_pos))
        pair_code = np.minimum(src_code, dst_code) * n_nodes + np.maximum(src_code, dst_code)

        # Candidate position by (src, dst), precomputed once for the whole
        # evaluation instead of a np.flatnonzero scan per matched arc.
        #
        # Why this is equivalent: within one observation a `(src_code, dst_code)`
        # pair occurs AT MOST ONCE. `action_space.neighbors[node]` appends each
        # adjacent node exactly once per edge (action_space.py:31-37), the
        # candidate list is the sorted, de-duplicated `legal` subset of that
        # (graph_builder.py:187-188), and IDLE filters out of the edge set here
        # so it cannot collide with a real node id. Therefore the scan always
        # returned a single-element array and `cand[0]` was simply this pair's
        # fixed index -- independent of `alive`. Verified empirically on 3600
        # candidate lists (0 duplicates); the previous form spent most of this
        # function's time there (79451 np.flatnonzero calls per update).
        #
        # The `alive` check below still catches a stored arc whose pair is
        # absent (lookup miss) or already consumed by an earlier arc -- the same
        # error the old `cand.size == 0 or not alive[cand[0]]` raised.
        arc_pos_lookup: dict[int, int] = {}
        for i in range(n_arcs):
            arc_pos_lookup.setdefault(
                int(src_code[i]) * (n_nodes + 1) + int(dst_code[i]), i
            )

        alive = np.ones(n_arcs, dtype=bool)
        rows = np.empty((len(matched_edges) + 1, n_arcs), dtype=np.float32)
        choices: list[int] = []
        n_rows = 0
        for arc in matched_edges:
            pos = node_pos.get(arc[0])
            if pos is None:
                raise ValueError(f"Stored matching arc {arc!r} is not in the observation.")
            dst_pos = node_pos.get(arc[1])
            if dst_pos is None:
                raise ValueError(f"Stored matching arc {arc!r} dst not in the observation.")
            arc_pos = arc_pos_lookup.get(pos * (n_nodes + 1) + dst_pos, -1)
            if arc_pos < 0 or not alive[arc_pos]:
                raise ValueError(f"Stored matching arc {arc!r} is not available for evaluation.")
            rows[n_rows] = alive
            choices.append(arc_pos)
            n_rows += 1
            alive &= ~(
                (src_code == src_code[arc_pos])
                | (dst_code == dst_code[arc_pos])
                | (pair_code == pair_code[arc_pos])
            )
        if alive.any():
            rows[n_rows] = alive
            choices.append(n_arcs)  # STOP among the still-feasible arcs
            n_rows += 1
        rows = rows[:n_rows]
        if n_rows == 0:
            return (
                torch.zeros((), dtype=torch.float32, device=self.device),
                torch.zeros((), dtype=torch.float32, device=self.device),
            )
        A = torch.from_numpy(rows).to(device=self.device)
        logits = (A * scores.unsqueeze(0)).masked_fill(A == 0.0, float("-inf"))
        logits = torch.cat(
            [logits, stop_score.reshape(1).expand(logits.size(0), 1)], dim=-1
        )
        logp = torch.log_softmax(logits, dim=-1)
        choices_t = torch.tensor(choices, dtype=torch.long, device=self.device).unsqueeze(-1)
        selected = logp.gather(-1, choices_t)
        mean_lp = _finalize_matching_log_prob(
            selected.sum(), n_decisions=selected.numel(), average=False
        )
        safe = torch.where(torch.isfinite(logits), logp, torch.zeros_like(logp))
        mean_entropy = -(safe.exp() * safe).sum(dim=-1).mean()
        return mean_lp, mean_entropy


def _masked_log_prob_entropy(
    logits: torch.Tensor, idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log probs and entropies for a masked categorical, ``-inf`` padding safe."""
    logp = torch.log_softmax(logits, dim=-1)
    log_probs = logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    safe = torch.where(torch.isfinite(logits), logp, torch.zeros_like(logp))
    entropies = -(safe.exp() * safe).sum(-1)
    return log_probs, entropies
