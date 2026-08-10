"""Dynamic relay-path scheduler (v2 greedy baseline).

v1 (``GreedyRelayPolicy``) re-scores every edge from scratch each slot, so the
relay nodes keep being pulled between different demand pairs and only ~3 of
~6 half-stocked paths get completed per slot; the key conversion is <1% even
though generation is ~11 Mbit/slot. The scheduler below fixes the pairs that
are actively being served and *keeps* them: once a pair gets a relay, the
relay alternates between the two hops until the pair demand is gone, which
lets both hops accumulate stock instead of being abandoned mid-path.

Each slot the scheduler:
1. aggregates remaining demand per GS pair (amount x deadline urgency),
2. keeps / creates a relay assignment for the top pairs (max bottleneck-rate
   common relay, re-picked only when the pair is new or its relay went
   unavailable),
3. activates the hop the pair still needs (completing hop first, then the
   emptier hop, then the faster hop when both are empty),
4. resolves node conflicts greedily by pair priority,
5. fills the remaining idle nodes with plain rate matching.

It is a pure heuristic baseline: stateful but consumes only GraphObservation
fields plus its own schedule, and never imports the training stack.
"""

from __future__ import annotations

from qkd_rl.baselines._matching import edge_map, greedy_matching_actions
from qkd_rl.env.graph_builder import GraphObservation


class GreedyRelayScheduler:
    """Stateful demand-pair relay scheduler."""

    def __init__(
        self,
        max_pairs: int = 14,
        deadline_window: float = 240.0,
        fill_with_rate: bool = True,
    ):
        self.max_pairs = int(max_pairs)
        self.deadline_window = max(1.0, float(deadline_window))
        self.fill_with_rate = bool(fill_with_rate)
        self._schedule: dict[tuple[str, str], str] = {}
        self._touch: dict[tuple[str, str], int] = {}
        self._last_t: int | None = None

    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        t = obs.state.t
        # Episode boundary: the previous episode ended and t moved backwards
        # (or jumped to a random day), so drop stale relay assignments.
        if self._last_t is not None and t < self._last_t:
            self._schedule.clear()
            self._touch.clear()
        self._last_t = t

        endpoints, pair_to_edge = edge_map(obs)
        candidates = {
            node_id: set(obs.action_candidates[node_id]) - {"idle"} for node_id in obs.node_ids
        }
        levels = dict(obs.state.qkp_snapshot)
        windows = obs.state.edge_windows
        rates = {edge_id: windows[edge_id].rates[0] for edge_id in obs.physical_edge_ids}
        max_rate = max(rates.values(), default=1.0) or 1.0

        # Aggregate remaining demand per pair.
        pairs: dict[tuple[str, str], dict[str, float]] = {}
        for req in obs.state.pending_requests:
            remaining = max(0.0, req.amount - req.served_amount)
            if remaining <= 1.0e-9:
                continue
            pair = tuple(sorted((req.src_gs, req.dst_gs)))
            slack = req.deadline_t - t
            urgency = 1.0 + max(0.0, 1.0 - slack / self.deadline_window)
            entry = pairs.setdefault(pair, {"amount": 0.0, "urgency": 1.0})
            entry["amount"] += remaining
            entry["urgency"] = max(entry["urgency"], urgency)
        demand_now = set(pairs)

        # Drop schedules whose relay is no longer legal for the pair, and
        # evict the least recently active pairs beyond max_pairs (LRU). A
        # pair is kept even while it briefly has no pending demand so a hot
        # pair's relay path is not torn down and rebuilt every slot.
        for pair in list(self._schedule):
            relay = self._schedule[pair]
            a, b = pair
            if relay not in candidates.get(a, set()) or relay not in candidates.get(b, set()):
                del self._schedule[pair]
        if len(self._schedule) > self.max_pairs:
            order = sorted(self._schedule, key=lambda p: self._touch.get(p, 0))
            for pair in order[: len(self._schedule) - self.max_pairs]:
                del self._schedule[pair]

        # Pick relays for the top pairs that still need one.
        ranked = sorted(pairs.items(), key=lambda kv: (-(kv[1]["amount"] * kv[1]["urgency"]), kv[0]))
        for pair, _info in ranked[: self.max_pairs]:
            if pair in self._schedule:
                continue
            relay = self._best_relay(pair, candidates, pair_to_edge, rates, max_rate)
            if relay is not None:
                self._schedule[pair] = relay

        # Decide which hop each scheduled pair activates this slot.
        self._touch = {pair: (self._touch.get(pair, 0) + 1 if pair in self._schedule else self._touch.get(pair, 0)) for pair in self._schedule}
        wants: list[tuple[float, tuple[str, str], str, str]] = []  # (priority, pair, src, relay)
        for pair, relay in self._schedule.items():
            a, b = pair
            e_a = pair_to_edge.get(tuple(sorted((a, relay))))
            e_b = pair_to_edge.get(tuple(sorted((b, relay))))
            if e_a is None or e_b is None:
                continue
            if e_a not in rates or e_b not in rates:
                continue
            l_a = levels.get(e_a, 0.0)
            l_b = levels.get(e_b, 0.0)
            if l_a > 0.0 and l_b <= 0.0:
                src = b
            elif l_b > 0.0 and l_a <= 0.0:
                src = a
            elif l_a > 0.0 and l_b > 0.0:
                src = a if l_a < l_b else b  # replenish the emptier hop
            else:
                src = a if rates[e_a] >= rates[e_b] else b  # start on the faster hop
            priority = pairs.get(pair, {"amount": 0.0, "urgency": 1.0})["amount"] * pairs.get(
                pair, {"amount": 0.0, "urgency": 1.0}
            )["urgency"]
            wants.append((priority, pair, src, relay))

        # Greedily resolve node conflicts by priority.
        wants.sort(key=lambda item: -item[0])
        actions: dict[str, str] = {}
        used: set[str] = set()
        for _priority, _pair, src, relay in wants:
            if src in used or relay in used:
                continue
            if src not in candidates or relay not in candidates:
                continue
            if relay not in candidates[src] or src not in candidates[relay]:
                continue
            actions[src] = relay
            actions[relay] = src
            used.update((src, relay))

        # Fill the remaining nodes with rate-only matching, but only for
        # nodes that are still free, and prefer edges that are not touching
        # a scheduled relay so we do not steal capacity from active paths.
        if self.fill_with_rate:
            free = [node_id for node_id in obs.node_ids if node_id not in used]
            remaining_scores: dict[str, float] = {}
            for edge_id in obs.physical_edge_ids:
                src, dst = endpoints[edge_id]
                if src in used or dst in used:
                    continue
                if src not in free or dst not in free:
                    continue
                remaining_scores[edge_id] = rates.get(edge_id, 0.0) / max_rate
            if remaining_scores:
                extra, _extra_scores = greedy_matching_actions(obs, remaining_scores, tie_rates=rates)
                for node_id, action in extra.items():
                    if action != "idle" and node_id not in used:
                        actions[node_id] = action
                        used.add(node_id)
                        used.add(action)

        for node_id in obs.node_ids:
            actions.setdefault(node_id, "idle")

        scores: dict[str, dict[str, float]] = {}
        for node_id in obs.node_ids:
            node_scores: dict[str, float] = {}
            for action in obs.action_candidates[node_id]:
                if action == "idle":
                    node_scores[action] = 0.0
                else:
                    edge_id = pair_to_edge.get(tuple(sorted((node_id, action))))
                    node_scores[action] = rates.get(edge_id, 0.0) / max_rate
            scores[node_id] = node_scores
        return actions, scores

    def _best_relay(self, pair, candidates, pair_to_edge, rates, max_rate):
        a, b = pair
        best = None
        best_score = -1.0
        for x in candidates.get(a, set()) & candidates.get(b, set()):
            e_a = pair_to_edge.get(tuple(sorted((a, x))))
            e_b = pair_to_edge.get(tuple(sorted((b, x))))
            if e_a is None or e_b is None:
                continue
            ra = rates.get(e_a, 0.0)
            rb = rates.get(e_b, 0.0)
            score = min(ra, rb) / max_rate
            if score > best_score:
                best_score, best = score, x
        return best
