"""Demand-aware relay greedy baseline.

The plain per-link greedy baselines pick links by rate / inventory / pending
demand but ignore *connectivity*. In this QKD-SAGIN topology every ground
station pair is exactly two hops apart (GS -> relay -> GS), each node can
activate at most one link per slot, and a request can only be served once a
2-hop path holds a positive key stock on *both* hops. Rate-only matching keeps
changing the active links, so stocked relay paths almost never form and the
extra generated keys are never used (served keys ~ random).

This policy scores every legal edge by the pending demand whose 2-hop path
uses it, and gives a completion bonus to the edge that would finish a path
whose other hop is already stocked (keys from a previous slot, TTL-protected).
The result is a stable, demand-pair-aware relay schedule that actually serves
requests while still keeping links active across slots to avoid the switch-cost
rate decay. It is a pure heuristic baseline: it consumes only GraphObservation
fields and never imports the training stack.
"""

from __future__ import annotations

from qkd_rl.baselines._matching import edge_map, greedy_matching_actions
from qkd_rl.env.graph_builder import GraphObservation


class GreedyRelayPolicy:
    """Global greedy matching over demand-aware relay paths.

    ``score(edge) = rate_weight * rate_norm + demand_weight * credit_norm
    + keep_weight * keep_norm``. ``credit_norm`` is the edge's share of the
    pending pair demand (normalised by the per-step maximum) weighted by the
    path's bottleneck rate, and the completing edge of a half-stocked path
    receives ``completion_multiplier *`` its base credit so finishing a path
    wins over starting new ones. ``keep_norm`` is 1 only for links active in the
    previous slot that are still on a demand path (a plain keep bonus anchors
    the schedule to rate-only links and measurably hurts service).
    """

    def __init__(
        self,
        rate_weight: float = 1.0,
        demand_weight: float = 2.0,
        completion_multiplier: float = 3.0,
        keep_weight: float = 0.5,
        deadline_window: float = 60.0,
    ):
        self.rate_weight = float(rate_weight)
        self.demand_weight = float(demand_weight)
        self.completion_multiplier = float(completion_multiplier)
        self.keep_weight = float(keep_weight)
        self.deadline_window = max(1.0, float(deadline_window))

    def act(self, obs: GraphObservation) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        endpoints, pair_to_edge = edge_map(obs)
        candidates = {
            node_id: set(obs.action_candidates[node_id]) - {"idle"} for node_id in obs.node_ids
        }
        levels = dict(obs.state.qkp_snapshot)
        last_activated = set(obs.state.last_activated_edges)
        windows = obs.state.edge_windows
        rates = {edge_id: windows[edge_id].rates[0] for edge_id in obs.physical_edge_ids}
        max_rate = max(rates.values(), default=1.0) or 1.0

        # Aggregate remaining demand per unordered GS pair, keeping the most
        # urgent deadline. Credit grows with remaining amount and closeness to
        # the deadline so near-expiring requests get priority.
        pairs: dict[tuple[str, str], dict[str, float]] = {}
        t = obs.state.t
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

        # Demand credit per edge, weighted by the path's bottleneck rate. A
        # request can only be served as fast as the weaker hop generates keys
        # (min(rate_a, rate_b) * slot), so a low-rate relay path is nearly
        # useless even when it matches a huge pending pair; scaling the credit
        # by bottleneck_rate/max_rate makes the greedy prefer high-capacity
        # paths instead of completing whatever relay happens to be stocked.
        credit: dict[str, float] = {}
        completing: set[str] = set()
        for pair, info in pairs.items():
            a, b = pair
            amount = info["amount"]
            urgency = info["urgency"]
            direct = pair_to_edge.get(pair)
            if direct is not None:
                credit[direct] = credit.get(direct, 0.0) + amount * urgency
                continue
            relays = candidates.get(a, set()) & candidates.get(b, set())
            for relay in relays:
                e_a = pair_to_edge.get(tuple(sorted((a, relay))))
                e_b = pair_to_edge.get(tuple(sorted((b, relay))))
                if e_a is None or e_b is None:
                    continue
                rate_a = rates.get(e_a, 0.0)
                rate_b = rates.get(e_b, 0.0)
                path_rate_norm = min(rate_a, rate_b) / max_rate
                stocked_a = levels.get(e_a, 0.0) > 0.0
                stocked_b = levels.get(e_b, 0.0) > 0.0
                if stocked_a and stocked_b:
                    continue  # path ready; serving happens automatically
                base = 0.5 * amount * urgency * path_rate_norm
                if not stocked_a and not stocked_b:
                    credit[e_a] = credit.get(e_a, 0.0) + base
                    credit[e_b] = credit.get(e_b, 0.0) + base
                elif not stocked_b:
                    credit[e_b] = credit.get(e_b, 0.0) + self.completion_multiplier * base
                    completing.add(e_b)
                else:  # not stocked_a
                    credit[e_a] = credit.get(e_a, 0.0) + self.completion_multiplier * base
                    completing.add(e_a)

        max_credit = max(credit.values(), default=1.0) or 1.0
        edge_scores: dict[str, float] = {}
        for edge_id in obs.physical_edge_ids:
            rate_norm = rates.get(edge_id, 0.0) / max_rate
            credit_norm = credit.get(edge_id, 0.0) / max_credit
            score = self.rate_weight * rate_norm + self.demand_weight * credit_norm
            # Keep bonus only for links that are still on a demand path: pure
            # rate links would be anchored away from the current best demand
            # paths, which measurably hurts service (keep=0 beats keep>0 on
            # all edges). This conditional form avoids the switch-cost decay
            # on paths the policy actually wants to keep without anchoring
            # to irrelevant links.
            if (
                self.keep_weight > 0.0
                and edge_id in last_activated
                and credit.get(edge_id, 0.0) > 0.0
            ):
                score += self.keep_weight
            edge_scores[edge_id] = score
        return greedy_matching_actions(obs, edge_scores, tie_rates=rates, priority_edges=completing)
