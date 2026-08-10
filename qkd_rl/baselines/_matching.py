"""Shared greedy-matching utilities for the baseline policies.

The per-node greedy baselines used to pick each node's best link
independently. In `mutual_choice` resolver mode that almost never produces
an activated edge (both endpoints must choose each other), so the policies
were effectively useless. These helpers replace that with a global greedy
matching: sort all legal edges by score, activate each edge whose two
endpoints are still free, and submit *mutual* actions so the resolver keeps
the match. This is a pure heuristic improvement; it consumes only
GraphObservation fields like every other baseline.
"""

from __future__ import annotations

from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.graph_builder import GraphObservation


def edge_map(obs: GraphObservation) -> tuple[dict[str, tuple[str, str]], dict[tuple[str, str], str]]:
    """Parse `E_{src}__{dst}` edge ids into (endpoints, pair->edge) maps."""
    endpoints: dict[str, tuple[str, str]] = {}
    pair_to_edge: dict[tuple[str, str], str] = {}
    for edge_id in obs.physical_edge_ids:
        body = edge_id[2:] if edge_id.startswith("E_") else edge_id
        if "__" in body:
            src, dst = body.split("__", 1)
        else:
            src = dst = edge_id  # malformed id: never matchable self-loop
        endpoints[edge_id] = (src, dst)
        pair_to_edge[tuple(sorted((src, dst)))] = edge_id
    return endpoints, pair_to_edge


def pending_demand(
    obs: GraphObservation,
) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    """Pending GS-GS demand: (per-pair amount, per-node incident amount)."""
    pair_demand: dict[tuple[str, str], float] = {}
    incident: dict[str, float] = {}
    for req in obs.state.pending_requests:
        remaining = max(0.0, req.amount - req.served_amount)
        pair = tuple(sorted((req.src_gs, req.dst_gs)))
        pair_demand[pair] = pair_demand.get(pair, 0.0) + remaining
        incident[req.src_gs] = incident.get(req.src_gs, 0.0) + remaining
        incident[req.dst_gs] = incident.get(req.dst_gs, 0.0) + remaining
    return pair_demand, incident


def edge_demand(
    src: str,
    dst: str,
    pair_demand: dict[tuple[str, str], float],
    incident: dict[str, float],
) -> float:
    """How much pending demand a physical edge can plausibly serve.

    A direct GS-GS pair uses its exact pair demand; any edge incident to a GS
    (GS-HAP / GS-SAT) uses the GS's total pending demand, because relayed key
    requests may route through it. Edges between non-GS nodes score zero.
    """
    direct = pair_demand.get(tuple(sorted((src, dst))), 0.0)
    if direct > 0.0:
        return direct
    return max(incident.get(src, 0.0), incident.get(dst, 0.0))


def greedy_matching_actions(
    obs: GraphObservation,
    edge_scores: dict[str, float],
    tie_rates: dict[str, float] | None = None,
    priority_edges: set[str] | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Global greedy matching over legal edges.

    Edges are ranked by descending score (tie-break: higher rate, then edge
    id for determinism) and accepted while both endpoints are still free.
    ``priority_edges`` (e.g. path-completing edges) are matched first among
    themselves so they are never starved by ordinary demand edges, then the
    remaining edges fill the free nodes. Accepted edges produce mutual
    actions; the per-node score dict keeps every candidate so the
    `priority_matching` resolver can still re-rank.
    """
    endpoints, pair_to_edge = edge_map(obs)
    candidates: dict[str, set[str]] = {
        node_id: set(obs.action_candidates[node_id]) for node_id in obs.node_ids
    }
    priority = [edge_id for edge_id in edge_scores if priority_edges and edge_id in priority_edges]
    ranked = sorted(
        priority,
        key=lambda edge_id: (
            -edge_scores[edge_id],
            -(tie_rates.get(edge_id, 0.0) if tie_rates else 0.0),
            edge_id,
        ),
    ) + sorted(
        (edge_id for edge_id in edge_scores if not priority_edges or edge_id not in priority_edges),
        key=lambda edge_id: (
            -edge_scores[edge_id],
            -(tie_rates.get(edge_id, 0.0) if tie_rates else 0.0),
            edge_id,
        ),
    )
    used: set[str] = set()
    actions: dict[str, str] = {}
    for edge_id in ranked:
        src, dst = endpoints[edge_id]
        if src == dst or src in used or dst in used:
            continue
        if src not in candidates or dst not in candidates:
            continue
        if src not in candidates[dst] or dst not in candidates[src]:
            continue
        actions[src] = dst
        actions[dst] = src
        used.update((src, dst))
    for node_id in obs.node_ids:
        actions.setdefault(node_id, NodeActionSpace.IDLE)

    scores: dict[str, dict[str, float]] = {}
    for node_id in obs.node_ids:
        node_scores: dict[str, float] = {}
        for action in obs.action_candidates[node_id]:
            if action == NodeActionSpace.IDLE:
                node_scores[action] = 0.0
            else:
                edge_id = pair_to_edge.get(tuple(sorted((node_id, action))))
                node_scores[action] = edge_scores.get(edge_id, 0.0)
        scores[node_id] = node_scores
    return actions, scores
