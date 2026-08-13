"""Shared dynamic relay-importance computation for RL and baselines."""

from __future__ import annotations

import math
from collections import deque

import numpy as np


def compute_relay_importance(
    node_ids: list[str],
    physical_edge_ids: list[str],
    pending_requests,
    qkp_snapshot: dict[str, float],
    qkp_capacity: dict[str, float] | None,
    t: int,
    max_path_links: int = 3,
    hop_decay_factor: float = 0.25,
    capacity_strength: float = 1.0,
    min_scarcity: float = 0.0,
    wait_urgency_tau_ratio: float = 0.8,
    ignore_consumption: bool = False,
    include_stocked_unavailable: bool = True,
    all_edge_ids: list[str] | None = None,
) -> dict[str, float]:
    """BFS relay importance with queue-age weighting.

    A pending request contributes ``remaining * urgency`` to every physical
    edge that lies on a short relay path. Urgency grows with queue time:

    - ``wait_urgency_tau_ratio > 0``:
      ``tau = deadline_length * ratio`` and ``urgency = exp(age / tau)``
    - otherwise: linear ``1 + age / deadline_length``
    """
    active_ids = list(physical_edge_ids)
    potential_ids = set(active_ids)
    if include_stocked_unavailable:
        for edge_id in (all_edge_ids or []):
            if edge_id not in potential_ids and qkp_snapshot.get(edge_id, 0.0) > 1.0e-9:
                potential_ids.add(edge_id)

    endpoints: dict[str, tuple[str, str]] = {}
    for edge_id in potential_ids:
        body = edge_id[2:] if edge_id.startswith("E_") else edge_id
        if "__" in body:
            src, dst = body.split("__", 1)
            endpoints[edge_id] = (src, dst)

    adj: dict[str, list[str]] = {node: [] for node in node_ids}
    for src, dst in endpoints.values():
        adj.setdefault(src, []).append(dst)
        adj.setdefault(dst, []).append(src)

    pair_demand: dict[tuple[str, str], float] = {}
    for req in pending_requests:
        remaining = (
            float(req.amount)
            if ignore_consumption
            else max(0.0, req.amount - req.served_amount)
        )
        if remaining <= 1.0e-9:
            continue
        age = max(0, int(t) - int(req.arrival_t))
        deadline_length = max(1, int(req.deadline_t) - int(req.arrival_t))
        if wait_urgency_tau_ratio > 0.0:
            tau = max(1.0, float(deadline_length) * float(wait_urgency_tau_ratio))
            urgency = math.exp(age / tau)
        else:
            urgency = 1.0 + age / float(deadline_length)
        pair = tuple(sorted((req.src_gs, req.dst_gs)))
        pair_demand[pair] = pair_demand.get(pair, 0.0) + remaining * urgency

    if not pair_demand or not active_ids:
        return {}

    gs_dist: dict[str, dict[str, int]] = {}
    for pair in pair_demand:
        for gs in pair:
            if gs not in gs_dist:
                gs_dist[gs] = _bfs(gs, adj)

    qkp_capacity = qkp_capacity or {}
    node_index = {node: i for i, node in enumerate(node_ids)}
    n_nodes = len(node_ids)
    inf = 10**6
    edge_src = np.empty(len(active_ids), dtype=np.int64)
    edge_dst = np.empty(len(active_ids), dtype=np.int64)
    capacities = np.empty(len(active_ids), dtype=np.float64)
    levels = np.empty(len(active_ids), dtype=np.float64)
    for i, edge_id in enumerate(active_ids):
        u, v = endpoints[edge_id]
        edge_src[i] = node_index.get(u, -1)
        edge_dst[i] = node_index.get(v, -1)
        capacities[i] = float(qkp_capacity.get(edge_id, 0.0))
        levels[i] = float(qkp_snapshot.get(edge_id, 0.0))
    valid_nodes = (edge_src >= 0) & (edge_dst >= 0)

    gs_dist_arrays: dict[str, np.ndarray] = {}
    for gs, dist in gs_dist.items():
        arr = np.full(n_nodes, inf, dtype=np.float64)
        for node, d in dist.items():
            j = node_index.get(node)
            if j is not None:
                arr[j] = d
        gs_dist_arrays[gs] = arr

    pairs = list(pair_demand.items())
    src_dist_matrix = np.stack([gs_dist_arrays[pair[0]] for pair, _ in pairs])
    dst_dist_matrix = np.stack([gs_dist_arrays[pair[1]] for pair, _ in pairs])
    budgets = np.array([budget for _, budget in pairs], dtype=np.float64)

    scarcity = np.zeros_like(capacities)
    np.divide(
        capacities - levels,
        capacities,
        out=scarcity,
        where=capacities > 0.0,
    )
    scarcity = np.maximum(0.0, scarcity)
    if capacity_strength != 1.0:
        scarcity = np.power(scarcity, capacity_strength)
    scarcity += min_scarcity
    valid_scarcity = (capacities > 0.0) & (scarcity > 0.0)

    a = np.minimum(src_dist_matrix[:, edge_src], src_dist_matrix[:, edge_dst])
    b = np.minimum(dst_dist_matrix[:, edge_src], dst_dist_matrix[:, edge_dst])
    total_hops = a + b + 1
    mask = (
        (a < inf)
        & (b < inf)
        & (total_hops <= max_path_links)
        & valid_nodes[None, :]
        & valid_scarcity[None, :]
    )
    if not np.any(mask):
        return {}
    decay = np.power(hop_decay_factor, np.maximum(0, total_hops.astype(np.int64) - 2))
    totals = np.where(mask, budgets[:, None] * decay * scarcity[None, :], 0.0).sum(axis=0)

    if not np.any(totals > 0.0):
        return {}
    max_value = float(totals.max())
    return {
        edge_id: float(totals[i]) / max_value
        for i, edge_id in enumerate(active_ids)
        if totals[i] > 0.0
    }


def _bfs(start: str, adj: dict[str, list[str]]) -> dict[str, int]:
    dist = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in adj.get(node, ()):
            if nxt in dist:
                continue
            dist[nxt] = dist[node] + 1
            queue.append(nxt)
    return dist
