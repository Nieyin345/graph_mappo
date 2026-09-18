"""Shared dynamic relay-importance computation for RL and baselines."""

from __future__ import annotations

import math

import numpy as np

# ``edge_id -> (src, dst)`` parsed from the ``E_{src}__{dst}`` naming convention.
# The mapping is static for the whole run, but the function is called once per
# graph per step (~4x per slot in the batched rollout), and re-splitting every
# candidate edge id on every call was pure string work.
_EDGE_ENDPOINTS: dict[str, tuple[str, str]] = {}


def edge_endpoints(edge_id: str) -> tuple[str, str] | None:
    """Cached ``(src, dst)`` for an ``E_{src}__{dst}`` edge id (None if malformed)."""
    cached = _EDGE_ENDPOINTS.get(edge_id)
    if cached is not None:
        return cached
    body = edge_id[2:] if edge_id.startswith("E_") else edge_id
    if "__" not in body:
        return None
    src, dst = body.split("__", 1)
    _EDGE_ENDPOINTS[edge_id] = (src, dst)
    return src, dst


_INF = 10**6


def _distances_to_sources(
    n_nodes: int,
    src_pos: np.ndarray,
    dst_pos: np.ndarray,
    sources: list[int],
    node_ids: list[str],
) -> np.ndarray:
    """Hop distances from every source to every node, shape ``(len(sources), n_nodes)``.

    scipy's C-level unweighted shortest path over a CSR built from the integer
    endpoint positions. Measured against a hand-written integer-index Python BFS
    on the real candidate graph (~500 undirected edges once the keyed-but-idle
    links are added, ~13-26 sources, 90 nodes): 0.92 ms vs 2.39 ms per call.
    The Python BFS only wins on the much smaller *active-only* subgraph (~150
    edges, 0.36 ms), which is not the graph this function has to search -- do
    not "optimize" this back to a Python loop without re-measuring on the full
    candidate set.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    ok = (src_pos >= 0) & (src_pos < n_nodes) & (dst_pos >= 0) & (dst_pos < n_nodes)
    src_pos = src_pos[ok]
    dst_pos = dst_pos[ok]
    if not sources or src_pos.size == 0:
        return np.full((len(sources), n_nodes), float(_INF), dtype=np.float64)
    rows = np.concatenate([src_pos, dst_pos])
    cols = np.concatenate([dst_pos, src_pos])
    data = np.ones(rows.size, dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    dist = shortest_path(
        graph,
        method="D",
        unweighted=True,
        directed=False,
        indices=np.asarray(sources, dtype=np.int64),
    )
    dist = np.asarray(dist, dtype=np.float64).reshape(len(sources), n_nodes)
    return np.where(np.isfinite(dist), dist, float(_INF))


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
    link_type_bonus: dict[str, float] | None = None,
    active_src_pos: np.ndarray | None = None,
    active_dst_pos: np.ndarray | None = None,
    stocked_src_pos: np.ndarray | None = None,
    stocked_dst_pos: np.ndarray | None = None,
) -> dict[str, float]:
    """BFS relay importance with queue-age weighting.

    A pending request contributes ``remaining * urgency`` to every physical
    edge that lies on a short relay path. Urgency grows with queue time:

    - ``wait_urgency_tau_ratio > 0``:
      ``tau = deadline_length * ratio`` and ``urgency = exp(age / tau)``
    - otherwise: linear ``1 + age / deadline_length``

    Fast path (used by :class:`~qkd_rl.env.graph_builder.GraphBuilder`): pass
    ``active_src_pos`` / ``active_dst_pos`` (node positions aligned with
    ``physical_edge_ids``), ``stocked_edge_ids`` (the edge ids that actually hold
    keys -- already filtered, so no per-edge dict probe is needed) and
    ``stocked_src_pos`` / ``stocked_dst_pos``. Without them the function falls
    back to parsing the ``E_{src}__{dst}`` ids, which is what the offline
    baselines and unit tests do.
    """
    active_ids = list(physical_edge_ids)
    n_active = len(active_ids)
    node_index = {node: i for i, node in enumerate(node_ids)}
    n_nodes = len(node_ids)

    # --- candidate edge set: the active ones plus any that still hold keys -----
    #
    # Only the ADJACENCY needs the stocked edges (they are never scored), so the
    # caller can hand over their endpoint positions directly and skip both the id
    # parsing and the per-edge key-stock probe over the whole link registry.
    if active_src_pos is not None and active_dst_pos is not None:
        act_src = np.asarray(active_src_pos, dtype=np.int64)
        act_dst = np.asarray(active_dst_pos, dtype=np.int64)
    else:
        act_src = np.full(n_active, -1, dtype=np.int64)
        act_dst = np.full(n_active, -1, dtype=np.int64)
        for i, edge_id in enumerate(active_ids):
            parsed = edge_endpoints(edge_id)
            if parsed is None:
                continue
            act_src[i] = node_index.get(parsed[0], -1)
            act_dst[i] = node_index.get(parsed[1], -1)

    if stocked_src_pos is not None and stocked_dst_pos is not None:
        extra_src = np.asarray(stocked_src_pos, dtype=np.int64)
        extra_dst = np.asarray(stocked_dst_pos, dtype=np.int64)
    else:
        extra_src = np.empty(0, dtype=np.int64)
        extra_dst = np.empty(0, dtype=np.int64)
        if include_stocked_unavailable:
            seen = set(active_ids)
            src_list: list[int] = []
            dst_list: list[int] = []
            for edge_id in (all_edge_ids or []):
                if edge_id in seen or qkp_snapshot.get(edge_id, 0.0) <= 1.0e-9:
                    continue
                seen.add(edge_id)
                parsed = edge_endpoints(edge_id)
                if parsed is None:
                    continue
                src_list.append(node_index.get(parsed[0], -1))
                dst_list.append(node_index.get(parsed[1], -1))
            extra_src = np.asarray(src_list, dtype=np.int64)
            extra_dst = np.asarray(dst_list, dtype=np.int64)

    # --- per-request urgency, aggregated per GS pair --------------------------
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

    if not pair_demand or n_active == 0:
        return {}

    # --- distances once per distinct endpoint GS ------------------------------
    gs_needed: list[str] = []
    for pair in pair_demand:
        for gs in pair:
            if gs not in gs_needed:
                gs_needed.append(gs)
    gs_pos = [node_index[gs] for gs in gs_needed if gs in node_index]
    gs_present = [gs for gs in gs_needed if gs in node_index]
    all_src = np.concatenate([act_src, extra_src]) if extra_src.size else act_src
    all_dst = np.concatenate([act_dst, extra_dst]) if extra_dst.size else act_dst
    valid = (all_src >= 0) & (all_dst >= 0)
    dist_matrix = _distances_to_sources(
        n_nodes,
        all_src[valid],
        all_dst[valid],
        gs_pos,
        node_ids,
    )
    gs_row = {gs: i for i, gs in enumerate(gs_present)}

    qkp_capacity = qkp_capacity or {}
    capacities = np.fromiter(
        (float(qkp_capacity.get(e, 0.0)) for e in active_ids),
        dtype=np.float64,
        count=n_active,
    )
    levels = np.fromiter(
        (float(qkp_snapshot.get(e, 0.0)) for e in active_ids),
        dtype=np.float64,
        count=n_active,
    )
    scarcity = np.zeros_like(capacities)
    np.divide(capacities - levels, capacities, out=scarcity, where=capacities > 0.0)
    scarcity = np.maximum(0.0, scarcity)
    if capacity_strength != 1.0:
        scarcity = np.power(scarcity, capacity_strength)
    scarcity += min_scarcity
    valid_scarcity = (capacities > 0.0) & (scarcity > 0.0)
    if not np.any(valid_scarcity):
        return {}

    # --- accumulate over demand pairs ----------------------------------------
    # Batched over pairs instead of a per-pair Python loop. Each pair used to
    # do ~8 full-length (n_active) array ops; with ~16 pairs per step that
    # dominated the function (relay_importance measured 0.70 ms/step = 63% of
    # build_edge_features = ~48% of the whole graph build). Stacking the pairs
    # into a (n_pairs, n_active) matrix does the same arithmetic with one set
    # of numpy calls. Verified equivalent to the loop at 1.2e-10 max absolute
    # difference (float64 reduction-order noise) in .tmp/verify_relay_batch.py.
    rows_s: list[int] = []
    rows_d: list[int] = []
    budgets: list[float] = []
    for pair, budget in pair_demand.items():
        row_s = gs_row.get(pair[0])
        row_d = gs_row.get(pair[1])
        if row_s is None or row_d is None or budget <= 0.0:
            continue
        rows_s.append(row_s)
        rows_d.append(row_d)
        budgets.append(budget)
    if not rows_s:
        return {}

    src_ok = act_src >= 0
    dst_ok = act_dst >= 0
    d_s = dist_matrix[np.asarray(rows_s, dtype=np.int64)]      # (P, n_nodes)
    d_d = dist_matrix[np.asarray(rows_d, dtype=np.int64)]
    a = np.minimum(d_s[:, act_src], d_s[:, act_dst])           # (P, n_active)
    b = np.minimum(d_d[:, act_src], d_d[:, act_dst])
    total_hops = a + b + 1.0
    pair_mask = (
        (a < _INF)
        & (b < _INF)
        & (total_hops <= max_path_links)
        & src_ok
        & dst_ok
        & valid_scarcity
    )
    if not np.any(pair_mask):
        return {}
    decay = np.power(hop_decay_factor, np.maximum(0.0, total_hops - 2.0))
    budget_col = np.asarray(budgets, dtype=np.float64)[:, None]  # (P, 1)
    totals = np.where(pair_mask, budget_col * decay * scarcity, 0.0).sum(axis=0)

    if not np.any(totals > 0.0):
        return {}
    max_value = float(totals.max())
    if link_type_bonus:
        for i, edge_id in enumerate(active_ids):
            if totals[i] <= 0.0:
                continue
            parsed = edge_endpoints(edge_id)
            if parsed is None:
                continue

            def _typ(name: str) -> str:
                if name.startswith("Sat_") or name.startswith("SAT_"):
                    return "SAT"
                if name.startswith("HAP_"):
                    return "HAP"
                return "GS"

            totals[i] *= float(link_type_bonus.get(f"{_typ(parsed[0])}-{_typ(parsed[1])}", 1.0))
        max_value = float(totals.max())
    return {
        edge_id: float(totals[i]) / max_value
        for i, edge_id in enumerate(active_ids)
        if totals[i] > 0.0
    }
