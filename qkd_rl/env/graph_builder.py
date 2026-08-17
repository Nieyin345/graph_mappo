from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from qkd_rl.core.types import Edge, LinkType, Node, NodeType
from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.qkp import LinkQKPPool
from qkd_rl.env.request import DemandEdgeStats, RequestHistoryTracker, RequestQueue, wait_bucket_edges
from qkd_rl.env.relay_importance import compute_relay_importance
from qkd_rl.env.state import EnvState
from qkd_rl.link.rate_provider import RateNormalizer


_EVER_AVAILABLE_CACHE: dict[tuple, frozenset[str]] = {}
_RELAY_EDGE_INFO_CACHE: dict[tuple, dict[tuple[str, str], list[tuple[str, int]]]] = {}


def _ever_available_cache_file(rate_provider) -> str:
    """Disk cache path for the ever-available scan, next to the H5 dataset."""
    path = getattr(rate_provider, "link_data_path", None)
    if path is None:
        return ""
    return str(Path(path).with_name("ever_available_cache.json"))


def _load_ever_available_disk_cache(cache_key: tuple, cache_file: str) -> frozenset[str] | None:
    """Load the persisted ever-available edge set for ``cache_key`` (None on miss)."""
    if not cache_file:
        return None
    key_hash = hashlib.md5(repr(cache_key).encode("utf-8")).hexdigest()
    try:
        with open(cache_file, encoding="utf-8") as handle:
            payload = json.load(handle)
        edges = payload.get(key_hash)
        return frozenset(edges) if edges is not None else None
    except (OSError, ValueError, TypeError):
        return None


def _save_ever_available_disk_cache(cache_key: tuple, edges: frozenset[str], cache_file: str) -> None:
    if not cache_file:
        return
    key_hash = hashlib.md5(repr(cache_key).encode("utf-8")).hexdigest()
    try:
        payload: dict = {}
        if Path(cache_file).exists():
            with open(cache_file, encoding="utf-8") as handle:
                payload = json.load(handle)
        payload[key_hash] = sorted(edges)
        with open(cache_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except (OSError, ValueError, TypeError):
        pass


@dataclass
class GraphObservation:
    node_features: list[list[float]]
    edge_index: list[tuple[int, int]]
    edge_features: list[list[float]]
    node_ids: list[str]
    edge_ids: list[str]
    physical_edge_ids: list[str]
    demand_edge_ids: list[str]
    action_candidates: dict[str, list[str]]
    action_masks: dict[str, list[bool]]
    state: EnvState
    # Raw per-node masks (one bool per static candidate, unfiltered) from
    # ActionMaskBuilder. The actor vectorizes legal-candidate selection from
    # these; None falls back to the Python loop path (manual test fixtures).
    raw_action_masks: dict[str, list[bool]] | None = None
    # Flat concatenation of raw_action_masks in node/candidate order, reused
    # by the actor plan to avoid rebuilding the same flat mask per step.
    flat_action_masks: np.ndarray | None = None
    # Per-entity history windows for the shared HistoryEncoder. Each element is
    # [entity][time_step][channel]; rows align with node_ids / physical_edge_ids
    # / demand_edge_ids. Empty lists when history_encoder is disabled or the
    # entity batch is empty. history_valid = [node_valids, phys_valids,
    # demand_valids] carries cold-start observed lengths.
    node_history: list[list[list[float]]] = field(default_factory=list)
    physical_edge_history: list[list[list[float]]] = field(default_factory=list)
    demand_edge_history: list[list[list[float]]] = field(default_factory=list)
    history_valid: list[list[int]] = field(default_factory=lambda: [[], [], []])


class GraphBuilder:
    def __init__(
        self,
        nodes: list[Node],
        edges: list[Edge],
        action_space: NodeActionSpace,
        qkp: LinkQKPPool,
        normalizer: RateNormalizer,
        config: dict,
        history_buffer=None,
        routing=None,
        rate_provider=None,
    ):
        self.nodes = nodes
        self.edges = edges
        self._node_ids_list = [node.node_id for node in nodes]
        self.action_space = action_space
        self.qkp = qkp
        self.normalizer = normalizer
        self.config = config
        self.history_buffer = history_buffer
        self.routing = routing
        self.rate_provider = rate_provider
        self.node_index = {node.node_id: idx for idx, node in enumerate(nodes)}
        self.gs_ids = [node.node_id for node in nodes if node.node_type == NodeType.GS]
        # Precomputed lookups avoid O(nodes x edges) scans and repeated
        # candidate-list construction on every step.
        node_ids = [node.node_id for node in nodes]
        self.candidates: dict[str, list[str]] = {
            node_id: self.action_space.candidates_for_node(node_id) for node_id in node_ids
        }
        self.action_pos: dict[tuple[str, str], int] = {}
        for node_id, cands in self.candidates.items():
            for pos, action in enumerate(cands):
                self.action_pos[(node_id, action)] = pos
        self.incident_by_node: dict[str, list[Edge]] = {
            node_id: [edge for edge in edges if node_id in (edge.src, edge.dst)] for node_id in node_ids
        }
        # Edge-id lists and static total capacity per node, so node feature
        # sums never re-scan the topology or call get_capacity per step.
        self._node_incident_ids: dict[str, list[str]] = {
            node_id: [edge.edge_id for edge in incident] for node_id, incident in self.incident_by_node.items()
        }
        capacities = self.qkp.capacities
        self._node_total_capacity: dict[str, float] = {
            node_id: sum(capacities[edge_id] for edge_id in self._node_incident_ids[node_id]) or 1.0
            for node_id in self._node_incident_ids
        }
        # Static per-edge parts for the vectorized edge-feature path.
        self._edge_list: list[str] = [edge.edge_id for edge in edges]
        self._edge_pos: dict[str, int] = {edge_id: i for i, edge_id in enumerate(self._edge_list)}
        # Link-id gather indices for self._edge_list; static for the whole run
        # (registry mapping never changes), filled lazily on the first build so
        # build_node_features / _build_physical_edge_rows_vectorized skip the
        # per-step np.fromiter over all registered links.
        self._edge_list_link_ids: np.ndarray | None = None
        include_one_hot = bool(self.config["features"]["edge"].get("include_link_type_one_hot", True))
        self._edge_static: dict[str, list[float]] = {
            edge.edge_id: one_hot_link_type(edge.link_type) if include_one_hot else [] for edge in edges
        }
        static_dim = len(self._edge_static[self._edge_list[0]]) if self._edge_list else 0
        self._edge_static_arr = np.array(
            [self._edge_static[edge_id] for edge_id in self._edge_list], dtype=np.float32
        )
        # Static per-edge QKP pool capacity (never changes), aligned with
        # self._edge_list for the per-link qkp_capacity_left feature.
        self._edge_capacity_arr = np.array(
            [self.qkp.capacities[edge_id] for edge_id in self._edge_list], dtype=np.float64
        )
        # Static per-node tables for the vectorized node-feature path: the
        # node-type one-hot and total capacity are fixed for the whole run.
        num_nodes = len(nodes)
        self._node_type_one_hot = np.array(
            [one_hot_node_type(node.node_type) for node in nodes], dtype=np.float32
        )
        self._node_total_capacity_arr = np.array(
            [self._node_total_capacity[node.node_id] for node in nodes], dtype=np.float32
        )
        # Bidirectional shortest-hop candidate edges per GS pair. An edge is a
        # relay candidate when both ground stations can reach it and the total
        # path through it is within max_path_links. Longer total hops get a
        # decayed weight, so short paths are preferred without enumerating
        # every path.
        self._relay_edge_info: dict[tuple[str, str], list[tuple[str, int]]] = {}
        self._last_relay_importance: dict[str, float] = {}
        edge_cfg = config["features"]["edge"]
        if edge_cfg.get("include_relay_importance", False) and self.routing is not None:
            relay_cfg = edge_cfg.get("relay_importance", {})
            max_links = int(relay_cfg.get("max_path_links", 4))
            # In code terms a/b are hops to the edge endpoints (the candidate
            # link itself is not included), so each side is bounded by k-1.
            side_limit = max(0, max_links - 1)
            ever_available = self._ever_available_edges(relay_cfg)
            cache_key = (
                tuple((node.node_id, node.node_type.value) for node in self.nodes),
                tuple(
                    (edge.edge_id, edge.src, edge.dst, edge.link_type.value)
                    for edge in self.edges
                ),
                max_links,
                ever_available,
            )
            cached = _RELAY_EDGE_INFO_CACHE.get(cache_key)
            if cached is not None:
                self._relay_edge_info = {
                    pair: list(entries) for pair, entries in cached.items()
                }
            else:
                for pair in all_gs_pairs(self.gs_ids):
                    src, dst = pair
                    entries: list[tuple[str, int]] = []
                    for edge in self.edges:
                        if ever_available is not None and edge.edge_id not in ever_available:
                            continue
                        ds_u = self.routing.hop_distance(src, edge.src)
                        ds_v = self.routing.hop_distance(src, edge.dst)
                        dt_u = self.routing.hop_distance(dst, edge.src)
                        dt_v = self.routing.hop_distance(dst, edge.dst)
                        disconnected = self.routing.DISCONNECTED
                        if min(ds_u, ds_v) > side_limit or min(dt_u, dt_v) > side_limit:
                            continue
                        if (
                            min(ds_u, ds_v) >= disconnected
                            or min(dt_u, dt_v) >= disconnected
                        ):
                            continue
                        total_hops = min(ds_u, ds_v) + min(dt_u, dt_v) + 1
                        if total_hops <= max_links:
                            entries.append((edge.edge_id, total_hops))
                    if entries:
                        self._relay_edge_info[pair] = sorted(
                            entries, key=lambda item: (item[1], item[0])
                        )
                _RELAY_EDGE_INFO_CACHE[cache_key] = {
                    pair: list(entries)
                    for pair, entries in self._relay_edge_info.items()
                }

        # Each undirected edge contributes to its two endpoint nodes; flat
        # endpoint-position arrays allow one bincount-based qkp/availability
        # reduction per step instead of per-node gathers.
        self._inc_src_pos = np.array([self.node_index[edge.src] for edge in edges], dtype=np.int64)
        self._inc_dst_pos = np.array([self.node_index[edge.dst] for edge in edges], dtype=np.int64)
        # Vectorized mask-first edge filtering: every edge maps to the two flat
        # mask positions of its directed actions (-1 when not actionable), so
        # the per-step active-edge test is a numpy gather instead of a Python
        # loop of dict lookups over all links.
        self._mask_offsets: dict[str, int] = {}
        offset = 0
        for node_id in node_ids:
            self._mask_offsets[node_id] = offset
            offset += len(self.candidates[node_id])
        self._mask_total = offset
        edge_flat_src = np.full(len(edges), -1, dtype=np.int64)
        edge_flat_dst = np.full(len(edges), -1, dtype=np.int64)
        for edge_pos, edge in enumerate(edges):
            pos_src = self.action_pos.get((edge.src, edge.dst), -1)
            pos_dst = self.action_pos.get((edge.dst, edge.src), -1)
            if pos_src >= 0:
                edge_flat_src[edge_pos] = self._mask_offsets[edge.src] + pos_src
            if pos_dst >= 0:
                edge_flat_dst[edge_pos] = self._mask_offsets[edge.dst] + pos_dst
        self._edge_flat_src = edge_flat_src
        self._edge_flat_dst = edge_flat_dst
        # Per-node positions inside self._edge_list: qkp levels and link
        # availability are gathered once per step and reduced per node without
        # repeated dict lookups.
        self._node_incident_pos: dict[str, np.ndarray] = {
            node_id: np.array([self._edge_pos[edge_id] for edge_id in incident], dtype=np.int64)
            for node_id, incident in self._node_incident_ids.items()
        }

    def _ever_available_edges(self, relay_cfg: dict) -> frozenset[str] | None:
        """Edges observed available in a sampled scan of the rate provider.

        Links that never appear in the dataset are useless relay hops; they are
        dropped from the one-hop path enumeration entirely.
        """
        if self.rate_provider is None:
            return None
        stride = max(1, int(relay_cfg.get("availability_sample_stride", 600)))
        limit = max(1, int(relay_cfg.get("availability_sample_limit", 876)))
        edge_ids = tuple(e.edge_id for e in self.edges)
        cache_key = (
            str(getattr(self.rate_provider, "link_data_path", "")),
            float(getattr(self.rate_provider, "min_link_rate", 0.0)),
            stride,
            limit,
            edge_ids,
        )
        if cache_key in _EVER_AVAILABLE_CACHE:
            return _EVER_AVAILABLE_CACHE[cache_key]
        cache_file = _ever_available_cache_file(self.rate_provider)
        cached = _load_ever_available_disk_cache(cache_key, cache_file)
        if cached is not None:
            _EVER_AVAILABLE_CACHE[cache_key] = cached
            return cached
        link_ids = self.rate_provider.edge_link_ids(list(edge_ids))
        edge_ids_np = np.asarray(edge_ids)
        available: set[str] = set()
        for t in range(0, min(int(stride * limit), int(getattr(self.rate_provider, "_T", 0))), stride):
            blocks = self.rate_provider.get_window_blocks(t)
            now = blocks[1][0]
            present = np.flatnonzero(now[link_ids])
            available.update(edge_ids_np[present].tolist())
        result = frozenset(available)
        _EVER_AVAILABLE_CACHE[cache_key] = result
        _save_ever_available_disk_cache(cache_key, result, cache_file)
        return result

    def build(
        self,
        env_state: EnvState,
        requests: RequestQueue,
        request_history: RequestHistoryTracker,
        masks: dict[str, list[bool]],
        flat_masks: np.ndarray | None = None,
    ) -> GraphObservation:
        # Pair stats are computed once and reused for both pair selection and
        # the demand-edge feature rows (previously computed twice per step).
        demand_cfg = self.config["features"].get("demand_edge", {})
        demand_stats = None
        if demand_cfg.get("enabled", False):
            demand_stats = requests.stats_by_pair(
                env_state.t,
                request_history,
                None,
                wait_bucket_count=int(demand_cfg.get("wait_bucket_count", 10)),
                deadline_steps=int(self.config["requests"].get("deadline_steps", 960.0)),
            )
        demand_pairs = self.build_demand_pairs(env_state, requests, request_history, demand_stats=demand_stats)
        demand_edge_ids = [demand_edge_id(pair) for pair in demand_pairs]
        # Mask-first filtering: links that are not legal for either endpoint are
        # dropped from the graph entirely (edge_index / edge_features /
        # physical_edge_ids), so the GNN only sees currently usable links.
        active_edges, active_pos = self._active_edges(masks, flat_masks)
        action_candidates: dict[str, list[str]] = {}
        action_masks: dict[str, list[bool]] = {}
        for node in self.nodes:
            node_id = node.node_id
            full_candidates = self.candidates[node_id]
            legal = [action for action, ok in zip(full_candidates, masks[node_id]) if ok]
            action_candidates[node_id] = legal
            action_masks[node_id] = [True] * len(legal)
        if self.history_buffer is not None:
            node_history, node_valid = self.history_buffer.node_sequences(self.nodes)
            phys_history, phys_valid = self.history_buffer.physical_sequences(active_edges)
            demand_history, demand_valid = self.history_buffer.demand_sequences(demand_pairs)
            history_valid = [node_valid, phys_valid, demand_valid]
        else:
            node_history, phys_history, demand_history = [], [], []
            history_valid = [[], [], []]
        return GraphObservation(
            node_features=self.build_node_features(env_state, requests, request_history),
            edge_index=self.build_edge_index(active_edges, demand_pairs),
            edge_features=self.build_edge_features(
                active_edges, active_pos, env_state, requests, request_history, demand_pairs,
                demand_stats_by_pair=demand_stats,
            ),
            node_ids=self._node_ids_list,
            edge_ids=[edge.edge_id for edge in active_edges] + demand_edge_ids,
            physical_edge_ids=[edge.edge_id for edge in active_edges],
            demand_edge_ids=demand_edge_ids,
            action_candidates=action_candidates,
            action_masks=action_masks,
            raw_action_masks=masks,
            flat_action_masks=flat_masks if flat_masks is not None and flat_masks.size == self._mask_total else None,
            state=env_state,
            node_history=node_history,
            physical_edge_history=phys_history,
            demand_edge_history=demand_history,
            history_valid=history_valid,
        )

    def _flat_mask(
        self, masks: dict[str, list[bool]], flat_masks: np.ndarray | None = None
    ) -> np.ndarray:
        """Flatten per-node mask lists into one array aligned with candidate
        positions so active-edge filtering and legality checks are vectorized."""
        if flat_masks is not None and flat_masks.size == self._mask_total:
            return flat_masks
        flat = np.empty(self._mask_total, dtype=bool)
        for node_id, mask_list in masks.items():
            start = self._mask_offsets[node_id]
            flat[start:start + len(mask_list)] = mask_list
        return flat

    def _active_edges(
        self, masks: dict[str, list[bool]], flat_masks: np.ndarray | None = None
    ) -> tuple[list[Edge], np.ndarray]:
        """Return (edges, positions) whose two directed actions are both legal
        (mask-first graph filtering). Positions index into ``self._edge_list``
        and are reused for the static per-edge feature gathers."""
        if self._edge_flat_src.size == 0:
            return [], np.array([], dtype=np.int64)
        flat = self._flat_mask(masks, flat_masks)
        src = self._edge_flat_src
        dst = self._edge_flat_dst
        legal = (src >= 0) & (dst >= 0)
        legal &= flat[np.maximum(src, 0)]
        legal &= flat[np.maximum(dst, 0)]
        positions = np.flatnonzero(legal)
        return [self.edges[i] for i in positions], positions

    def _edge_active(self, edge: Edge, masks: dict[str, list[bool]]) -> bool:
        """A physical link is kept iff both endpoints may legally activate it."""
        if (edge.src, edge.dst) not in self.action_pos or (edge.dst, edge.src) not in self.action_pos:
            return False
        return bool(
            masks[edge.src][self.action_pos[(edge.src, edge.dst)]]
            and masks[edge.dst][self.action_pos[(edge.dst, edge.src)]]
        )

    def build_node_features(
        self,
        env_state: EnvState,
        requests: RequestQueue,
        request_history: RequestHistoryTracker,
    ) -> np.ndarray:
        """Build node features following ``features.node`` switches.

        The feature order matches ``ConfigValidator.resolve_feature_dims`` so
        resolved dimensions stay consistent when switches are turned off.
        Returns a ``(num_nodes, node_dim)`` float32 array; the per-node Python
        loop only writes scalar columns, all reductions are vectorized.
        """
        node_cfg = self.config["features"]["node"]
        node_dim = int(self.config["features"]["dims"]["node_dim_resolved"])
        windows = [int(value) for value in node_cfg.get("recent_demand_windows", [])]
        amount_scale = float(node_cfg.get("normalize_amount_by", 1000.0)) or 1.0
        log1p_amount = bool(node_cfg.get("normalize_amount_log1p", False))
        amount_log_denom = math.log1p(float(node_cfg.get("normalize_amount_reference", amount_scale)))
        count_scale = float(node_cfg.get("normalize_count_by", 10.0)) or 1.0
        log1p_count = bool(node_cfg.get("normalize_count_log1p", False))
        count_log_denom = math.log1p(float(node_cfg.get("normalize_count_reference", count_scale)))
        nodes = self.nodes
        num_nodes = len(nodes)
        node_index = self.node_index
        demand_in = requests.demand_in_by_node()
        demand_out = requests.demand_out_by_node()
        pending_count = requests.pending_count_by_node()

        levels = self.qkp.levels
        level_arr = np.fromiter(
            (levels[edge_id] for edge_id in self._edge_list), dtype=np.float64, count=len(self._edge_list)
        )
        # One flat reduction per step: each node's total QKP level is the sum
        # over its incident edges (each edge contributes to its two endpoints).
        total_level = np.bincount(self._inc_src_pos, weights=level_arr, minlength=num_nodes)
        total_level += np.bincount(self._inc_dst_pos, weights=level_arr, minlength=num_nodes)
        ewindows = env_state.edge_windows
        has_vector_avail = bool(node_cfg.get("include_is_available", True)) and hasattr(ewindows, "blocks")
        avail_node = None
        if has_vector_avail:
            # bincount-with-weights is C-speed; ufunc.at would loop in Python.
            if self._edge_list_link_ids is None:
                self._edge_list_link_ids = ewindows.link_ids(self._edge_list)
            avail_arr = np.asarray(ewindows.blocks[1][0][self._edge_list_link_ids], dtype=bool)
            avail_count = np.bincount(
                self._inc_src_pos, weights=avail_arr.astype(np.float32), minlength=num_nodes
            )
            avail_count += np.bincount(
                self._inc_dst_pos, weights=avail_arr.astype(np.float32), minlength=num_nodes
            )
            avail_node = avail_count > 0
        # Dict -> array once per step; the per-node loop then only indexes.
        demand_in_arr = np.zeros(num_nodes, dtype=np.float32)
        for nid, value in demand_in.items():
            demand_in_arr[node_index[nid]] = value
        demand_out_arr = np.zeros(num_nodes, dtype=np.float32)
        for nid, value in demand_out.items():
            demand_out_arr[node_index[nid]] = value
        pending_arr = np.zeros(num_nodes, dtype=np.float32)
        for nid, value in pending_count.items():
            pending_arr[node_index[nid]] = value
        # Time features are node-independent: compute once instead of per node.
        time_cols: list[float] = []
        if node_cfg.get("include_time_features", False):
            time_cfg = node_cfg.get("time_features", {})
            minute = env_state.t % 1440
            day = env_state.t % 525600
            if time_cfg.get("minute_of_day_sin_cos", False):
                time_cols.extend([math.sin(2 * math.pi * minute / 1440), math.cos(2 * math.pi * minute / 1440)])
            if time_cfg.get("day_of_year_sin_cos", False):
                time_cols.extend([math.sin(2 * math.pi * day / 525600), math.cos(2 * math.pi * day / 525600)])

        # Column-wise vectorized assembly: each feature block is one array
        # write over all nodes; only recent-demand sums stay per node.
        features = np.zeros((num_nodes, node_dim), dtype=np.float32)
        node_ids = self._node_ids_list
        cap_arr = self._node_total_capacity_arr
        col = 0
        if node_cfg.get("include_node_type_one_hot", True):
            features[:, col:col + 3] = self._node_type_one_hot
            col += 3
        if node_cfg.get("include_qkp_level", True):
            features[:, col] = total_level / cap_arr
            col += 1
        if node_cfg.get("include_qkp_capacity_left", True):
            features[:, col] = (cap_arr - total_level) / cap_arr
            col += 1
        if node_cfg.get("include_qkp_utilization", True):
            features[:, col] = total_level / cap_arr
            col += 1
        if node_cfg.get("include_demand_in", True):
            features[:, col] = (
                np.log1p(np.maximum(demand_in_arr, 0.0)) / amount_log_denom
                if log1p_amount
                else demand_in_arr / amount_scale
            )
            col += 1
        if node_cfg.get("include_demand_out", True):
            features[:, col] = (
                np.log1p(np.maximum(demand_out_arr, 0.0)) / amount_log_denom
                if log1p_amount
                else demand_out_arr / amount_scale
            )
            col += 1
        if node_cfg.get("include_queue_pressure", True):
            features[:, col] = (
                np.log1p(np.maximum(pending_arr, 0.0)) / count_log_denom
                if log1p_count
                else pending_arr / count_scale
            )
            col += 1
        if node_cfg.get("include_is_available", True):
            if avail_node is not None:
                features[:, col] = avail_node.astype(np.float32)
            else:
                for i, node in enumerate(nodes):
                    features[i, col] = float(
                        any(ewindows[edge_id].available[0] for edge_id in self._node_incident_ids[node.node_id])
                    )
            col += 1
        if node_cfg.get("include_recent_demand", False):
            for window in windows:
                window_sums = np.fromiter(
                    (request_history.sum_for_node(node_id, "arrived", env_state.t, window) for node_id in node_ids),
                    dtype=np.float32,
                    count=num_nodes,
                )
                features[:, col] = (
                    np.log1p(np.maximum(window_sums, 0.0)) / amount_log_denom
                    if log1p_amount
                    else window_sums / amount_scale
                )
                col += 1
        if time_cols:
            features[:, col:col + len(time_cols)] = time_cols
            col += len(time_cols)
        if node_cfg.get("include_position", False):
            pos_rows = np.asarray([self._position_features(node) for node in nodes], dtype=np.float32)
            features[:, col:col + pos_rows.shape[1]] = pos_rows
            col += pos_rows.shape[1]
        if col != node_dim:
            raise ValueError(f"Node feature dim mismatch: {col} != {node_dim}")
        return features

    def build_edge_index(
        self, active_edges: list[Edge], demand_pairs: list[tuple[str, str]]
    ) -> np.ndarray:
        """Directed edge pairs as a ``(2 * (n_phys + n_demand), 2)`` int64
        array: physical edges first as (src, dst) then (dst, src), followed by
        the same doubled representation for demand pairs."""
        n_phys = len(active_edges)
        n_demand = len(demand_pairs)
        index = np.empty((2 * (n_phys + n_demand), 2), dtype=np.int64)
        if n_phys:
            srcs = np.fromiter(
                (self.node_index[edge.src] for edge in active_edges), dtype=np.int64, count=n_phys
            )
            dsts = np.fromiter(
                (self.node_index[edge.dst] for edge in active_edges), dtype=np.int64, count=n_phys
            )
            index[0 : 2 * n_phys : 2, 0] = srcs
            index[0 : 2 * n_phys : 2, 1] = dsts
            index[1 : 2 * n_phys : 2, 0] = dsts
            index[1 : 2 * n_phys : 2, 1] = srcs
        if n_demand:
            srcs = np.fromiter(
                (self.node_index[src_id] for src_id, _ in demand_pairs), dtype=np.int64, count=n_demand
            )
            dsts = np.fromiter(
                (self.node_index[dst_id] for _, dst_id in demand_pairs), dtype=np.int64, count=n_demand
            )
            start = 2 * n_phys
            index[start : start + 2 * n_demand : 2, 0] = srcs
            index[start : start + 2 * n_demand : 2, 1] = dsts
            index[start + 1 : start + 2 * n_demand : 2, 0] = dsts
            index[start + 1 : start + 2 * n_demand : 2, 1] = srcs
        return index

    def build_edge_features(
        self,
        active_edges: list[Edge],
        active_pos: np.ndarray,
        env_state: EnvState,
        requests: RequestQueue,
        request_history: RequestHistoryTracker,
        demand_pairs: list[tuple[str, str]],
        demand_stats_by_pair: dict[tuple[str, str], "DemandEdgeStats"] | None = None,
    ) -> np.ndarray:
        """Build physical + demand edge features following ``features.edge``
        and ``features.demand_edge`` switches. Returns a
        ``(n_phys + n_demand, physical_dim + demand_dim)`` float32 array."""
        edge_cfg = self.config["features"]["edge"]
        physical_dim = int(self.config["features"]["dims"]["physical_edge_dim_resolved"])
        demand_dim = int(self.config["features"]["dims"]["demand_edge_dim_resolved"])
        relay_cfg = edge_cfg.get("relay_importance", {})
        include_stocked_unavailable = bool(
            relay_cfg.get("include_stocked_unavailable", True)
        )
        relay_importance = compute_relay_importance(
            node_ids=self._node_ids_list,
            physical_edge_ids=[edge.edge_id for edge in active_edges],
            pending_requests=requests.get_pending(),
            qkp_snapshot=self.qkp.snapshot(),
            qkp_capacity=self.qkp.capacities,
            t=env_state.t,
            max_path_links=int(relay_cfg.get("max_path_links", 3)),
            hop_decay_factor=float(relay_cfg.get("hop_decay_factor", 0.25)),
            capacity_strength=float(relay_cfg.get("capacity_decay_strength", 1.0)),
            min_scarcity=float(relay_cfg.get("min_scarcity", 0.0)),
            wait_urgency_tau_ratio=float(relay_cfg.get("wait_urgency_tau_ratio", 0.8)),
            ignore_consumption=bool(relay_cfg.get("ignore_consumption", False)),
            include_stocked_unavailable=include_stocked_unavailable,
            all_edge_ids=(
                list(env_state.edge_windows.keys()) if include_stocked_unavailable else None
            ),
            link_type_bonus=relay_cfg.get("link_type_bonus", None),
        )
        self._last_relay_importance = relay_importance or {}
        ewindows = env_state.edge_windows
        blocks = getattr(ewindows, "blocks", None)
        if blocks is not None and blocks[2] is not None:
            rows = self._build_physical_edge_rows_vectorized(
                active_edges,
                active_pos,
                env_state,
                blocks,
                physical_dim,
                demand_dim,
                relay_importance=relay_importance,
            )
        else:
            rows = self._build_physical_edge_rows_loop(
                active_edges,
                env_state,
                edge_cfg,
                physical_dim,
                demand_dim,
                relay_importance=relay_importance,
            )
        dem_rows = self.build_demand_edge_features(
            env_state, requests, request_history, demand_pairs, demand_stats_by_pair=demand_stats_by_pair
        )
        if len(dem_rows):
            dem_padded = np.concatenate(
                [np.zeros((len(dem_rows), physical_dim), dtype=np.float32), dem_rows], axis=1
            )
            if len(rows):
                return np.concatenate([rows, dem_padded], axis=0)
            return dem_padded
        return rows

    def _build_physical_edge_rows_vectorized(
        self,
        active_edges: list[Edge],
        active_pos: np.ndarray,
        env_state: EnvState,
        blocks,
        physical_dim: int,
        demand_dim: int,
        relay_importance: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Physical edge feature rows assembled from the cached numpy blocks.

        All window-derived columns (availability, normalized rates, delta,
        mean, max) are computed vectorized over the active edges; the result is
        one ``(n_active, physical_dim + demand_dim)`` float32 array (static
        one-hot, dynamic columns, and the demand-dimension zero padding).
        """
        edge_cfg = self.config["features"]["edge"]
        horizon = int(edge_cfg.get("prediction_horizon", 0))
        rates_clean, avail, rates_norm = blocks
        active_ids = [edge.edge_id for edge in active_edges]
        if not active_ids:
            return np.zeros((0, physical_dim + demand_dim), dtype=np.float32)
        if self._edge_list_link_ids is None:
            self._edge_list_link_ids = env_state.edge_windows.link_ids(self._edge_list)
        link_idx = self._edge_list_link_ids[active_pos]
        rn = rates_norm
        av = avail
        cols: list[np.ndarray] = []
        if edge_cfg.get("include_available_now", True):
            cols.append(av[0][link_idx])
        if edge_cfg.get("include_rate_now", True):
            cols.append(rn[0][link_idx])
        if edge_cfg.get("include_rate_future_window", True) and horizon > 0:
            cols.append(rn[1 : horizon + 1, link_idx].T)
        if edge_cfg.get("include_available_future_window", True) and horizon > 0:
            cols.append(av[1 : horizon + 1, link_idx].T)
        if edge_cfg.get("include_rate_delta", True):
            cols.append((rn[-1] - rn[0])[link_idx])
        if edge_cfg.get("include_rate_mean", True):
            cols.append(rn.mean(axis=0)[link_idx])
        if edge_cfg.get("include_rate_max", True):
            cols.append(rn.max(axis=0)[link_idx])
        # Vectorized membership for the activation-history columns.
        active_np = np.asarray(active_ids)
        last_set = set(env_state.last_activated_edges)
        if edge_cfg.get("include_last_activated", True):
            if last_set:
                cols.append(np.isin(active_np, np.asarray(list(last_set))).astype(np.float32))
            else:
                cols.append(np.zeros(len(active_ids), dtype=np.float32))
        if edge_cfg.get("include_relay_importance", False):
            importance = relay_importance or {}
            cols.append(
                np.fromiter(
                    (importance.get(edge_id, 0.0) for edge_id in active_ids),
                    dtype=np.float32,
                    count=len(active_ids),
                )
            )
        # Per-link remaining QKP capacity ratio (capacity - level) / capacity.
        if edge_cfg.get("include_qkp_capacity_left", False):
            cap = self._edge_capacity_arr[active_pos]
            lvl = np.fromiter(
                (self.qkp.levels.get(edge.edge_id, 0.0) for edge in active_edges),
                dtype=np.float64,
                count=len(active_edges),
            )
            cap_left = np.divide(cap - lvl, cap, out=np.zeros_like(cap), where=cap > 0)
            cols.append(cap_left.astype(np.float32))
        dyn = np.column_stack(cols) if cols else np.zeros((len(active_ids), 0), dtype=np.float32)
        static_rows = self._edge_static_arr[active_pos]
        rows = np.concatenate(
            [static_rows, dyn, np.zeros((len(active_ids), demand_dim), dtype=np.float32)], axis=1
        )
        if rows.shape[1] != physical_dim + demand_dim:
            raise ValueError(
                f"Physical edge feature dim mismatch: {rows.shape[1]} != {physical_dim + demand_dim}"
            )
        return rows

    def _build_physical_edge_rows_loop(
        self,
        active_edges: list[Edge],
        env_state: EnvState,
        edge_cfg: dict,
        physical_dim: int,
        demand_dim: int,
        relay_importance: dict[str, float] | None = None,
    ) -> list[list[float]]:
        """Legacy per-window loop for directly-constructed windows (tests)."""
        horizon = int(edge_cfg.get("prediction_horizon", 0))
        rows: list[list[float]] = []
        for edge in active_edges:
            window = env_state.edge_windows[edge.edge_id]
            rates_norm = window.rates_norm
            if rates_norm is None:
                rates_norm = [
                    self.normalizer.transform_scalar(rate, edge.link_type.value) for rate in window.rates
                ]
            row: list[float] = []
            if edge_cfg.get("include_link_type_one_hot", True):
                row.extend(one_hot_link_type(edge.link_type))
            if edge_cfg.get("include_available_now", True):
                row.append(float(window.available[0]))
            if edge_cfg.get("include_rate_now", True):
                row.append(rates_norm[0])
            if edge_cfg.get("include_rate_future_window", True) and horizon > 0:
                row.extend(rates_norm[1 : horizon + 1])
            if edge_cfg.get("include_available_future_window", True) and horizon > 0:
                row.extend(float(value) for value in window.available[1 : horizon + 1])
            if edge_cfg.get("include_rate_delta", True):
                row.append(rates_norm[-1] - rates_norm[0] if len(rates_norm) > 1 else 0.0)
            if edge_cfg.get("include_rate_mean", True):
                row.append(sum(rates_norm) / len(rates_norm))
            if edge_cfg.get("include_rate_max", True):
                row.append(max(rates_norm))
            if edge_cfg.get("include_last_activated", True):
                row.append(float(edge.edge_id in env_state.last_activated_edges))
            if edge_cfg.get("include_relay_importance", False):
                row.append(float((relay_importance or {}).get(edge.edge_id, 0.0)))
            if edge_cfg.get("include_qkp_capacity_left", False):
                capacity = self.qkp.get_capacity(edge.edge_id)
                level = self.qkp.get_level(edge.edge_id)
                row.append((capacity - level) / capacity if capacity > 0 else 0.0)
            row.extend([0.0] * demand_dim)
            if len(row) != physical_dim + demand_dim:
                raise ValueError(f"Physical edge feature dim mismatch: {len(row)} != {physical_dim + demand_dim}")
            rows.append(row)
        if not rows:
            return np.zeros((0, physical_dim + demand_dim), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def build_demand_pairs(
        self,
        env_state: EnvState,
        requests: RequestQueue,
        request_history: RequestHistoryTracker,
        demand_stats: dict[tuple[str, str], "DemandEdgeStats"] | None = None,
    ) -> list[tuple[str, str]]:
        cfg = self.config["features"].get("demand_edge", {})
        if not cfg.get("enabled", False):
            return []
        mode = cfg.get("build_mode", "active_pairs")
        if mode == "all_gs_pairs":
            return all_gs_pairs(self.gs_ids)
        if mode == "active_pairs":
            return sorted(requests.demand_by_pair())
        raise ValueError(f"Unknown demand edge build mode: {mode}")

    def _wait_bucket_edges(self) -> list[int]:
        cfg = self.config["features"].get("demand_edge", {})
        deadline = int(self.config["requests"].get("deadline_steps", 960.0)) or 1
        return wait_bucket_edges(deadline, int(cfg.get("wait_bucket_count", 10)))

    @property
    def last_relay_importance(self) -> dict[str, float]:
        """Relay importance cached during the most recent graph build.

        Timing note (why the reward and the model share this cache): the cache
        is refreshed only in ``GraphBuilder.build``, which runs at the END of
        ``QKDEnv.step`` (after serve/expire, t += 1). The reward for the NEXT
        step reads this cache BEFORE ``_build_observation`` refreshes it, so
        the importance used by ``dense_reward`` reflects the state the policy
        actually observed when it made the decision (pre-service snapshot),
        not the post-service state. Do not recompute it mid-step or the dense
        reward would no longer match the model input.
        """
        return self._last_relay_importance

    def _relay_importance(
        self,
        requests: RequestQueue,
        demand_stats: dict[tuple[str, str], "DemandEdgeStats"] | None = None,
        env_state: EnvState | None = None,
        active_edges: list[Edge] | None = None,
    ) -> dict[str, float]:
        """Bidirectional shortest-hop demand importance for physical edges.

        Each demand edge's remaining volume is bucketed by wait time and
        decayed with ``exp(-wait / tau)``. A physical edge is a relay candidate
        when both GS endpoints can reach it within ``max_path_links`` hops; its
        weight uses the shortest total path through it and is decayed by
        ``hop_decay_factor`` for every extra hop. Scarcity keeps full links
        from receiving pressure. The accumulated value is normalized by the
        current-step maximum so it stays in [0, 1] regardless of how much
        historical demand is queued.
        """
        edge_cfg = self.config["features"]["edge"]
        if not edge_cfg.get("include_relay_importance", False):
            return {}
        if active_edges is not None:
            return self._relay_importance_on_active_graph(active_edges, demand_stats)
        if not self._relay_edge_info:
            return {}
        relay_cfg = edge_cfg.get("relay_importance", {})
        hop_decay = max(0.0, float(relay_cfg.get("hop_decay_factor", 0.5)))
        capacity_strength = max(0.0, float(relay_cfg.get("capacity_decay_strength", 1.0)))
        min_scarcity = max(0.0, float(relay_cfg.get("min_scarcity", 0.0)))
        demand_cfg = self.config["features"].get("demand_edge", {})
        tau = max(0.0, float(demand_cfg.get("wait_decay_tau", 0.0)))
        bucket_edges = self._wait_bucket_edges()
        totals: dict[str, float] = {}
        for pair, stats in (demand_stats or {}).items():
            entries = self._relay_edge_info.get(pair)
            if not entries:
                continue
            budget = 0.0
            for idx, amount in enumerate(stats.wait_bucket_amounts):
                if amount <= 0.0:
                    continue
                if tau <= 0.0:
                    budget += amount
                else:
                    age_center = (bucket_edges[idx] + bucket_edges[idx + 1]) / 2.0
                    budget += amount * math.exp(-age_center / tau)
            if budget <= 1.0e-9:
                continue
            for edge_id, total_hops in entries:
                capacity = self.qkp.get_capacity(edge_id)
                if capacity <= 0.0:
                    continue
                scarcity = max(0.0, (capacity - self.qkp.get_level(edge_id)) / capacity)
                if capacity_strength != 1.0:
                    scarcity = scarcity ** capacity_strength
                scarcity += min_scarcity
                if scarcity <= 0.0:
                    continue
                decay = hop_decay ** max(0, total_hops - 2)
                totals[edge_id] = totals.get(edge_id, 0.0) + budget * decay * scarcity
        if not totals:
            return {}
        max_value = max(totals.values())
        if max_value <= 0.0:
            return {}
        return {edge_id: value / max_value for edge_id, value in totals.items()}

    def _relay_importance_on_active_graph(
        self,
        active_edges: list[Edge],
        demand_stats: dict[tuple[str, str], "DemandEdgeStats"] | None,
    ) -> dict[str, float]:
        """Relay importance computed on the current visible subgraph.

        Hop distances are recomputed per step from the active (mask-passing)
        edges, so unavailable links never shorten a relay path and the
        importance does not rely on edges that are not in the current graph.
        """
        if not active_edges or not demand_stats:
            return {}
        adj: dict[str, list[str]] = {node_id: [] for node_id in self._node_ids_list}
        for edge in active_edges:
            adj[edge.src].append(edge.dst)
            adj[edge.dst].append(edge.src)

        gs_dist: dict[str, dict[str, int]] = {}
        for gs in self.gs_ids:
            dist = {gs: 0}
            queue = deque([gs])
            while queue:
                node = queue.popleft()
                for nxt in adj.get(node, ()):
                    if nxt in dist:
                        continue
                    dist[nxt] = dist[node] + 1
                    queue.append(nxt)
            gs_dist[gs] = dist

        edge_cfg = self.config["features"]["edge"]
        relay_cfg = edge_cfg.get("relay_importance", {})
        max_links = int(relay_cfg.get("max_path_links", 4))
        hop_decay = max(0.0, float(relay_cfg.get("hop_decay_factor", 0.5)))
        capacity_strength = max(0.0, float(relay_cfg.get("capacity_decay_strength", 1.0)))
        min_scarcity = max(0.0, float(relay_cfg.get("min_scarcity", 0.0)))
        demand_cfg = self.config["features"].get("demand_edge", {})
        tau = max(0.0, float(demand_cfg.get("wait_decay_tau", 0.0)))
        bucket_edges = self._wait_bucket_edges()
        inf = self.routing.DISCONNECTED
        totals: dict[str, float] = {}

        for pair, stats in (demand_stats or {}).items():
            src_dist = gs_dist.get(pair[0])
            dst_dist = gs_dist.get(pair[1])
            if src_dist is None or dst_dist is None:
                continue
            budget = 0.0
            for idx, amount in enumerate(stats.wait_bucket_amounts):
                if amount <= 0.0:
                    continue
                if tau <= 0.0:
                    budget += amount
                else:
                    age_center = (bucket_edges[idx] + bucket_edges[idx + 1]) / 2.0
                    budget += amount * math.exp(-age_center / tau)
            if budget <= 1.0e-9:
                continue
            for edge in active_edges:
                a = min(
                    src_dist.get(edge.src, inf),
                    src_dist.get(edge.dst, inf),
                )
                b = min(
                    dst_dist.get(edge.src, inf),
                    dst_dist.get(edge.dst, inf),
                )
                if a >= inf or b >= inf:
                    continue
                total_hops = a + b + 1
                if total_hops > max_links:
                    continue
                capacity = self.qkp.get_capacity(edge.edge_id)
                if capacity <= 0.0:
                    continue
                scarcity = max(0.0, (capacity - self.qkp.get_level(edge.edge_id)) / capacity)
                if capacity_strength != 1.0:
                    scarcity = scarcity ** capacity_strength
                scarcity += min_scarcity
                if scarcity <= 0.0:
                    continue
                decay = hop_decay ** max(0, total_hops - 2)
                totals[edge.edge_id] = totals.get(edge.edge_id, 0.0) + budget * decay * scarcity

        if not totals:
            return {}
        max_value = max(totals.values())
        if max_value <= 0.0:
            return {}
        return {edge_id: value / max_value for edge_id, value in totals.items()}

    def build_demand_edge_features(
        self,
        env_state: EnvState,
        requests: RequestQueue,
        request_history: RequestHistoryTracker,
        demand_pairs: list[tuple[str, str]],
        demand_stats_by_pair: dict[tuple[str, str], "DemandEdgeStats"] | None = None,
    ) -> np.ndarray:
        cfg = self.config["features"].get("demand_edge", {})
        demand_dim = int(self.config["features"]["dims"]["demand_edge_dim_resolved"])
        if not cfg.get("enabled", False):
            return np.zeros((0, demand_dim), dtype=np.float32)
        amount_scale = float(cfg.get("normalize_amount_by", 1000.0)) or 1.0
        log1p_amount = bool(cfg.get("normalize_amount_log1p", False))
        amount_log_denom = math.log1p(float(cfg.get("normalize_amount_reference", amount_scale)))
        count_scale = float(cfg.get("normalize_count_by", 10.0)) or 1.0
        log1p_count = bool(cfg.get("normalize_count_log1p", False))
        count_log_denom = math.log1p(float(cfg.get("normalize_count_reference", count_scale)))
        deadline_scale = float(cfg.get("normalize_deadline_by", 0.0) or 0.0)
        if deadline_scale <= 0.0:
            deadline_scale = float(self.config["requests"].get("deadline_steps", 960.0)) or 1.0
        if demand_stats_by_pair is None:
            stats_by_pair = requests.stats_by_pair(
                env_state.t,
                request_history,
                None,
                wait_bucket_count=int(cfg.get("wait_bucket_count", 10)),
                deadline_steps=int(self.config["requests"].get("deadline_steps", 960.0)),
            )
        else:
            stats_by_pair = demand_stats_by_pair
        rows = np.zeros((len(demand_pairs), demand_dim), dtype=np.float32)
        for r, pair in enumerate(demand_pairs):
            stats = stats_by_pair.get(pair) or empty_demand_stats(
                int(cfg.get("wait_bucket_count", 10))
            )
            col = 0
            if cfg.get("include_pending_amount", False):
                rows[r, col] = (
                    math.log1p(max(stats.pending_amount, 0.0)) / amount_log_denom
                    if log1p_amount
                    else stats.pending_amount / amount_scale
                )
                col += 1
            if cfg.get("include_pending_count", False):
                rows[r, col] = (
                    math.log1p(max(stats.pending_count, 0.0)) / count_log_denom
                    if log1p_count
                    else stats.pending_count / count_scale
                )
                col += 1
            if cfg.get("include_min_deadline_left", False):
                rows[r, col] = stats.min_deadline_left / deadline_scale
                col += 1
            if cfg.get("include_mean_deadline_left", False):
                rows[r, col] = stats.mean_deadline_left / deadline_scale
                col += 1
            if cfg.get("include_mean_wait_time", False):
                rows[r, col] = stats.mean_wait_time / deadline_scale
                col += 1
            if cfg.get("include_priority_sum", False):
                rows[r, col] = (
                    math.log1p(max(stats.priority_sum, 0.0)) / amount_log_denom
                    if log1p_amount
                    else stats.priority_sum / amount_scale
                )
                col += 1
            if cfg.get("include_wait_bucket_amounts", False):
                for amount in stats.wait_bucket_amounts:
                    rows[r, col] = (
                        math.log1p(max(amount, 0.0)) / amount_log_denom
                        if log1p_amount
                        else amount / amount_scale
                    )
                    col += 1
            if col != demand_dim:
                raise ValueError(f"Demand edge feature dim mismatch: {col} != {demand_dim}")
        return rows

    def _position_features(self, node: Node) -> list[float]:
        dim = int(self.config["features"]["node"].get("position_dim", 3))
        if node.lat is None or node.lon is None:
            return [0.0] * dim
        return [node.lat / 90.0, node.lon / 180.0, (node.alt_m or 0.0) / 10000.0][:dim]


def one_hot_node_type(node_type: NodeType) -> list[float]:
    return [float(node_type == item) for item in (NodeType.GS, NodeType.HAP, NodeType.SAT)]


def one_hot_link_type(link_type: LinkType) -> list[float]:
    return [
        float(link_type == item)
        for item in (LinkType.GS_HAP, LinkType.GS_SAT, LinkType.HAP_SAT, LinkType.SAT_SAT)
    ]


def all_gs_pairs(gs_ids: list[str]) -> list[tuple[str, str]]:
    return [(src, dst) for idx, src in enumerate(sorted(gs_ids)) for dst in sorted(gs_ids)[idx + 1 :]]


def demand_edge_id(pair: tuple[str, str]) -> str:
    src, dst = pair
    return f"D_{src}__{dst}"


def empty_demand_stats(bucket_count: int) -> DemandEdgeStats:
    return DemandEdgeStats(
        pending_amount=0.0,
        pending_count=0,
        min_deadline_left=0,
        mean_deadline_left=0.0,
        mean_wait_time=0.0,
        priority_sum=0.0,
        wait_bucket_amounts=[0.0] * max(0, int(bucket_count)),
    )


