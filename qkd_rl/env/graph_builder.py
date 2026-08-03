from __future__ import annotations

import math
from dataclasses import dataclass

from qkd_rl.core.types import Edge, LinkType, Node, NodeType
from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.qkp import LinkQKPPool
from qkd_rl.env.request import DemandEdgeStats, RequestHistoryTracker, RequestQueue
from qkd_rl.env.state import EnvState
from qkd_rl.link.rate_provider import RateNormalizer


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


class GraphBuilder:
    def __init__(
        self,
        nodes: list[Node],
        edges: list[Edge],
        action_space: NodeActionSpace,
        qkp: LinkQKPPool,
        normalizer: RateNormalizer,
        config: dict,
    ):
        self.nodes = nodes
        self.edges = edges
        self.action_space = action_space
        self.qkp = qkp
        self.normalizer = normalizer
        self.config = config
        self.node_index = {node.node_id: idx for idx, node in enumerate(nodes)}
        self.gs_ids = [node.node_id for node in nodes if node.node_type == NodeType.GS]

    def build(
        self,
        env_state: EnvState,
        requests: RequestQueue,
        request_history: RequestHistoryTracker,
        masks: dict[str, list[bool]],
    ) -> GraphObservation:
        demand_pairs = self.build_demand_pairs(env_state, requests, request_history)
        demand_edge_ids = [demand_edge_id(pair) for pair in demand_pairs]
        action_candidates: dict[str, list[str]] = {}
        action_masks: dict[str, list[bool]] = {}
        for node in self.nodes:
            node_id = node.node_id
            full_candidates = self.action_space.candidates_for_node(node_id)
            legal = [action for action, ok in zip(full_candidates, masks[node_id]) if ok]
            action_candidates[node_id] = legal
            action_masks[node_id] = [True] * len(legal)
        return GraphObservation(
            node_features=self.build_node_features(env_state, requests, request_history),
            edge_index=self.build_edge_index(demand_pairs),
            edge_features=self.build_edge_features(env_state, requests, request_history, demand_pairs),
            node_ids=[node.node_id for node in self.nodes],
            edge_ids=[edge.edge_id for edge in self.edges] + demand_edge_ids,
            physical_edge_ids=[edge.edge_id for edge in self.edges],
            demand_edge_ids=demand_edge_ids,
            action_candidates=action_candidates,
            action_masks=action_masks,
            state=env_state,
        )

    def build_node_features(
        self,
        env_state: EnvState,
        requests: RequestQueue,
        request_history: RequestHistoryTracker,
    ) -> list[list[float]]:
        """Build node features following ``features.node`` switches.

        The feature order matches ``ConfigValidator.resolve_feature_dims`` so
        resolved dimensions stay consistent when switches are turned off.
        """
        node_cfg = self.config["features"]["node"]
        node_dim = int(self.config["features"]["dims"]["node_dim_resolved"])
        windows = [int(value) for value in node_cfg.get("recent_demand_windows", [])]
        amount_scale = 1000.0
        count_scale = 10.0
        demand_in = requests.demand_in_by_node()
        demand_out = requests.demand_out_by_node()
        pending_count = requests.pending_count_by_node()

        features: list[list[float]] = []
        for node in self.nodes:
            row: list[float] = []
            if node_cfg.get("include_node_type_one_hot", True):
                row.extend(one_hot_node_type(node.node_type))
            incident_edges = [edge for edge in self.edges if node.node_id in (edge.src, edge.dst)]
            total_level = sum(self.qkp.get_level(edge.edge_id) for edge in incident_edges)
            total_capacity = sum(self.qkp.get_capacity(edge.edge_id) for edge in incident_edges) or 1.0
            if node_cfg.get("include_qkp_level", True):
                row.append(total_level / total_capacity)
            if node_cfg.get("include_qkp_capacity_left", True):
                row.append((total_capacity - total_level) / total_capacity)
            if node_cfg.get("include_qkp_utilization", True):
                row.append(total_level / total_capacity)
            node_id = node.node_id
            if node_cfg.get("include_demand_in", True):
                row.append(demand_in.get(node_id, 0.0) / amount_scale)
            if node_cfg.get("include_demand_out", True):
                row.append(demand_out.get(node_id, 0.0) / amount_scale)
            if node_cfg.get("include_queue_pressure", True):
                row.append(pending_count.get(node_id, 0) / count_scale)
            if node_cfg.get("include_is_available", True):
                row.append(
                    float(
                        any(env_state.edge_windows[edge.edge_id].available[0] for edge in incident_edges)
                    )
                )
            if node_cfg.get("include_recent_demand", False):
                for window in windows:
                    row.append(request_history.sum_for_node(node_id, "arrived", env_state.t, window) / amount_scale)
            if node_cfg.get("include_time_features", False):
                time_cfg = node_cfg.get("time_features", {})
                minute = env_state.t % 1440
                day = env_state.t % 525600
                if time_cfg.get("minute_of_day_sin_cos", False):
                    row.extend([math.sin(2 * math.pi * minute / 1440), math.cos(2 * math.pi * minute / 1440)])
                if time_cfg.get("day_of_year_sin_cos", False):
                    row.extend([math.sin(2 * math.pi * day / 525600), math.cos(2 * math.pi * day / 525600)])
            if node_cfg.get("include_position", False):
                row.extend(self._position_features(node))
            if len(row) != node_dim:
                raise ValueError(f"Node feature dim mismatch for {node_id}: {len(row)} != {node_dim}")
            features.append(row)
        return features

    def build_edge_index(self, demand_pairs: list[tuple[str, str]]) -> list[tuple[int, int]]:
        index: list[tuple[int, int]] = []
        for edge in self.edges:
            src = self.node_index[edge.src]
            dst = self.node_index[edge.dst]
            index.append((src, dst))
            index.append((dst, src))
        for src_id, dst_id in demand_pairs:
            src = self.node_index[src_id]
            dst = self.node_index[dst_id]
            index.append((src, dst))
            index.append((dst, src))
        return index

    def build_edge_features(
        self,
        env_state: EnvState,
        requests: RequestQueue,
        request_history: RequestHistoryTracker,
        demand_pairs: list[tuple[str, str]],
    ) -> list[list[float]]:
        """Build physical + demand edge features following ``features.edge``
        and ``features.demand_edge`` switches."""
        edge_cfg = self.config["features"]["edge"]
        physical_dim = int(self.config["features"]["dims"]["physical_edge_dim_resolved"])
        demand_dim = int(self.config["features"]["dims"]["demand_edge_dim_resolved"])
        horizon = int(edge_cfg.get("prediction_horizon", 0))
        rows: list[list[float]] = []
        for edge in self.edges:
            window = env_state.edge_windows[edge.edge_id]
            rates_norm = [self.normalizer.transform_scalar(rate) for rate in window.rates]
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
            if edge_cfg.get("include_switch_cost", False):
                # Only the previous activation flag is available; use it so the
                # dimension stays consistent with the config when enabled.
                row.append(float(edge.edge_id in env_state.last_activated_edges))
            row.extend([0.0] * demand_dim)
            if len(row) != physical_dim + demand_dim:
                raise ValueError(f"Physical edge feature dim mismatch: {len(row)} != {physical_dim + demand_dim}")
            rows.append(row)
        for row in self.build_demand_edge_features(env_state, requests, request_history, demand_pairs):
            rows.append(([0.0] * physical_dim) + row)
        return rows

    def build_demand_pairs(
        self,
        env_state: EnvState,
        requests: RequestQueue,
        request_history: RequestHistoryTracker,
    ) -> list[tuple[str, str]]:
        cfg = self.config["features"].get("demand_edge", {})
        if not cfg.get("enabled", False):
            return []
        windows = [int(value) for value in cfg.get("history_windows", [])]
        mode = cfg.get("build_mode", "active_or_recent_pairs")
        if mode == "all_gs_pairs":
            return all_gs_pairs(self.gs_ids)
        if mode == "active_pairs":
            return sorted(requests.demand_by_pair())
        if mode == "active_or_recent_pairs":
            stats = requests.stats_by_pair(env_state.t, request_history, windows)
            return sorted(stats)
        raise ValueError(f"Unknown demand edge build mode: {mode}")

    def build_demand_edge_features(
        self,
        env_state: EnvState,
        requests: RequestQueue,
        request_history: RequestHistoryTracker,
        demand_pairs: list[tuple[str, str]],
    ) -> list[list[float]]:
        cfg = self.config["features"].get("demand_edge", {})
        demand_dim = int(self.config["features"]["dims"]["demand_edge_dim_resolved"])
        if not cfg.get("enabled", False):
            return []
        windows = [int(value) for value in cfg.get("history_windows", [])]
        amount_scale = float(cfg.get("normalize_amount_by", 1000.0)) or 1.0
        count_scale = float(cfg.get("normalize_count_by", 10.0)) or 1.0
        deadline_scale = float(cfg.get("normalize_deadline_by", 60.0)) or 1.0
        stats_by_pair = requests.stats_by_pair(env_state.t, request_history, windows)
        rows: list[list[float]] = []
        for pair in demand_pairs:
            stats = stats_by_pair.get(pair) or empty_demand_stats(windows)
            row: list[float] = []
            if cfg.get("include_pending_amount", False):
                row.append(stats.pending_amount / amount_scale)
            if cfg.get("include_pending_count", False):
                row.append(stats.pending_count / count_scale)
            if cfg.get("include_min_deadline_left", False):
                row.append(stats.min_deadline_left / deadline_scale)
            if cfg.get("include_mean_deadline_left", False):
                row.append(stats.mean_deadline_left / deadline_scale)
            if cfg.get("include_priority_sum", False):
                row.append(stats.priority_sum / amount_scale)
            if cfg.get("include_arrival_history", False):
                row.extend(stats.arrival_history[window] / amount_scale for window in windows)
            if cfg.get("include_served_history", False):
                row.extend(stats.served_history[window] / amount_scale for window in windows)
            if cfg.get("include_failed_history", False):
                row.extend(stats.failed_history[window] / amount_scale for window in windows)
            if len(row) != demand_dim:
                raise ValueError(f"Demand edge feature dim mismatch: {len(row)} != {demand_dim}")
            rows.append(row)
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


def empty_demand_stats(windows: list[int]) -> DemandEdgeStats:
    return DemandEdgeStats(
        pending_amount=0.0,
        pending_count=0,
        min_deadline_left=0,
        mean_deadline_left=0.0,
        priority_sum=0.0,
        arrival_history={window: 0.0 for window in windows},
        served_history={window: 0.0 for window in windows},
        failed_history={window: 0.0 for window in windows},
    )
