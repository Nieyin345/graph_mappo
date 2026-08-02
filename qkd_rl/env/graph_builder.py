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
        return GraphObservation(
            node_features=self.build_node_features(env_state, requests),
            edge_index=self.build_edge_index(demand_pairs),
            edge_features=self.build_edge_features(env_state, requests, request_history, demand_pairs),
            node_ids=[node.node_id for node in self.nodes],
            edge_ids=[edge.edge_id for edge in self.edges] + demand_edge_ids,
            physical_edge_ids=[edge.edge_id for edge in self.edges],
            demand_edge_ids=demand_edge_ids,
            action_candidates={
                node.node_id: self.action_space.candidates_for_node(node.node_id) for node in self.nodes
            },
            action_masks=masks,
            state=env_state,
        )

    def build_node_features(self, env_state: EnvState, requests: RequestQueue) -> list[list[float]]:
        demand = requests.demand_by_node()
        features: list[list[float]] = []
        for node in self.nodes:
            row: list[float] = []
            row.extend(one_hot_node_type(node.node_type))
            incident_edges = [edge for edge in self.edges if node.node_id in (edge.src, edge.dst)]
            total_level = sum(self.qkp.get_level(edge.edge_id) for edge in incident_edges)
            total_capacity = sum(self.qkp.get_capacity(edge.edge_id) for edge in incident_edges) or 1.0
            row.extend(
                [
                    total_level / total_capacity,
                    (total_capacity - total_level) / total_capacity,
                    total_level / total_capacity,
                    demand.get(node.node_id, 0.0) / 1000.0,
                    demand.get(node.node_id, 0.0) / 1000.0,
                    demand.get(node.node_id, 0.0) / 1000.0,
                ]
            )
            row.extend([demand.get(node.node_id, 0.0) / 1000.0] * 3)
            row.append(1.0)
            minute = env_state.t % 1440
            day = env_state.t % 525600
            row.extend([math.sin(2 * math.pi * minute / 1440), math.cos(2 * math.pi * minute / 1440)])
            row.extend([math.sin(2 * math.pi * day / 525600), math.cos(2 * math.pi * day / 525600)])
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
        rows: list[list[float]] = []
        physical_dim = int(self.config["features"]["dims"]["physical_edge_dim_resolved"])
        demand_dim = int(self.config["features"]["dims"]["demand_edge_dim_resolved"])
        for edge in self.edges:
            window = env_state.edge_windows[edge.edge_id]
            rates_norm = [self.normalizer.transform_scalar(rate) for rate in window.rates]
            row: list[float] = []
            row.extend(one_hot_link_type(edge.link_type))
            row.append(float(window.available[0]))
            row.append(rates_norm[0])
            row.extend(rates_norm[1:])
            row.extend(float(value) for value in window.available[1:])
            row.append(rates_norm[-1] - rates_norm[0] if len(rates_norm) > 1 else 0.0)
            row.append(sum(rates_norm) / len(rates_norm))
            row.append(max(rates_norm))
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
