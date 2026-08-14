"""Per-entity rolling history windows feeding the shared history encoder.

For every node, every physical link, and every active GS-GS demand pair the
buffer keeps a fixed-length sliding window of per-step snapshots. The windows
are assembled by ``GraphBuilder`` into ``GraphObservation`` and encoded by
``HistoryEncoder`` (shared LSTM); the per-entity embeddings are concatenated to
the matching node / physical-edge / demand-edge features before the GNN.

All channel switches, ``seq_len`` and normalization live in
``configs/features.yaml -> history_encoder`` (see the design section of the
condensed architecture doc).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict, deque

from qkd_rl.core.types import Edge, Node
from qkd_rl.env.qkp import LinkQKPPool
from qkd_rl.env.request import ServeResult, wait_bucket_edges


def _pair_key(req) -> tuple[str, str]:
    return tuple(sorted((req.src_gs, req.dst_gs)))


class HistoryBuffer:
    """Sliding per-entity windows. Disabled entirely when ``enabled: false``."""

    def __init__(self, nodes: list[Node], edges: list[Edge], config: dict):
        he = config["features"]["history_encoder"]
        self.seq_len = int(he.get("seq_len", 240))
        self.amount_scale = float(he.get("normalize_amount_by", 1000.0)) or 1.0
        self.log1p_amount = bool(he.get("normalize_amount_log1p", False))
        self.amount_log_denom = math.log1p(float(he.get("normalize_amount_reference", self.amount_scale)))

        node_cfg = he.get("node", {})
        phys_cfg = he.get("physical_edge", {})
        demand_cfg = he.get("demand_edge", {})
        self.node_channels = [
            ("arrived", bool(node_cfg.get("include_arrived", True))),
            ("served", bool(node_cfg.get("include_served", True))),
            ("failed", bool(node_cfg.get("include_failed", True))),
            ("qkp_total", bool(node_cfg.get("include_qkp_total", True))),
        ]
        self.phys_channels = [
            ("qkp_level", bool(phys_cfg.get("include_qkp_level", True))),
            ("available", bool(phys_cfg.get("include_available", True))),
            ("activated", bool(phys_cfg.get("include_activated", False))),
        ]
        self.demand_wait_buckets = bool(demand_cfg.get("include_pending_wait_buckets", False))
        self.demand_channels: list[tuple[str, bool]] = (
            [("pending_wait_buckets", True)] if self.demand_wait_buckets else []
        )

        self.node_ch = sum(1 for _, enabled in self.node_channels if enabled)
        self.phys_ch = sum(1 for _, enabled in self.phys_channels if enabled)
        self.demand_ch = (
            int(config["features"]["demand_edge"].get("wait_bucket_count", 10))
            if self.demand_wait_buckets
            else 0
        )
        self.deadline_steps = int(config["requests"].get("deadline_steps", 960)) or 1
        self.bucket_edges = (
            wait_bucket_edges(self.deadline_steps, self.demand_ch)
            if self.demand_ch > 0
            else []
        )

        self.node_ids = [node.node_id for node in nodes]
        self.edges = edges
        self.incident_by_node: dict[str, list[Edge]] = {
            node_id: [edge for edge in edges if node_id in (edge.src, edge.dst)] for node_id in self.node_ids
        }
        self.node_seqs: dict[str, deque] = {node_id: deque(maxlen=self.seq_len) for node_id in self.node_ids}
        self.phys_seqs: dict[str, deque] = {edge.edge_id: deque(maxlen=self.seq_len) for edge in edges}
        self.demand_seqs: dict[tuple[str, str], deque] = {}
        self._pair_last_active: dict[tuple[str, str], int] = {}

    def reset(self) -> None:
        for dq in self.node_seqs.values():
            dq.clear()
        for dq in self.phys_seqs.values():
            dq.clear()
        self.demand_seqs.clear()
        self._pair_last_active.clear()

    # ------------------------------------------------------------------ push
    def push(
        self,
        *,
        arrivals: list,
        serve_result: ServeResult,
        expired_requests: list,
        qkp: LinkQKPPool,
        edge_windows: dict,
        activated_edges: list[str],
        pending_requests: list,
        t: int,
    ) -> None:
        if self.node_ch > 0:
            self._push_nodes(arrivals, serve_result, expired_requests, qkp)
        if self.phys_ch > 0:
            self._push_physical(edge_windows, activated_edges, qkp)
        if self.demand_ch > 0:
            self._push_demand(pending_requests, t)

    def _push_nodes(
        self,
        arrivals: list,
        serve_result: ServeResult,
        expired_requests: list,
        qkp: LinkQKPPool,
    ) -> None:
        per_node = {node_id: [0.0, 0.0, 0.0] for node_id in self.node_ids}  # arrived/served/failed
        for req in arrivals:
            for node_id in (req.src_gs, req.dst_gs):
                if node_id in per_node:
                    per_node[node_id][0] += req.amount
        for req in serve_result.served_requests:
            for node_id in (req.src_gs, req.dst_gs):
                if node_id in per_node:
                    per_node[node_id][1] += req.amount
        for req in list(serve_result.failed_requests) + list(expired_requests):
            remaining = max(0.0, req.amount - req.served_amount)
            for node_id in (req.src_gs, req.dst_gs):
                if node_id in per_node:
                    per_node[node_id][2] += remaining

        want = {name: enabled for name, enabled in self.node_channels}
        for node_id in self.node_ids:
            row: list[float] = []
            amounts = per_node[node_id]
            if want["arrived"]:
                row.append(self._norm_amount(amounts[0]))
            if want["served"]:
                row.append(self._norm_amount(amounts[1]))
            if want["failed"]:
                row.append(self._norm_amount(amounts[2]))
            if want["qkp_total"]:
                row.append(self._qkp_total(node_id, qkp))
            self.node_seqs[node_id].append(row)

    def _push_physical(self, edge_windows: dict, activated_edges: list[str], qkp: LinkQKPPool) -> None:
        want = {name: enabled for name, enabled in self.phys_channels}
        activated_set = set(activated_edges)
        for edge in self.edges:
            edge_id = edge.edge_id
            row: list[float] = []
            if want["qkp_level"]:
                row.append(qkp.get_level(edge_id) / (qkp.get_capacity(edge_id) or 1.0))
            if want["available"]:
                row.append(float(edge_windows[edge_id].available[0]))
            if want["activated"]:
                row.append(float(edge_id in activated_set))
            self.phys_seqs[edge_id].append(row)

    def _push_demand(self, pending_requests: list, t: int) -> None:
        per_pair: dict[tuple[str, str], list[float]] = defaultdict(
            lambda: [0.0] * self.demand_ch
        )
        for req in pending_requests:
            remaining = max(0.0, req.amount - req.served_amount)
            if remaining <= 0.0:
                continue
            age = max(0, t - req.arrival_t)
            idx = min(self.demand_ch - 1, max(0, bisect_right(self.bucket_edges, age) - 1))
            per_pair[_pair_key(req)][idx] += remaining

        for pair in per_pair:
            self.demand_seqs.setdefault(pair, deque(maxlen=self.seq_len))
            self._pair_last_active[pair] = t
        for pair, dq in self.demand_seqs.items():
            vals = per_pair.get(pair)
            if vals is None:
                dq.append([0.0] * self.demand_ch)
            else:
                dq.append([self._norm_amount(value) for value in vals])
        self._prune_demand(t)

    def _prune_demand(self, t: int) -> None:
        stale = [pair for pair, last in self._pair_last_active.items() if t - last > self.seq_len * 2]
        for pair in stale:
            self.demand_seqs.pop(pair, None)
            self._pair_last_active.pop(pair, None)

    # ------------------------------------------------------------------ reads
    def node_sequences(self, nodes: list[Node]) -> tuple[list[list[list[float]]], list[int]]:
        return self._collect(self.node_seqs, [node.node_id for node in nodes], self.node_ch)

    def physical_sequences(self, edges: list[Edge]) -> tuple[list[list[list[float]]], list[int]]:
        return self._collect(self.phys_seqs, [edge.edge_id for edge in edges], self.phys_ch)

    def demand_sequences(self, pairs: list[tuple[str, str]]) -> tuple[list[list[list[float]]], list[int]]:
        return self._collect(self.demand_seqs, [pair for pair in pairs], self.demand_ch)

    def _collect(
        self,
        store: dict,
        keys: list,
        n_channels: int,
    ) -> tuple[list[list[list[float]]], list[int]]:
        if n_channels <= 0:
            return [], []
        # Return right-padded fixed-length windows ([entity][time][channel])
        # plus per-entity valid lengths, so the encoder always receives a
        # dense tensor even on cold start. The valid data leads and the zeros
        # trail, matching ``pack_padded_sequence`` semantics (valid prefix,
        # padding suffix): a left-padded layout would make the encoder pack
        # the zero prefix as "valid" and drop the real history instead.
        seqs, valids = [], []
        for key in keys:
            dq = store.get(key)
            rows = [list(row) for row in dq] if dq is not None else []
            valid = len(rows)
            if valid < self.seq_len:
                rows = rows + [[0.0] * n_channels] * (self.seq_len - valid)
            seqs.append(rows)
            valids.append(valid)
        return seqs, valids

    # ----------------------------------------------------------------- helpers
    def _qkp_total(self, node_id: str, qkp: LinkQKPPool) -> float:
        # Precomputed incident edges avoid an O(N x E) scan every step.
        total_level = 0.0
        total_capacity = 0.0
        for edge in self.incident_by_node[node_id]:
            total_level += qkp.get_level(edge.edge_id)
            total_capacity += qkp.get_capacity(edge.edge_id)
        return total_level / (total_capacity or 1.0)

    def _norm_amount(self, value: float) -> float:
        if self.log1p_amount:
            return math.log1p(max(value, 0.0)) / self.amount_log_denom
        return value / self.amount_scale
