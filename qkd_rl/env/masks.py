"""Action mask builder: defines which neighbor links a node may legally select.

The mask is the single source of legality for candidate actions. It is used in
two places: ``GraphBuilder`` filters ``action_candidates`` to legal actions at
graph-build time, and ``ActionResolver`` re-validates submitted actions against
a fresh mask as defense in depth.

The per-step build is vectorized: candidate edge positions are precomputed per
node, and each step gathers the dynamic ``available`` / ``rate`` arrays once
instead of calling a Python legality function per (node, action) pair.
"""

from __future__ import annotations

import numpy as np

from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.qkp import LinkQKPPool
from qkd_rl.env.request import RequestQueue
from qkd_rl.env.state import EnvState


class ActionMaskBuilder:
    def __init__(
        self,
        action_space: NodeActionSpace,
        config: dict,
        min_link_rate: float = 0.0,
        allowed_link_types: list[str] | None = None,
    ):
        self.action_space = action_space
        self.config = config
        self.min_link_rate = float(min_link_rate)
        self.allowed_link_types = set(allowed_link_types or [])
        self._mask_available = bool(config.get("mask_unavailable_edges", True))
        self._mask_rate = bool(config.get("mask_below_min_rate", True))
        self._mask_types = bool(config.get("mask_disallowed_link_types", False)) and bool(
            self.allowed_link_types
        )
        self._mask_full = bool(config.get("mask_full_qkp_edges", False))
        # Precompute candidate lists and their edge ids once per env build.
        self.candidates: dict[str, list[str]] = {
            node_id: self.action_space.candidates_for_node(node_id) for node_id in self.action_space.node_ids
        }
        self.edge_by_action: dict[tuple[str, str], str | None] = {
            (node_id, action): self.action_space.action_to_edge(node_id, action)
            for node_id, cands in self.candidates.items()
            for action in cands
        }
        # Referenced edge ids -> column index in the per-step gather arrays.
        referenced: set[str] = set()
        for node_id in self.action_space.node_ids:
            for action in self.candidates[node_id]:
                if action != NodeActionSpace.IDLE:
                    edge_id = self.edge_by_action.get((node_id, action))
                    if edge_id is not None:
                        referenced.add(edge_id)
        self._ref_edges: list[str] = sorted(referenced)
        self._edge_pos: dict[str, int] = {edge_id: i for i, edge_id in enumerate(self._ref_edges)}
        self._node_pos: dict[str, np.ndarray] = {
            node_id: np.array(
                [
                    -1
                    if action == NodeActionSpace.IDLE
                    else self._edge_pos.get(self.edge_by_action.get((node_id, action)), -1)
                    for action in self.candidates[node_id]
                ],
                dtype=np.int64,
            )
            for node_id in self.action_space.node_ids
        }
        self._edge_type: dict[str, str] = {}
        # All candidate positions across nodes in one flat array so build() is
        # a few vectorized ops instead of per-node numpy work (8625 candidates).
        self._flat_pos = np.concatenate([self._node_pos[nid] for nid in self.action_space.node_ids])
        self._splits = np.cumsum([len(self._node_pos[nid]) for nid in self.action_space.node_ids])
        # Link-id gather indices are static for the whole run (they depend only
        # on the registry mapping, not on t); cache them so build() avoids a
        # per-step np.fromiter over the referenced edges.
        self._ref_link_ids: np.ndarray | None = None
        # The flat legal array from the most recent build(), aligned with the
        # per-node candidate order. GraphBuilder and the actor plan reuse it so
        # the same 8625-element flat mask is not rebuilt three times per step.
        self._last_flat_legal: np.ndarray | None = None

    @property
    def last_flat_legal(self) -> np.ndarray | None:
        return self._last_flat_legal

    def build(self, env_state: EnvState, qkp: LinkQKPPool, requests: RequestQueue) -> dict[str, list[bool]]:
        n = len(self._ref_edges)
        if n == 0:
            legal = np.ones(self._flat_pos.shape[0], dtype=bool)
            self._last_flat_legal = legal
            out = {}
            start = 0
            for i, node_id in enumerate(self.action_space.node_ids):
                end = int(self._splits[i])
                out[node_id] = legal[start:end].tolist()
                start = end
            return out
        windows = env_state.edge_windows
        edge_pos = self._edge_pos
        available = np.empty(n, dtype=bool)
        rates = np.empty(n, dtype=float)
        if hasattr(windows, "edge_columns"):
            # Lazy window store: one vectorized column gather instead of a
            # per-edge EdgeWindow materialization loop over all 4241 links.
            if self._ref_link_ids is None:
                self._ref_link_ids = windows.link_ids(self._ref_edges)
            idx = self._ref_link_ids
            blocks = windows.blocks
            avail_col, rate_col = blocks[1][0][idx], blocks[0][0][idx]
            available[:] = avail_col
            rates[:] = rate_col
        else:
            for edge_id in self._ref_edges:
                window = windows[edge_id]
                i = edge_pos[edge_id]
                available[i] = window.available[0]
                rates[i] = window.rates[0]
        type_ok: np.ndarray | None = None
        if self._mask_types:
            type_ok = np.zeros(n, dtype=bool)
            for edge_id in self._ref_edges:
                link_type = self._edge_type.get(edge_id)
                if link_type is None:
                    link_type = windows[edge_id].link_type.value
                    self._edge_type[edge_id] = link_type
                type_ok[self._edge_pos[edge_id]] = link_type in self.allowed_link_types
        cap_left: np.ndarray | None = None
        if self._mask_full:
            cap_left = np.array([qkp.get_capacity_left(edge_id) > 0 for edge_id in self._ref_edges], dtype=bool)

        # -1 encodes the idle action, which is always legal; only real
        # candidate edges are subject to the availability/rate/type/capacity
        # conditions.
        legal = np.ones(self._flat_pos.shape[0], dtype=bool)
        non_idle = self._flat_pos >= 0
        p = np.maximum(self._flat_pos, 0)
        if self._mask_available:
            legal[non_idle] = available[p[non_idle]]
        if self._mask_rate:
            legal[non_idle] &= rates[p[non_idle]] >= self.min_link_rate
        if type_ok is not None:
            legal[non_idle] &= type_ok[p[non_idle]]
        if cap_left is not None:
            legal[non_idle] &= cap_left[p[non_idle]]
        out: dict[str, list[bool]] = {}
        start = 0
        for i, node_id in enumerate(self.action_space.node_ids):
            end = int(self._splits[i])
            out[node_id] = legal[start:end].tolist()
            start = end
        self._last_flat_legal = legal
        return out

    def is_action_legal(
        self,
        node_id: str,
        action: str,
        env_state: EnvState,
        qkp: LinkQKPPool,
        requests: RequestQueue,
    ) -> bool:
        if action == NodeActionSpace.IDLE:
            return True
        edge_id = self.edge_by_action.get((node_id, action))
        if edge_id is None:
            return False
        window = env_state.edge_windows[edge_id]
        if self._mask_available and not window.available[0]:
            return False
        if self._mask_rate and window.rates[0] < self.min_link_rate:
            return False
        if self._mask_types and window.link_type.value not in self.allowed_link_types:
            return False
        if self._mask_full and qkp.get_capacity_left(edge_id) <= 0:
            return False
        return True
