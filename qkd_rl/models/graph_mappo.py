from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.graph_builder import GraphObservation
from qkd_rl.models.history_encoder import HistoryEncoder
from qkd_rl.models.mlp import build_mlp, masked_logits


@dataclass
class GraphTensors:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features_directed: torch.Tensor
    node_ids: list[str]
    edge_ids: list[str]
    num_physical_directed: int


@dataclass
class ActorCriticOutput:
    logits: dict[str, torch.Tensor]
    value: torch.Tensor
    # Already-padded per-node logits (one row per node, -inf beyond the node's
    # candidate count). Lets the policy sample one batched Categorical without
    # re-padding the per-node dict tensors.
    logits_padded: torch.Tensor | None = None
    logits_node_order: list[str] | None = None
    logits_lengths: list[int] | None = None
    # Raw edge-scorer output for every legal physical edge of this observation
    # (edge_id -> scalar). This is the model's global, cross-node-comparable
    # estimate of edge quality; the priority-matching resolver uses it instead
    # of per-node log probabilities, which are only comparable within a node.
    edge_scores: dict[str, torch.Tensor] | None = None


@dataclass
class BatchedActorCriticOutput:
    # Per-observation padded logits (one row per node of that graph), node
    # order, per-node candidate lengths, and critic value.
    padded_logits: list[torch.Tensor]
    node_orders: list[list[str]]
    lengths: list[list[int]]
    values: list[torch.Tensor]
    edge_score_maps: list[dict[str, torch.Tensor]]


def _edge_score_map(
    plan: tuple,
    node_index: dict[str, int],
    edge_by_pair: dict[tuple[str, str], str],
    edge_scores: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Map raw edge-scorer outputs back to edge ids for one observation.

    ``plan[0]`` / ``plan[1]`` are the source/destination node positions of the
    legal edge candidates in flat candidate order, which is exactly the order
    of ``edge_scores`` returned by the batched scorer. Both endpoints of a
    physical edge map to the same edge id and the same learned score, so the
    resolver can compare edges globally instead of comparing per-node log
    probabilities that are only identifiable up to a node-wise constant.
    """
    if node_index is None or edge_by_pair is None:
        return {}
    index_to_node = {idx: node_id for node_id, idx in node_index.items()}
    edge_map: dict[str, torch.Tensor] = {}
    srcs = plan[0]
    dsts = plan[1]
    for i in range(int(srcs.size)):
        src = index_to_node[int(srcs[i])]
        dst = index_to_node[int(dsts[i])]
        edge_id = edge_by_pair.get(tuple(sorted((src, dst))))
        if edge_id is not None:
            edge_map[edge_id] = edge_scores[i]
    return edge_map


def observation_to_tensors(
    obs: GraphObservation,
    device: torch.device | str = "cpu",
    edge_dim: int | None = None,
) -> GraphTensors:
    """Convert a :class:`GraphObservation` into batched tensors.

    Mask-first filtering can leave a graph with zero physical links (and zero
    demand edges), so the empty tensors are built with the correct 2-D shapes
    ``(2, 0)`` for ``edge_index`` and ``(0, edge_dim)`` for edge features
    instead of degenerate 1-D tensors.
    """
    # numpy -> torch.from_numpy -> .to(device) is much faster than
    # torch.tensor(list) for the per-step list-of-lists conversions. The graph
    # builder already returns float32/int64 arrays, so np.asarray is a no-op
    # and torch.from_numpy shares the memory (zero copy).
    if isinstance(obs.node_features, np.ndarray):
        node_features = torch.from_numpy(obs.node_features).to(device)
    else:
        node_features = torch.from_numpy(np.asarray(obs.node_features, dtype=np.float32)).to(device)
    if isinstance(obs.edge_index, np.ndarray):
        edge_index = torch.from_numpy(np.ascontiguousarray(obs.edge_index.T)).to(device)
    elif obs.edge_index:
        edge_index = torch.from_numpy(np.asarray(obs.edge_index, dtype=np.int64).T.copy()).to(device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    if isinstance(obs.edge_features, np.ndarray) and obs.edge_features.ndim == 2:
        edge_features = torch.from_numpy(obs.edge_features).to(device)
    elif obs.edge_features:
        edge_features = torch.from_numpy(np.asarray(obs.edge_features, dtype=np.float32)).to(device)
    else:
        if edge_dim is None:
            raise ValueError("edge_dim is required when the observation has no edge features.")
        edge_features = torch.zeros((0, edge_dim), dtype=torch.float32, device=device)
    edge_features_directed = edge_features.repeat_interleave(2, dim=0)
    return GraphTensors(
        node_features=node_features,
        edge_index=edge_index,
        edge_features_directed=edge_features_directed,
        node_ids=obs.node_ids,
        edge_ids=obs.edge_ids,
        num_physical_directed=2 * len(obs.physical_edge_ids),
    )


class EdgeConditionedGraphLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        activation: str,
        dropout: float,
        layer_norm: bool,
        fuse_physical_to_node: bool = True,
    ):
        super().__init__()
        # 物理链路边和逻辑请求边使用各自独立的 message MLP：物理边传递链路
        # 能力（速率/容量/可用性），逻辑边传递需求（待服务量/deadline/优先级），
        # 两者的语义标准不同，不能共用同一个变换。
        self.message_mlp_phys = build_mlp(
            input_dim=hidden_dim + edge_dim,
            hidden_dims=[hidden_dim],
            output_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
        )
        self.message_mlp_demand = build_mlp(
            input_dim=hidden_dim + edge_dim,
            hidden_dims=[hidden_dim],
            output_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
        )
        self.update = build_mlp(
            input_dim=hidden_dim * 2,
            hidden_dims=[hidden_dim],
            output_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.fuse_physical_to_node = fuse_physical_to_node

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_physical_directed: int,
    ) -> torch.Tensor:
        src, dst = edge_index
        aggregated = torch.zeros_like(node_emb)
        # 物理边位于张量前 num_physical_directed 行，逻辑请求边随后；两类边
        # 各自独立聚合并按各自的邻居数归一化再相加，保证两类信息互不稀释。
        for message_mlp, start, end in (
            (self.message_mlp_phys, 0, int(num_physical_directed)),
            (self.message_mlp_demand, int(num_physical_directed), edge_index.size(1)),
        ):
            if start >= end:
                continue
            if not self.fuse_physical_to_node and start == 0:
                continue
            src_s, dst_s = src[start:end], dst[start:end]
            messages = message_mlp(torch.cat([node_emb[src_s], edge_attr[start:end]], dim=-1))
            # mean = sum / count via index_add_: scatter_reduce_("mean") is
            # measurably slower on CUDA (~26% in micro-benchmark) and can be
            # numerically unstable for small graphs; the two-kernel version is
            # exactly sum / count with clamp keeping isolated nodes at 0.
            agg = torch.zeros_like(node_emb)
            counts = torch.zeros(node_emb.size(0), device=messages.device, dtype=messages.dtype)
            counts.index_add_(0, dst_s, torch.ones_like(messages[:, 0]))
            agg.index_add_(0, dst_s, messages)
            agg = agg / counts.clamp(min=1.0).unsqueeze(-1)
            aggregated = aggregated + agg
        updated = self.update(torch.cat([node_emb, aggregated], dim=-1))
        return self.norm(updated)


class GraphEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        config: dict,
        fuse_physical_to_node: bool = True,
    ):
        super().__init__()
        self.edge_dim = int(edge_dim)
        hidden_dim = int(config["hidden_dim"])
        activation = config.get("activation", "relu")
        dropout = float(config.get("dropout", 0.0))
        layer_norm = bool(config.get("layer_norm", True))
        self.residual = bool(config.get("residual", True))
        self.node_proj = build_mlp(node_dim, [hidden_dim], hidden_dim, activation, dropout)
        # 物理链路边与逻辑需求边使用各自独立的输入投影：两类边特征语义
        # 不同（链路能力 vs 请求压力），投影阶段就不应共享同一变换；
        # 投影后仍在同一 hidden 空间，消息传递层再按类型各自聚合。
        self.edge_proj_phys = build_mlp(edge_dim, [hidden_dim], hidden_dim, activation, dropout)
        self.edge_proj_demand = build_mlp(edge_dim, [hidden_dim], hidden_dim, activation, dropout)
        self.layers = nn.ModuleList(
            [
                EdgeConditionedGraphLayer(
                    hidden_dim,
                    hidden_dim,
                    activation,
                    dropout,
                    layer_norm,
                    fuse_physical_to_node=fuse_physical_to_node,
                )
                for _ in range(int(config["num_layers"]))
            ]
        )

    def forward(self, tensors: GraphTensors) -> tuple[torch.Tensor, torch.Tensor]:
        node_emb = self.node_proj(tensors.node_features)
        num_phys = int(tensors.num_physical_directed)
        # Skip the empty type's projection kernel (common when a scenario has
        # no demand edges) instead of launching a no-op GEMM every forward.
        edge_parts: list[torch.Tensor] = []
        if num_phys > 0:
            edge_parts.append(self.edge_proj_phys(tensors.edge_features_directed[:num_phys]))
        if num_phys < tensors.edge_features_directed.size(0):
            edge_parts.append(self.edge_proj_demand(tensors.edge_features_directed[num_phys:]))
        edge_emb = (
            torch.cat(edge_parts, dim=0)
            if edge_parts
            else tensors.edge_features_directed.new_zeros((0, node_emb.size(1)))
        )
        for layer in self.layers:
            next_node_emb = layer(node_emb, tensors.edge_index, edge_emb, tensors.num_physical_directed)
            node_emb = node_emb + next_node_emb if self.residual else next_node_emb
        return node_emb, edge_emb


class SharedNodeActor(nn.Module):
    def __init__(self, hidden_dim: int, config: dict, invalid_logit_value: float):
        super().__init__()
        activation = config.get("activation", "relu")
        dropout = float(config.get("dropout", 0.0))
        self.mode = config.get("mode", "mixed")
        # Both modes score an edge from its two endpoint node embeddings plus
        # the physical edge embedding. In demand_edge mode the encoder still
        # keeps physical-link messages out of the node representation, so the
        # node embeddings carry the dynamic demand signal without being
        # polluted by raw link rates; the actor can therefore see pending
        # demand even on slots where relay_importance is empty.
        edge_scorer_input_dim = hidden_dim * 3
        self.edge_scorer = build_mlp(
            edge_scorer_input_dim,
            list(config["actor"]["edge_scorer_hidden_dims"]),
            1,
            activation,
            dropout,
        )
        self.idle_scorer = build_mlp(
            hidden_dim,
            list(config["actor"]["idle_scorer_hidden_dims"]),
            1,
            activation,
            dropout,
        )
        self.invalid_logit_value = invalid_logit_value
        self.temperature = float(config["actor"].get("temperature", 1.0))
        self._node_idx: dict[str, int] | None = None
        self._action_to_edge: dict[tuple[str, str], str] | None = None

    def _ensure_static(self, action_space: NodeActionSpace) -> None:
        """Build the topology-dependent maps once (node order and candidate
        actions are static; only masks/edge filtering change per step)."""
        if self._node_idx is not None:
            return
        self._node_idx = {node_id: i for i, node_id in enumerate(action_space.node_ids)}
        self._node_ids = list(action_space.node_ids)
        self._action_to_edge = {
            (node_id, action): edge_id
            for node_id in action_space.node_ids
            for action in action_space.candidates_for_node(node_id)
            if (edge_id := action_space.action_to_edge(node_id, action)) is not None
        }
        # Flat static candidate tables (one row per candidate across all nodes,
        # in candidate order): src/dst node positions, edge id, idle flag, and
        # the per-node candidate offsets. The per-step legal mask selects rows.
        cand_srcs: list[int] = []
        cand_dsts: list[int] = []
        cand_edge_ids: list[str] = []
        cand_is_idle: list[bool] = []
        cand_offsets: list[int] = []
        for i, node_id in enumerate(self._node_ids):
            cand_offsets.append(len(cand_srcs))
            for action in action_space.candidates_for_node(node_id):
                cand_srcs.append(i)
                if action == NodeActionSpace.IDLE:
                    cand_dsts.append(-1)
                    cand_edge_ids.append("")
                    cand_is_idle.append(True)
                else:
                    cand_dsts.append(self._node_idx[action])
                    cand_edge_ids.append(self._action_to_edge[(node_id, action)])
                    cand_is_idle.append(False)
        self._cand_srcs = np.asarray(cand_srcs, dtype=np.int64)
        self._cand_dsts = np.asarray(cand_dsts, dtype=np.int64)
        self._cand_edge_ids = np.asarray(cand_edge_ids)
        self._cand_is_idle = np.asarray(cand_is_idle, dtype=bool)
        self._cand_offsets = np.asarray(cand_offsets, dtype=np.int64)
        self._full_lengths = [
            len(action_space.candidates_for_node(node_id)) for node_id in self._node_ids
        ]
        self._total_cands = len(cand_srcs)
        # Static "which node owns this candidate row" table; np.repeat over the
        # full candidate list is identical every step, so build it once.
        self._full_node_of = np.repeat(np.arange(len(self._node_ids), dtype=np.int64), self._full_lengths)

    def _build_plan(self, obs: GraphObservation, action_space: NodeActionSpace):
        """Candidate->index plan shared by ``act`` and ``evaluate_actions``.

        Depends only on the (stable) observation structure, so it is cached on
        the obs object: the rollout buffer reuses the same obs during PPO
        update passes, where this Python loop would otherwise run again for
        every minibatch epoch. Uses the raw masks for a fully vectorized plan
        when available (real env observations); manual test fixtures without
        ``raw_action_masks`` fall back to the Python loop.
        """
        self._ensure_static(action_space)
        raw_masks = obs.raw_action_masks
        if (
            raw_masks is not None
            and list(obs.node_ids) == self._node_ids
            and [len(raw_masks[node_id]) for node_id in self._node_ids] == self._full_lengths
        ):
            return self._build_plan_vectorized(obs, raw_masks)
        return self._build_plan_loop(obs, action_space)

    def _build_plan_vectorized(
        self, obs: GraphObservation, raw_masks: dict[str, list[bool]]
    ) -> tuple:
        """Numpy plan from the raw per-node masks: one flat legal-candidate
        gather instead of per-node Python loops over candidates."""
        node_ids = self._node_ids
        n_nodes = len(node_ids)
        flat_legal = obs.flat_action_masks
        if flat_legal is None or flat_legal.size != self._total_cands:
            flat_legal = np.empty(self._total_cands, dtype=bool)
            for i, node_id in enumerate(node_ids):
                start = self._cand_offsets[i]
                mask_list = raw_masks[node_id]
                flat_legal[start : start + len(mask_list)] = mask_list
        lengths = [len(obs.action_candidates[node_id]) for node_id in node_ids]
        max_n = max(lengths) if lengths else 0
        legal_pos = np.flatnonzero(flat_legal)
        if legal_pos.size == 0:
            idx_mat = np.full((n_nodes, max_n), -1, dtype=np.int64)
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                lengths,
                max_n,
                idx_mat,
            )
        node_ids_for_legit = self._full_node_of[legal_pos]
        lengths_np = np.asarray(lengths, dtype=np.int64)
        # within-node index among the LEGAL candidates (0..n_i-1), not among
        # the full candidate list, so it stays below max_n.
        within = np.arange(legal_pos.size, dtype=np.int64) - np.repeat(
            np.cumsum(lengths_np) - lengths_np, lengths_np
        )
        is_idle = self._cand_is_idle[legal_pos]
        edge_sel = ~is_idle
        n_edge = int(edge_sel.sum())
        n_idle = legal_pos.size - n_edge
        edge_srcs = self._cand_srcs[legal_pos][edge_sel]
        edge_dsts = self._cand_dsts[legal_pos][edge_sel]
        idle_srcs = self._cand_srcs[legal_pos][~edge_sel]
        edge_ids = self._cand_edge_ids[legal_pos][edge_sel]
        if edge_ids.size:
            # Dict lookup is O(1) and fails loudly when a candidate edge is
            # not in the physical edge list; searchsorted could silently map
            # a missing id to the wrong position.
            edge_pos_by_id = {
                edge_id: 2 * pos for pos, edge_id in enumerate(obs.physical_edge_ids)
            }
            edge_poss = np.fromiter(
                (edge_pos_by_id[edge_id] for edge_id in edge_ids),
                dtype=np.int64,
                count=len(edge_ids),
            )
        else:
            edge_poss = np.empty(0, dtype=np.int64)
        # Flat score positions: all edge scores first, idle scores after.
        flat_pos = np.empty(legal_pos.size, dtype=np.int64)
        flat_pos[edge_sel] = np.arange(n_edge, dtype=np.int64)
        flat_pos[~edge_sel] = n_edge + np.arange(n_idle, dtype=np.int64)
        idx_mat = np.full((n_nodes, max_n), -1, dtype=np.int64)
        idx_mat[node_ids_for_legit, within] = flat_pos
        return (edge_srcs, edge_dsts, edge_poss, idle_srcs, lengths, max_n, idx_mat)

    def _build_plan_loop(self, obs: GraphObservation, action_space: NodeActionSpace) -> tuple:
        """Python-loop plan used when raw masks are unavailable (test
        fixtures); returns the same 7-tuple layout as the vectorized path."""
        node_index = self._node_idx
        action_to_edge = self._action_to_edge
        edge_pos_by_edge: dict[str, int] = {
            edge_id: edge_pos * 2 for edge_pos, edge_id in enumerate(obs.physical_edge_ids)
        }
        edge_srcs: list[int] = []
        edge_dsts: list[int] = []
        edge_poss: list[int] = []
        idle_srcs: list[int] = []
        order: dict[str, list[tuple[str, int]]] = {node_id: [] for node_id in obs.node_ids}
        for node_id in obs.node_ids:
            src_idx = node_index[node_id]
            for action in obs.action_candidates[node_id]:
                if action == NodeActionSpace.IDLE:
                    order[node_id].append(("idle", len(idle_srcs)))
                    idle_srcs.append(src_idx)
                    continue
                dst_idx = node_index[action]
                edge_id = action_to_edge[(node_id, action)]
                edge_pos = edge_pos_by_edge[edge_id]
                order[node_id].append(("edge", len(edge_srcs)))
                edge_srcs.append(src_idx)
                edge_dsts.append(dst_idx)
                edge_poss.append(edge_pos)
        lengths = [len(obs.action_candidates[node_id]) for node_id in obs.node_ids]
        max_n = max(lengths) if lengths else 0
        n_edge = len(edge_srcs)
        score_idx_rows: list[list[int]] = []
        for node_id in obs.node_ids:
            row_idx = [pos if kind == "edge" else n_edge + pos for kind, pos in order[node_id]]
            pad = max_n - len(row_idx)
            score_idx_rows.append(row_idx + [-1] * pad)
        return (
            np.asarray(edge_srcs, dtype=np.int64),
            np.asarray(edge_dsts, dtype=np.int64),
            np.asarray(edge_poss, dtype=np.int64),
            np.asarray(idle_srcs, dtype=np.int64),
            lengths,
            max_n,
            np.asarray(score_idx_rows, dtype=np.int64),
        )

    def forward(
        self,
        obs: GraphObservation,
        node_emb: torch.Tensor,
        edge_emb_directed: torch.Tensor,
        action_space: NodeActionSpace,
        build_logits_dict: bool = True,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[str], list[int], dict[str, torch.Tensor]]:
        plan = getattr(obs, "_actor_plan", None)
        if plan is None:
            plan = self._build_plan(obs, action_space)
            setattr(obs, "_actor_plan", plan)
        edge_srcs, edge_dsts, edge_poss, idle_srcs, lengths, max_n, score_idx_rows = plan

        device = node_emb.device
        if edge_srcs.size:
            src_t = torch.from_numpy(edge_srcs).to(device)
            dst_t = torch.from_numpy(edge_dsts).to(device)
            pos_t = torch.from_numpy(edge_poss).to(device)
            pair_emb = torch.cat(
                [node_emb[src_t], node_emb[dst_t], edge_emb_directed[pos_t]],
                dim=-1,
            )
            edge_scores = self.edge_scorer(pair_emb).squeeze(-1)
        else:
            edge_scores = torch.zeros((0,), dtype=torch.float32, device=device)
        if idle_srcs.size:
            idle_srcs_t = torch.from_numpy(idle_srcs).to(device)
            idle_scores = self.idle_scorer(node_emb[idle_srcs_t]).squeeze(-1)
        else:
            idle_scores = torch.zeros((0,), dtype=torch.float32, device=device)

        # Batched logit assembly: pad every node's candidate scores to a common
        # width and apply the invalid-logit masking with a single tensor op.
        # score_idx_rows encodes the per-node candidate order (-1 = padding),
        # so no separate mask matrix is needed (the masks are already applied
        # when the observation's candidate lists were filtered at graph build).
        if obs.node_ids:
            n_nodes = len(obs.node_ids)
            n_edge = edge_scores.size(0)
            all_scores = (
                torch.cat([edge_scores, idle_scores])
                if idle_scores.numel()
                else edge_scores
            )
            idx_t = torch.from_numpy(score_idx_rows).to(device)
            valid = idx_t >= 0
            raw = torch.where(
                valid,
                all_scores[idx_t.clamp(min=0)],
                torch.full((n_nodes, max_n), float("-inf"), dtype=torch.float32, device=device),
            )
            if self.temperature != 1.0:
                raw = raw / self.temperature
            masked = torch.where(valid, raw, torch.full_like(raw, self.invalid_logit_value))
        else:
            masked = torch.zeros((0, 0), dtype=torch.float32, device=device)
            lengths = []
        if build_logits_dict:
            logits = {
                node_id: masked[i, : lengths[i]].contiguous()
                for i, node_id in enumerate(obs.node_ids)
            }
        else:
            # Rollout only needs the padded tensor; skip ~143 small GPU
            # slice+contiguous ops per step (tests request the dict via the
            # default flag).
            logits = {}
        edge_map = _edge_score_map(plan, self._node_idx, self._action_to_edge, edge_scores)
        return logits, masked, list(obs.node_ids), lengths, edge_map


class GlobalCritic(nn.Module):
    def __init__(self, hidden_dim: int, config: dict):
        super().__init__()
        activation = config.get("activation", "relu")
        dropout = float(config.get("dropout", 0.0))
        self.pooling = config["critic"].get("pooling", "mean")
        if self.pooling == "typed_mean":
            # 节点平均 + 物理边平均 + 逻辑边平均 + 3 个图规模计数。
            value_input_dim = hidden_dim * 3 + 3
        elif self.pooling == "mean":
            value_input_dim = hidden_dim * 2
        else:
            raise NotImplementedError(f"Unsupported critic pooling: {self.pooling}")
        self.value_head = build_mlp(
            value_input_dim,
            list(config["critic"]["hidden_dims"]),
            1,
            activation,
            dropout,
        )

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_emb_directed: torch.Tensor,
        num_physical_directed: int = 0,
    ) -> torch.Tensor:
        hidden = node_emb.size(1)
        device = node_emb.device
        node_pool = node_emb.mean(dim=0)
        num_physical_directed = int(num_physical_directed)
        num_demand_directed = edge_emb_directed.size(0) - num_physical_directed
        if self.pooling == "typed_mean":
            if num_physical_directed > 0:
                physical_pool = edge_emb_directed[:num_physical_directed].mean(dim=0)
            else:
                physical_pool = torch.zeros(hidden, device=device)
            if num_demand_directed > 0:
                demand_pool = edge_emb_directed[num_physical_directed:].mean(dim=0)
            else:
                demand_pool = torch.zeros(hidden, device=device)
            # log1p 压缩数量级，让 Critic 感知图的规模（节点数/物理边数/请求对数）。
            scale_counts = torch.log1p(
                torch.tensor(
                    [node_emb.size(0), num_physical_directed // 2, num_demand_directed // 2],
                    dtype=torch.float32,
                    device=device,
                )
            )
            graph_emb = torch.cat([node_pool, physical_pool, demand_pool, scale_counts], dim=-1)
        else:
            if edge_emb_directed.shape[0] == 0:
                # Mask-first filtering may leave the graph with no physical links.
                edge_pool = torch.zeros(hidden, device=device)
            else:
                edge_pool = edge_emb_directed.mean(dim=0)
            graph_emb = torch.cat([node_pool, edge_pool], dim=-1)
        return self.value_head(graph_emb).squeeze(-1)


class GraphMAPPOActorCritic(nn.Module):
    def __init__(self, action_space: NodeActionSpace, config: dict):
        super().__init__()
        model_cfg = config["model"]
        self.mode = model_cfg.get("mode", "mixed")
        feature_dims = config["features"]["dims"]
        history_dim = int(feature_dims.get("history_dim_resolved", 0))
        node_history_dim = int(feature_dims.get("node_history_dim_resolved", history_dim))
        # The HistoryEncoder output is concatenated to the base node/edge
        # features before the GNN, so the encoder input dims include history_dim.
        node_dim = int(feature_dims["node_dim_resolved"]) + node_history_dim
        edge_dim = int(feature_dims["edge_dim_resolved"]) + history_dim
        hidden_dim = int(model_cfg["encoder"]["hidden_dim"])
        invalid_logit_value = float(model_cfg["distribution"]["invalid_logit_value"])
        self.action_space = action_space
        self.history_dim = history_dim
        self.node_history_dim = node_history_dim
        history_cfg = config["features"].get("history_encoder", {})
        self.history_encoder = HistoryEncoder(config) if history_cfg.get("enabled", False) else None
        self.encoder = GraphEncoder(
            node_dim,
            edge_dim,
            model_cfg["encoder"],
            fuse_physical_to_node=(self.mode == "mixed"),
        )
        self.actor = SharedNodeActor(hidden_dim, model_cfg, invalid_logit_value)
        self.critic = GlobalCritic(hidden_dim, model_cfg)

    def forward(
        self,
        obs: GraphObservation,
        device: torch.device | str = "cpu",
        build_logits_dict: bool = True,
    ) -> ActorCriticOutput:
        # Empty-graph fallback must use the base edge dim (without history)
        # because history embeddings are concatenated below.
        base_edge_dim = self.encoder.edge_dim - self.history_dim
        device = torch.device(device)
        # A rollout buffer owns every observation until the PPO update ends.
        # Caching CUDA tensors on those observations therefore retains one
        # graph's inputs per rollout step and exhausts GPU memory on full-day
        # rollouts. CPU caching is small and useful for CPU training, while
        # CUDA tensors are intentionally scoped to this forward/chunk.
        cache = getattr(obs, "_tensors_cache", None) if device.type == "cpu" else None
        device_key = str(device)
        if cache is not None and cache[0] == device_key:
            tensors = cache[1]
        else:
            tensors = observation_to_tensors(obs, device, edge_dim=base_edge_dim)
            node_features = tensors.node_features
            edge_features_directed = tensors.edge_features_directed
            if self.history_encoder is not None:
                node_h, phys_h, demand_h = self.history_encoder(obs, device)
                if node_h is not None:
                    node_features = torch.cat([node_features, node_h], dim=-1)
                elif self.node_history_dim > 0:
                    node_features = torch.cat(
                        [
                            node_features,
                            node_features.new_zeros(
                                (node_features.size(0), self.node_history_dim)
                            ),
                        ],
                        dim=-1,
                    )
                # Physical edges occupy the first num_physical_directed rows of
                # the directed edge tensor, demand edges follow; embed each type
                # with its own h (repeat_interleave because one undirected
                # physical edge is stored as two directed rows).
                h_rows = []
                if self.history_dim > 0:
                    n_phys_directed = int(tensors.num_physical_directed)
                    if phys_h is not None:
                        h_rows.append(phys_h.repeat_interleave(2, dim=0))
                    else:
                        h_rows.append(
                            edge_features_directed.new_zeros(
                                (n_phys_directed, self.history_dim)
                            )
                        )
                    n_demand_directed = edge_features_directed.size(0) - n_phys_directed
                    if demand_h is not None:
                        h_rows.append(demand_h.repeat_interleave(2, dim=0))
                    else:
                        h_rows.append(
                            edge_features_directed.new_zeros(
                                (n_demand_directed, self.history_dim)
                            )
                        )
                if h_rows:
                    edge_h = torch.cat(h_rows, dim=0)
                else:
                    edge_h = torch.zeros((0, self.history_dim), dtype=torch.float32, device=device)
                edge_features_directed = torch.cat([edge_features_directed, edge_h], dim=-1)
            tensors = GraphTensors(
                node_features=node_features,
                edge_index=tensors.edge_index,
                edge_features_directed=edge_features_directed,
                node_ids=tensors.node_ids,
                edge_ids=tensors.edge_ids,
                num_physical_directed=tensors.num_physical_directed,
            )
            if device.type == "cpu":
                setattr(obs, "_tensors_cache", (device_key, tensors))
        node_emb, edge_emb = self.encoder(tensors)
        logits, logits_padded, node_order, lengths, edge_scores = self.actor(
            obs, node_emb, edge_emb, self.action_space, build_logits_dict=build_logits_dict
        )
        return ActorCriticOutput(
            logits=logits,
            value=self.critic(node_emb, edge_emb, tensors.num_physical_directed),
            logits_padded=logits_padded,
            logits_node_order=node_order,
            logits_lengths=lengths,
            edge_scores=edge_scores,
        )

    def batched_forward(
        self,
        obs_list: list[GraphObservation],
        device: torch.device | str = "cpu",
    ) -> BatchedActorCriticOutput:
        """Block-diagonal batching of several observations into one forward.

        Each graph keeps its own nodes/edges (block-diagonal adjacency); edges
        are permuted so all physical links precede all demand links (sum-based
        message passing is order-independent); actor scoring and critic pooling
        are computed per graph. The per-graph math is identical to ``forward``,
        only the small per-graph CUDA kernels are merged into larger batched
        ones, so this is a pure implementation-level speedup for PPO minibatches.
        """
        device = torch.device(device)
        base_edge_dim = self.encoder.edge_dim - self.history_dim
        self.actor._ensure_static(self.action_space)
        if self.actor._node_idx is None or self.actor._action_to_edge is None:
            raise RuntimeError("SharedNodeActor static maps were not initialized.")
        tensors_list: list[GraphTensors] = []
        for obs in obs_list:
            cache = getattr(obs, "_tensors_cache", None) if device.type == "cpu" else None
            if cache is not None and cache[0] == str(device):
                tensors_list.append(cache[1])
                continue
            tensors = observation_to_tensors(obs, device, edge_dim=base_edge_dim)
            if self.history_encoder is not None:
                node_features = tensors.node_features
                edge_features_directed = tensors.edge_features_directed
                node_h, phys_h, demand_h = self.history_encoder(obs, device)
                if node_h is not None:
                    node_features = torch.cat([node_features, node_h], dim=-1)
                elif self.node_history_dim > 0:
                    node_features = torch.cat(
                        [
                            node_features,
                            node_features.new_zeros(
                                (node_features.size(0), self.node_history_dim)
                            ),
                        ],
                        dim=-1,
                    )
                h_rows = []
                if self.history_dim > 0:
                    n_phys_directed = int(tensors.num_physical_directed)
                    if phys_h is not None:
                        h_rows.append(phys_h.repeat_interleave(2, dim=0))
                    else:
                        h_rows.append(
                            edge_features_directed.new_zeros(
                                (n_phys_directed, self.history_dim)
                            )
                        )
                    n_demand_directed = edge_features_directed.size(0) - n_phys_directed
                    if demand_h is not None:
                        h_rows.append(demand_h.repeat_interleave(2, dim=0))
                    else:
                        h_rows.append(
                            edge_features_directed.new_zeros(
                                (n_demand_directed, self.history_dim)
                            )
                        )
                if h_rows:
                    edge_h = torch.cat(h_rows, dim=0)
                else:
                    edge_h = torch.zeros((0, self.history_dim), dtype=torch.float32, device=device)
                edge_features_directed = torch.cat([edge_features_directed, edge_h], dim=-1)
                tensors = GraphTensors(
                    node_features=node_features,
                    edge_index=tensors.edge_index,
                    edge_features_directed=edge_features_directed,
                    node_ids=tensors.node_ids,
                    edge_ids=tensors.edge_ids,
                    num_physical_directed=tensors.num_physical_directed,
                )
            if device.type == "cpu":
                setattr(obs, "_tensors_cache", (str(device), tensors))
            tensors_list.append(tensors)

        node_off: list[int] = []
        edge_off: list[int] = []
        n_phys: list[int] = []
        n_demand: list[int] = []
        node_acc = 0
        edge_acc = 0
        for tensors in tensors_list:
            node_off.append(node_acc)
            node_acc += int(tensors.node_features.size(0))
            edge_off.append(edge_acc)
            n_phys.append(int(tensors.num_physical_directed))
            n_demand.append(int(tensors.edge_features_directed.size(0)) - int(tensors.num_physical_directed))
            edge_acc += int(tensors.edge_features_directed.size(0))

        node_features_all = torch.cat([t.node_features for t in tensors_list])
        edge_features_all = torch.cat([t.edge_features_directed for t in tensors_list])
        edge_index_all = torch.cat(
            [t.edge_index + off for t, off in zip(tensors_list, node_off)], dim=1
        )
        # Permute edges: all physical edges (per graph, in order) first, then
        # all demand edges, so the shared GNN layer keeps a single split point.
        perm_parts: list[torch.Tensor] = []
        for i in range(len(tensors_list)):
            n = n_phys[i]
            if n > 0:
                perm_parts.append(torch.arange(edge_off[i], edge_off[i] + n, device=device))
        for i in range(len(tensors_list)):
            n = n_demand[i]
            if n > 0:
                perm_parts.append(torch.arange(edge_off[i] + n_phys[i], edge_off[i] + n_phys[i] + n, device=device))
        perm = torch.cat(perm_parts) if perm_parts else torch.zeros((0,), dtype=torch.long, device=device)
        num_phys_total = sum(n_phys)
        edge_index_p = edge_index_all[:, perm]
        edge_features_p = edge_features_all[perm]

        node_emb = self.encoder.node_proj(node_features_all)
        edge_parts: list[torch.Tensor] = []
        if num_phys_total > 0:
            edge_parts.append(self.encoder.edge_proj_phys(edge_features_p[:num_phys_total]))
        if num_phys_total < edge_features_p.size(0):
            edge_parts.append(self.encoder.edge_proj_demand(edge_features_p[num_phys_total:]))
        edge_emb = (
            torch.cat(edge_parts, dim=0)
            if edge_parts
            else edge_features_p.new_zeros((0, node_emb.size(1)))
        )
        for layer in self.encoder.layers:
            next_node_emb = layer(node_emb, edge_index_p, edge_emb, num_phys_total)
            node_emb = node_emb + next_node_emb if self.encoder.residual else next_node_emb

        # Actor: merge per-graph candidate plans into one batched scoring pass.
        plans = []
        for obs in obs_list:
            plan = getattr(obs, "_actor_plan", None)
            if plan is None:
                plan = self.actor._build_plan(obs, self.action_space)
                setattr(obs, "_actor_plan", plan)
            plans.append(plan)
        # perm[i] is the ORIGINAL position of the edge at permuted position i,
        # so the inverse (argsort) maps original positions -> permuted positions,
        # which is what the actor's per-graph edge_poss refer to.
        perm_np = perm.detach().cpu().numpy()
        perm_inv_np = np.argsort(perm_np)
        src_parts: list[np.ndarray] = []
        dst_parts: list[np.ndarray] = []
        pos_parts: list[np.ndarray] = []
        idle_parts: list[np.ndarray] = []
        edge_score_off: list[int] = []
        idle_score_off: list[int] = []
        n_edge_cand = 0
        n_idle_cand = 0
        for i, plan in enumerate(plans):
            edge_score_off.append(n_edge_cand)
            n_edge_cand += int(plan[0].size)
            idle_score_off.append(n_idle_cand)
            n_idle_cand += int(plan[3].size)
            if plan[0].size:
                src_parts.append(plan[0] + node_off[i])
                dst_parts.append(plan[1] + node_off[i])
                pos_parts.append(perm_inv_np[edge_off[i] + plan[2]])
            if plan[3].size:
                idle_parts.append(plan[3] + node_off[i])
        if src_parts:
            src_t = torch.from_numpy(np.concatenate(src_parts)).to(device)
            dst_t = torch.from_numpy(np.concatenate(dst_parts)).to(device)
            pos_t = torch.from_numpy(np.concatenate(pos_parts)).to(device)
            pair_emb = torch.cat([node_emb[src_t], node_emb[dst_t], edge_emb[pos_t]], dim=-1)
            edge_scores = self.actor.edge_scorer(pair_emb).squeeze(-1)
        else:
            edge_scores = torch.zeros((0,), dtype=torch.float32, device=device)
        if idle_parts:
            idle_t = torch.from_numpy(np.concatenate(idle_parts)).to(device)
            idle_scores = self.actor.idle_scorer(node_emb[idle_t]).squeeze(-1)
        else:
            idle_scores = torch.zeros((0,), dtype=torch.float32, device=device)

        # Per-graph edge_id -> raw score maps for the priority-matching
        # resolver (see _edge_score_map).
        edge_score_maps: list[dict[str, torch.Tensor]] = []
        for i, plan in enumerate(plans):
            n_edge_i = int(plan[0].size)
            graph_scores = (
                edge_scores[edge_score_off[i] : edge_score_off[i] + n_edge_i]
                if n_edge_i
                else edge_scores.new_zeros((0,))
            )
            edge_score_maps.append(
                _edge_score_map(
                    plan,
                    self.actor._node_idx,
                    self.actor._action_to_edge,
                    graph_scores,
                )
            )

        # Remap each node's candidate rows to the global score positions.
        global_max_n = max(int(plan[5]) for plan in plans) if plans else 0
        total_edge_cand = n_edge_cand
        row_parts: list[np.ndarray] = []
        for i, plan in enumerate(plans):
            n_edge_i = int(plan[0].size)
            rows = plan[6]
            remapped = np.where(
                rows < 0,
                -1,
                np.where(
                    rows < n_edge_i,
                    edge_score_off[i] + rows,
                    total_edge_cand + idle_score_off[i] + (rows - n_edge_i),
                ),
            )
            n_i, w_i = rows.shape
            out = np.full((n_i, global_max_n), -1, dtype=np.int64)
            out[:, :w_i] = remapped
            row_parts.append(out)
        if row_parts:
            idx_t = torch.from_numpy(np.concatenate(row_parts, axis=0)).to(device)
            all_scores = torch.cat([edge_scores, idle_scores]) if n_idle_cand else edge_scores
            valid = idx_t >= 0
            raw = torch.where(
                valid,
                all_scores[idx_t.clamp(min=0)],
                torch.full((idx_t.size(0), global_max_n), float("-inf"), dtype=torch.float32, device=device),
            )
            if self.actor.temperature != 1.0:
                raw = raw / self.actor.temperature
            masked_all = torch.where(valid, raw, torch.full_like(raw, self.actor.invalid_logit_value))
        else:
            masked_all = torch.zeros((0, global_max_n), dtype=torch.float32, device=device)

        padded_logits: list[torch.Tensor] = []
        node_orders: list[list[str]] = []
        lengths_list: list[list[int]] = []
        for i, obs in enumerate(obs_list):
            n_i = int(tensors_list[i].node_features.size(0))
            padded_logits.append(masked_all[node_off[i]:node_off[i] + n_i])
            node_orders.append(list(obs.node_ids))
            lengths_list.append(list(plans[i][4]))

        # Critic: per-graph typed pooling over the permuted edge embeddings, then
        # ONE value-head MLP over the stacked graph embeddings instead of one
        # tiny MLP per graph (the per-graph pooling math is identical to
        # ``GlobalCritic.forward``, only the MLP launch is merged).
        values: list[torch.Tensor] = []
        cum_phys = 0
        cum_demand = 0
        hidden = node_emb.size(1)
        pooling = self.critic.pooling
        value_inputs: list[torch.Tensor] = []
        for i in range(len(tensors_list)):
            n_i = int(tensors_list[i].node_features.size(0))
            node_pool = node_emb[node_off[i]:node_off[i] + n_i].mean(dim=0)
            phys_slice = edge_emb[cum_phys:cum_phys + n_phys[i]]
            demand_slice = edge_emb[num_phys_total + cum_demand:num_phys_total + cum_demand + n_demand[i]]
            if pooling == "typed_mean":
                if n_phys[i] > 0:
                    physical_pool = phys_slice.mean(dim=0)
                else:
                    physical_pool = torch.zeros(hidden, device=edge_emb.device, dtype=edge_emb.dtype)
                if n_demand[i] > 0:
                    demand_pool = demand_slice.mean(dim=0)
                else:
                    demand_pool = torch.zeros(hidden, device=edge_emb.device, dtype=edge_emb.dtype)
                scale_counts = torch.log1p(
                    torch.tensor(
                        [n_i, n_phys[i] // 2, n_demand[i] // 2],
                        dtype=torch.float32,
                        device=edge_emb.device,
                    )
                )
                value_inputs.append(torch.cat([node_pool, physical_pool, demand_pool, scale_counts], dim=-1))
            else:
                if n_phys[i] or n_demand[i]:
                    edge_i = torch.cat([phys_slice, demand_slice], dim=0)
                    edge_pool = edge_i.mean(dim=0)
                else:
                    edge_pool = torch.zeros(hidden, device=edge_emb.device, dtype=edge_emb.dtype)
                value_inputs.append(torch.cat([node_pool, edge_pool], dim=-1))
            cum_phys += n_phys[i]
            cum_demand += n_demand[i]
        if value_inputs:
            values_all = self.critic.value_head(torch.stack(value_inputs, dim=0)).squeeze(-1)
            values = [values_all[i] for i in range(len(tensors_list))]
        else:
            # Degenerate fallback (no nodes at all): mirror GlobalCritic's
            # empty-graph behavior per graph.
            for i in range(len(tensors_list)):
                empty = edge_emb.new_empty((0, hidden))
                values.append(self.critic(node_emb.new_empty((0, hidden)), empty, 0))

        return BatchedActorCriticOutput(
            padded_logits=padded_logits,
            node_orders=node_orders,
            lengths=lengths_list,
            values=values,
            edge_score_maps=edge_score_maps,
        )


def _nodes_from_edge_id(edge_id: str) -> tuple[str, str]:
    edge_name = edge_id[2:] if edge_id.startswith("E_") else edge_id
    if "__" not in edge_name:
        raise ValueError(f"Cannot infer edge endpoints from edge id {edge_id!r}.")
    return edge_name.split("__", 1)
