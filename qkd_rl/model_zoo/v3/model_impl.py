from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from qkd_rl.env.action_space import NodeActionSpace
from .graph_builder import GraphObservation
from qkd_rl.rl.models.history_encoder import HistoryEncoder
from qkd_rl.rl.models.mlp import build_mlp


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
    edge_scores: dict[tuple[str, str], torch.Tensor] | None = None
    arc_embeddings: torch.Tensor | None = None


@dataclass
class BatchedActorCriticOutput:
    # Per-observation padded logits (one row per node of that graph), node
    # order, per-node candidate lengths, and critic value.
    padded_logits: list[torch.Tensor]
    node_orders: list[list[str]]
    lengths: list[list[int]]
    values: list[torch.Tensor]
    edge_score_maps: list[dict[tuple[str, str], torch.Tensor]]
    # Per graph: (src_local_idx, dst_local_idx, scores) for the directed edge
    # candidates, in the same candidate order as ``edge_score_maps``. The
    # matching sampler only needs vectors, so handing it these skips rebuilding
    # a ~300-entry arc dict per graph per step (~8% of a batched rollout step)
    # and lets the sampler run vectorized across graphs.
    edge_arrays: list[tuple[np.ndarray, np.ndarray, torch.Tensor]] | None = None
    arc_embeddings: list[torch.Tensor] | None = None


def _edge_score_map(
    plan: tuple,
    node_index: dict[str, int],
    edge_scores: torch.Tensor,
) -> dict[tuple[str, str], torch.Tensor]:
    """Map raw edge-scorer outputs back to directed arcs for one observation.

    ``plan[0]`` / ``plan[1]`` are the source/destination node positions of the
    legal edge candidates in flat candidate order, which is exactly the order
    of ``edge_scores`` returned by the batched scorer. Each candidate is the
    *directed* proposal of one node (``src -> dst``), so the map is keyed by
    the arc tuple: ``u -> v`` and ``v -> u`` keep their own learned scores.
    The global matching sampler can therefore distinguish directions instead
    of sharing one undirected pair score between both arcs.
    """
    if node_index is None:
        return {}
    # Position -> node id table plus `tolist()` on the index arrays: this runs
    # once per graph per forward (and per graph per PPO chunk), so the old
    # per-element `int(srcs[i])` + dict lookup cost more than the tensor work it
    # feeds. `edge_scores.unbind(0)` yields per-arc 0-dim views, which keeps the
    # autograd graph intact (unlike `.tolist()`).
    node_by_idx: list[str] = [""] * len(node_index)
    for node_id, idx in node_index.items():
        node_by_idx[idx] = node_id
    arc_map: dict[tuple[str, str], torch.Tensor] = {}
    for src_i, dst_i, score in zip(
        plan[0].tolist(), plan[1].tolist(), edge_scores.unbind(0)
    ):
        if src_i != dst_i:
            arc_map[(node_by_idx[src_i], node_by_idx[dst_i])] = score
    return arc_map


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


def _segment_ids(sizes: list[int], device: torch.device) -> torch.Tensor:
    """Graph id per row for a list of contiguous segment sizes."""
    if not sizes:
        return torch.zeros(0, dtype=torch.long, device=device)
    return torch.repeat_interleave(
        torch.arange(len(sizes), device=device),
        torch.tensor(sizes, dtype=torch.long, device=device),
    )


def _segment_sum(
    x: torch.Tensor, gid: torch.Tensor, n_segments: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum the rows of ``x`` grouped by ``gid``.

    Replaces the ``x[start:end]`` + ``mean(dim=0)`` idiom used for per-graph
    pooling. A slice of a gradient-carrying tensor records a ``SliceBackward``,
    whose backward allocates ``zeros_like(input)`` for the WHOLE input and
    copies the slice's gradient into it -- so pooling N graphs out of one
    ``[E, H]`` edge tensor memsets N full-size buffers. ``index_add_`` into a
    ``[n_segments, H]`` buffer needs no such thing: the only zero buffer in its
    backward is that same small one.

    Returns ``(sums, counts)``; both are zero for an empty segment.
    """
    if x.size(0) == 0:
        return x.new_zeros((n_segments, x.size(1))), x.new_zeros(n_segments)
    sums = x.new_zeros((n_segments, x.size(1))).index_add_(0, gid, x)
    counts = x.new_zeros(n_segments).index_add_(0, gid, torch.ones_like(x[:, 0]))
    return sums, counts


def _segment_max(x: torch.Tensor, gid: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Per-segment **max** over the rows of ``x``.

    Why max exists alongside mean: the scheduler's bottleneck is a *min* over a
    path (``routing.partial_consume_for_request``: ``serve_now = min(hop_levels)``),
    so the discriminating signal sits in the single worst entity, not the
    average one. Mean pooling dilutes that entity to 1/N and the critic cannot
    see it; max is the aggregation that preserves it.

    Empty segments yield 0 (matching ``_segment_sum``'s placeholder), not -inf,
    so a graph with no demand edges contributes no spurious extreme.
    """
    if x.size(0) == 0:
        return x.new_zeros((n_segments, x.size(1)))
    out = x.new_full((n_segments, x.size(1)), float("-inf")).scatter_reduce_(
        0, gid.unsqueeze(1).expand_as(x), x, reduce="amax", include_self=True
    )
    counts = x.new_zeros(n_segments).index_add_(0, gid, torch.ones_like(x[:, 0]))
    return torch.where((counts == 0).unsqueeze(1), torch.zeros_like(out), out)


class EdgeConditionedGraphLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        activation: str,
        dropout: float,
        layer_norm: bool,
        fuse_physical_to_node: bool = True,
        fuse_demand_to_node: bool = True,
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
        # 节点→边 更新：边吸收两端点的最新节点上下文，把"需求压力 / 链路能力"
        # 的扩散结果写回边嵌入，跨层累积后边嵌入即携带通路上下文。
        # 物理边与需求边语义不同（链路能力 vs 请求压力），各自独立 MLP。
        self.edge_update_mlp_phys = build_mlp(
            input_dim=hidden_dim * 3,
            hidden_dims=[hidden_dim],
            output_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
        )
        self.edge_update_mlp_demand = build_mlp(
            input_dim=hidden_dim * 3,
            hidden_dims=[hidden_dim],
            output_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
        )
        self.edge_norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.fuse_physical_to_node = fuse_physical_to_node
        self.fuse_demand_to_node = fuse_demand_to_node

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        phys_emb: torch.Tensor,
        demand_emb: torch.Tensor,
        num_physical_directed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One message-passing step.

        The two edge types arrive as separate tensors rather than as one
        concatenated tensor plus a split point. That is not cosmetic: a slice of
        a tensor that requires grad costs a ``SliceBackward``, whose backward
        materialises ``zeros_like(input)`` for the WHOLE input and copies the
        slice's gradient into it. With a single ``[E, H]`` edge tensor (E ~ 100k
        for a minibatch step) each slice therefore zeroed ~50 MB, and this layer
        took six slices per layer -- profiled at 11.2 s of a 165 s update, 32 GB
        of memset. Kept apart, the same math costs no slice at all. LayerNorm is
        row-wise, so normalising the two parts separately is exactly equivalent.
        """
        src, dst = edge_index
        n_phys = int(num_physical_directed)
        src_p, dst_p = src[:n_phys], dst[:n_phys]
        src_d, dst_d = src[n_phys:], dst[n_phys:]

        aggregated = torch.zeros_like(node_emb)
        # 物理边与逻辑请求边各自独立聚合并按各自的邻居数归一化再相加，
        # 保证两类信息互不稀释。
        if self.fuse_physical_to_node and n_phys > 0:
            aggregated = aggregated + self._node_message(
                self.message_mlp_phys, node_emb, src_p, dst_p, phys_emb
            )
        if self.fuse_demand_to_node and demand_emb.size(0) > 0:
            aggregated = aggregated + self._node_message(
                self.message_mlp_demand, node_emb, src_d, dst_d, demand_emb
            )
        updated = self.update(torch.cat([node_emb, aggregated], dim=-1))
        new_node = self.norm(updated)
        # 节点→边：边嵌入吸收两端点的最新上下文（残差 + LayerNorm）。
        # 需求信息从需求边 → 端点节点 → 物理边逐层"落"到边嵌入上，
        # 物理边之间通过节点中介完成信息融合（边→节点→边）。
        new_phys = self._edge_message(self.edge_update_mlp_phys, phys_emb, new_node, src_p, dst_p)
        new_demand = self._edge_message(
            self.edge_update_mlp_demand, demand_emb, new_node, src_d, dst_d
        )
        return new_node, new_phys, new_demand

    @staticmethod
    def _node_message(
        message_mlp: nn.Module,
        node_emb: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_emb: torch.Tensor,
    ) -> torch.Tensor:
        messages = message_mlp(torch.cat([node_emb[src], edge_emb], dim=-1))
        # mean = sum / count via index_add_: scatter_reduce_("mean") is
        # measurably slower on CUDA (~26% in micro-benchmark) and can be
        # numerically unstable for small graphs; the two-kernel version is
        # exactly sum / count with clamp keeping isolated nodes at 0.
        agg = torch.zeros_like(node_emb, dtype=messages.dtype)
        counts = torch.zeros(node_emb.size(0), device=messages.device, dtype=messages.dtype)
        counts.index_add_(0, dst, torch.ones_like(messages[:, 0]))
        agg.index_add_(0, dst, messages)
        return agg / counts.clamp(min=1.0).unsqueeze(-1)

    def _edge_message(
        self,
        edge_mlp: nn.Module,
        edge_emb: torch.Tensor,
        new_node: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        if edge_emb.size(0) == 0:
            return edge_emb
        msg = edge_mlp(torch.cat([edge_emb, new_node[src], new_node[dst]], dim=-1))
        return self.edge_norm(msg + edge_emb)


class GraphEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        config: dict,
        fuse_physical_to_node: bool = True,
        fuse_demand_to_node: bool = True,
    ):
        super().__init__()
        self.edge_dim = int(edge_dim)
        hidden_dim = int(config["hidden_dim"])
        # Needed by project_edges to size the empty-type placeholder.
        self.hidden_dim = hidden_dim
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
                    fuse_demand_to_node=fuse_demand_to_node,
                )
                for _ in range(int(config["num_layers"]))
            ]
        )

    def project_edges(
        self, edge_features: torch.Tensor, num_phys: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the two edge types separately, avoiding a split later.

        Slicing the *input* is free -- it is a leaf produced from numpy and
        carries no gradient -- so the type split happens here, once, and the two
        tensors then travel side by side through every layer.
        """
        hidden = self.hidden_dim
        n_edges = edge_features.size(0)
        phys = (
            self.edge_proj_phys(edge_features[:num_phys])
            if num_phys > 0
            else edge_features.new_zeros((0, hidden))
        )
        demand = (
            self.edge_proj_demand(edge_features[num_phys:])
            if num_phys < n_edges
            else edge_features.new_zeros((0, hidden))
        )
        return phys, demand

    def run_layers(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        phys_emb: torch.Tensor,
        demand_emb: torch.Tensor,
        num_phys: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            next_node_emb, phys_emb, demand_emb = layer(
                node_emb, edge_index, phys_emb, demand_emb, num_phys
            )
            node_emb = node_emb + next_node_emb if self.residual else next_node_emb
        return node_emb, phys_emb, demand_emb

    @staticmethod
    def merge_edges(phys_emb: torch.Tensor, demand_emb: torch.Tensor) -> torch.Tensor:
        """Re-join the two edge types for consumers that need a single indexable
        tensor (the actor gathers candidate arcs out of it).

        One ``torch.cat`` per forward instead of one per layer: ``CatBackward``
        hands each input back a *view* of the output gradient, so unlike a slice
        it does not memset a full-size buffer.
        """
        if phys_emb.size(0) == 0:
            return demand_emb
        if demand_emb.size(0) == 0:
            return phys_emb
        return torch.cat([phys_emb, demand_emb], dim=0)

    def forward(self, tensors: GraphTensors) -> tuple[torch.Tensor, torch.Tensor]:
        node_emb = self.node_proj(tensors.node_features)
        num_phys = int(tensors.num_physical_directed)
        phys_emb, demand_emb = self.project_edges(tensors.edge_features_directed, num_phys)
        node_emb, phys_emb, demand_emb = self.run_layers(
            node_emb, tensors.edge_index, phys_emb, demand_emb, num_phys
        )
        return node_emb, self.merge_edges(phys_emb, demand_emb)


class DemandResidual(nn.Module):
    """Per-arc demand-conditioned residual on the base edge score.

    Adds ``Δs_a = MLP(CrossAttn(Q=arc, K/V=demand))`` on top of the existing
    ``edge_scorer`` output, so the base scorer (and the BC warm start that
    trained it) is left untouched:

        s_a = edge_scorer(pair_emb_a) + Δs_a

    Attention direction is **arc as Query, demand as Key/Value**. The reverse
    (demand as Query) is the textbook formulation, but it yields one output
    *per demand* -- and the sampler needs one score *per arc*. Querying from
    the arc gives the required ``(n_arcs, hidden)`` shape directly, with no
    scatter step.

    The final residual layer is zero-initialised, so at step 0 ``Δs_a == 0``
    and the forward pass is bit-identical to the base model. That is what
    makes this a single-variable experiment and keeps the BC warm start valid.

    A learnable scalar gate (``s = base + α·Δ``, ``α=0``) is deliberately NOT
    used: with ``α = 0`` the residual branch receives ``∂L/∂θ ∝ α = 0`` and
    cannot start learning. Zeroing only the last layer keeps the branch's own
    parameters in the graph while still producing zero output.
    """

    def __init__(
        self,
        arc_dim: int,
        hidden_dim: int,
        num_heads: int,
        activation: str,
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"demand_residual.hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )
        self.arc_proj = nn.Linear(arc_dim, hidden_dim)
        self.demand_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = build_mlp(hidden_dim, [hidden_dim], 1, activation, dropout)
        # Zero the last layer so the residual contributes exactly 0 at init.
        last = self.out[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        pair_emb: torch.Tensor,
        demand_emb: torch.Tensor,
        demand_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``pair_emb`` (n_arcs, arc_dim); ``demand_emb`` (n_demand, hidden).

        Returns ``(n_arcs,)``. An empty demand set (or all-masked demand)
        yields exactly 0 -- not NaN -- so slots with no pending request leave
        the base score unchanged.
        """
        if demand_emb.size(0) == 0 or pair_emb.size(0) == 0:
            return pair_emb.new_zeros((pair_emb.size(0),))
        if demand_valid is not None and not bool((demand_valid > 0).any()):
            return pair_emb.new_zeros((pair_emb.size(0),))

        h = self.hidden_dim
        nh = self.num_heads
        dh = h // nh

        q = self.arc_proj(pair_emb).view(-1, nh, dh)              # (n_arcs, nh, dh)
        k = self.demand_proj(demand_emb).view(-1, nh, dh)         # (n_dem, nh, dh)
        v = self.value_proj(demand_emb).view(-1, nh, dh)

        # (n_arcs, nh, n_dem): each arc attends over the demand tokens.
        logits = torch.einsum("ahd,thd->aht", q, k) / math.sqrt(dh)
        if demand_valid is not None:
            # (n_dem,) bool; -inf on invalid tokens so softmax ignores them.
            mask = demand_valid.view(1, 1, -1).to(logits.dtype)
            logits = logits.masked_fill(mask <= 0, float("-inf"))
            # The all-masked case was handled before the softmax.
        attn = torch.softmax(logits, dim=-1)
        ctx = torch.einsum("aht,thd->ahd", attn, v).reshape(-1, h)  # (n_arcs, h)
        return self.out(ctx).squeeze(-1)


class PathEdgeResidual(nn.Module):
    """Score legal arcs by attending to GNN tokens on current request paths."""

    def __init__(self, arc_dim: int, hidden_dim: int, num_heads: int, activation: str, dropout: float):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if hidden_dim % self.num_heads != 0:
            raise ValueError("path_edge_attention hidden_dim must be divisible by num_heads")
        self.arc_proj = nn.Linear(arc_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = build_mlp(hidden_dim, [hidden_dim], 1, activation, dropout)
        last = self.out[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, pair_emb: torch.Tensor, physical_emb: torch.Tensor | None,
                path_mask: torch.Tensor | None) -> torch.Tensor:
        if (pair_emb.size(0) == 0 or physical_emb is None
                or physical_emb.size(0) == 0 or path_mask is None):
            return pair_emb.new_zeros((pair_emb.size(0),))
        selected = physical_emb[path_mask.to(device=physical_emb.device, dtype=torch.bool)]
        if selected.size(0) == 0:
            return pair_emb.new_zeros((pair_emb.size(0),))
        nh, dh = self.num_heads, self.hidden_dim // self.num_heads
        q = self.arc_proj(pair_emb).view(-1, nh, dh)
        k = self.key_proj(selected).view(-1, nh, dh)
        v = self.value_proj(selected).view(-1, nh, dh)
        attn = torch.softmax(torch.einsum("ahd,thd->aht", q, k) / math.sqrt(dh), dim=-1)
        ctx = torch.einsum("aht,thd->ahd", attn, v).reshape(-1, self.hidden_dim)
        return self.out(ctx).squeeze(-1)


class PairPathAttentionScorer(nn.Module):
    """Shared per-GS-pair attention over bounded-hop path edge tokens.

    Path tokens explicitly contain both endpoint identity vectors and the edge
    embedding. Stock-only edges supply context; only legal generation arcs get
    logits. Parameters are shared across pairs rather than one module per pair.
    """

    def __init__(self, hidden_dim: int, num_heads: int, activation: str, dropout: float,
                 arc_query: bool = False, logit_temperature: float = 1.0):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("pair_path_attention hidden_dim must be divisible by num_heads")
        if logit_temperature <= 0:
            raise ValueError("pair_path_attention logit_temperature must be positive")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.logit_temperature = float(logit_temperature)
        # ★ arc_query: query 由 **候选弧自身** 与其所属 pair 的 demand token 共同
        #   构成，而不是只用 pair token。原因（实测，见 2026-09-24 诊断）：
        #   用 pair 做 query 时 context 与 demand 都按 pair 取再广播给该 pair 的
        #   所有弧 ⟹ 同 pair 内弧与弧的差异只剩 arc_emb（而 arc_emb 在 BC 里
        #   已收敛、梯度极小）。后果是注意力**结构上无法给同一站对内的候选弧
        #   排序**，而匹配采样器要的正是这个排序。实测 within-pair 方差仅占
        #   15%，消融 attention context 对分数的影响只有 arc_emb 的 1/2400。
        self.arc_query = arc_query
        # arc_emb 是 3*hidden（两个端点 node 向量 + 物理边向量），
        # 所以 arc-conditioned query 的输入是 3H + H = 4H。
        # arc_emb 与 physical_tokens 都是 3H（两端点 node + 边向量），
        # 加上 demand 的 H ⟹ arc-conditioned query 输入为 3H+3H+H = 7H。
        q_in = 7 * hidden_dim if arc_query else hidden_dim
        self.query = nn.Linear(q_in, hidden_dim)
        self.key = nn.Linear(3 * hidden_dim, hidden_dim)
        self.value = nn.Linear(3 * hidden_dim, hidden_dim)
        # A line-graph attention pass makes edge connectivity explicit: two
        # physical edges may exchange information exactly when they share a
        # node. Endpoint IDs are equality labels, never ordered magnitudes.
        self.link_query = nn.Linear(3 * hidden_dim, hidden_dim)
        self.link_key = nn.Linear(3 * hidden_dim, hidden_dim)
        self.link_value = nn.Linear(3 * hidden_dim, hidden_dim)
        self.link_out = nn.Linear(hidden_dim, 3 * hidden_dim)
        nn.init.normal_(self.link_out.weight, std=0.01)
        nn.init.zeros_(self.link_out.bias)
        self.out = build_mlp(5 * hidden_dim, [hidden_dim, hidden_dim], 1, activation, dropout)

    def forward(
        self, arc_emb: torch.Tensor, arc_physical_pos: torch.Tensor,
        physical_tokens: torch.Tensor, demand_emb: torch.Tensor,
        pair_masks: list[list[bool]], pair_weights: list[float],
        physical_endpoints: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_arcs = arc_emb.size(0)
        result = arc_emb.new_zeros((n_arcs,))
        if not n_arcs:
            return result
        if not pair_masks or not physical_tokens.size(0):
            zeros = arc_emb.new_zeros((n_arcs, 2 * self.hidden_dim))
            return self.out(torch.cat([arc_emb, zeros], dim=-1)).squeeze(-1)
        if len(pair_masks) != demand_emb.size(0):
            raise ValueError("pair path masks must align with demand embeddings")
        mask = torch.as_tensor(pair_masks, dtype=torch.bool, device=arc_emb.device)
        weights = torch.as_tensor(pair_weights, dtype=arc_emb.dtype, device=arc_emb.device)
        if mask.shape != (demand_emb.size(0), physical_tokens.size(0)):
            raise ValueError("pair path mask must align with physical edges")
        if weights.shape != (demand_emb.size(0),):
            raise ValueError("pair path weights must align with demand embeddings")

        # Every positive-weight pair with a bounded path gets its own query.
        # Edges outside all paths use the weighted aggregate pair context.
        candidate_mask = mask[:, arc_physical_pos] & (weights[:, None] > 0)
        pair_idx, arc_idx = candidate_mask.nonzero(as_tuple=True)
        active_pairs = torch.nonzero((weights > 0) & mask.any(dim=1)).flatten()
        if active_pairs.numel() == 0:
            zeros = arc_emb.new_zeros((n_arcs, 2 * self.hidden_dim))
            return self.out(torch.cat([arc_emb, zeros], dim=-1)).squeeze(-1)
        active_mask = mask[active_pairs]
        nh, dh = self.num_heads, self.hidden_dim // self.num_heads
        if physical_endpoints is not None:
            if physical_endpoints.shape != (physical_tokens.size(0), 2):
                raise ValueError("physical endpoints must align with physical tokens")
            shared = (
                (physical_endpoints[:, None, 0] == physical_endpoints[None, :, 0])
                | (physical_endpoints[:, None, 0] == physical_endpoints[None, :, 1])
                | (physical_endpoints[:, None, 1] == physical_endpoints[None, :, 0])
                | (physical_endpoints[:, None, 1] == physical_endpoints[None, :, 1])
            )
            link_q = self.link_query(physical_tokens).view(-1, nh, dh)
            link_k = self.link_key(physical_tokens).view(-1, nh, dh)
            link_v = self.link_value(physical_tokens).view(-1, nh, dh)
            link_logits = torch.einsum("ehd,fhd->ehf", link_q, link_k) / math.sqrt(dh)
            link_logits = link_logits.masked_fill(~shared[:, None, :], float("-inf"))
            link_attn = torch.softmax(link_logits, dim=-1)
            link_ctx = torch.einsum("ehf,fhd->ehd", link_attn, link_v).reshape(-1, self.hidden_dim)
            physical_tokens = physical_tokens + self.link_out(link_ctx)
        if self.arc_query:
            # 每个 (弧, pair) 候选对一条独立 query，含该弧自己的边向量。
            # 只拼 [端点向量 | 边向量 | 需求] 不够：arc_emb 的 edge 段在旧 BC
            # 里已被冻住，query 会退化成对同 pair 内所有弧几乎相同的内容
            # （实测逐弧离散度仅 2.3e-4）。把物理边向量再单独拼一次并给足
            # 权重，让 query 真正逐弧可分辨。
            q = self.query(torch.cat([
                arc_emb[arc_idx], physical_tokens[arc_physical_pos[arc_idx]],
                demand_emb[pair_idx],
            ], dim=-1))
        else:
            q = self.query(demand_emb[active_pairs])
        q = q.view(-1, nh, dh)
        k = self.key(physical_tokens).view(-1, nh, dh)
        v = self.value(physical_tokens).view(-1, nh, dh)
        if self.arc_query:
            # q 的第 i 行对应 (pair_idx[i], arc_idx[i]) ⟹ 同 pair 内的不同弧
            # 得到**不同**的 context，注意力终于能在站对内部给候选弧排序
            # （这正是匹配采样器需要的量）。
            logits = torch.einsum("ihd,ehd->ihe", q, k) / (math.sqrt(dh) * self.logit_temperature)
            logits = logits.masked_fill(~mask[pair_idx][:, None, :], float("-inf"))
        else:
            logits = torch.einsum("phd,ehd->phe", q, k) / (math.sqrt(dh) * self.logit_temperature)
            logits = logits.masked_fill(~active_mask[:, None, :], float("-inf"))
        attention = torch.softmax(logits, dim=-1)
        context = torch.einsum("ihe,ehd->ihd", attention, v).reshape(-1, self.hidden_dim)
        norm = result.new_zeros((n_arcs,))
        if pair_idx.numel():
            if self.arc_query:
                # context 与 pair_idx/arc_idx 同行对齐，无需再按 pair 索引
                ctx_rows = context
            else:
                local = torch.searchsorted(active_pairs, pair_idx)
                ctx_rows = context[local]
            paired = torch.cat([arc_emb[arc_idx], ctx_rows, demand_emb[pair_idx]], dim=-1)
            contribution = self.out(paired).squeeze(-1) * weights[pair_idx]
            result = result.index_add(0, arc_idx, contribution)
            norm = norm.index_add(0, arc_idx, weights[pair_idx])
        missing = norm == 0
        if missing.any():
            pair_weight = weights[active_pairs].clamp_min(0)
            pair_weight = pair_weight / pair_weight.sum().clamp_min(1e-6)
            n_missing = int(missing.sum())
            if self.arc_query and context.size(0):
                pooled_context = context.mean(dim=0, keepdim=True).expand(n_missing, -1)
            elif self.arc_query:
                pooled_context = arc_emb.new_zeros((n_missing, self.hidden_dim))
            else:
                pooled_context = (context * pair_weight[:, None]).sum(dim=0).expand(
                    n_missing, -1)
            pooled_demand = (demand_emb[active_pairs] * pair_weight[:, None]).sum(dim=0)
            fallback = torch.cat([
                arc_emb[missing],
                pooled_context,
                pooled_demand.expand(n_missing, -1),
            ], dim=-1)
            result = result.index_add(0, missing.nonzero().flatten(),
                                      self.out(fallback).squeeze(-1))
        # ★ 归一化口径修正：原实现返回 result / norm.clamp_min(1.0)。实测权重和
        #   p50=0.0346 且 100% < 1 ⟹ clamp_min(1.0) 把**所有**分数整体压到 0.059
        #   倍，且该缩放依弧而变（覆盖该弧的权重和不同）。改为除以真实权重和，
        #   于是贡献是凸组合、尺度不塌。
        # Arcs outside every active pair path already received a complete
        # fallback score above. Their norm is zero; dividing that score by
        # 1e-6 amplified it a millionfold and effectively banned off-path
        # preparation edges from the matching policy.
        return torch.where(missing, result, result / norm.clamp_min(1e-6))


class PreparationHead(nn.Module):
    """Stage-two score and STOP, registered after the warm-start actor modules."""

    def __init__(self, hidden_dim: int, activation: str):
        super().__init__()
        self.stop_logit = nn.Parameter(torch.zeros(()))
        self.value_scale = nn.Parameter(torch.ones(()))
        self.scorer = build_mlp(6 * hidden_dim + 2, [hidden_dim], 1, activation, 0.0)


class SharedNodeActor(nn.Module):
    def __init__(self, hidden_dim: int, config: dict, invalid_logit_value: float):
        super().__init__()
        # ★ 本类收到的是**整个 `model` 段**（`graph_mappo.py` 构造处传
        #   `model_cfg`），而 `activation`/`dropout` 住在 `model.encoder` 下
        #   （`configs/graph_mappo.yaml:14-15`）⟹ 从顶层读**永远取不到**，
        #   静默回落 relu/0.0：yaml 那两行看起来在管策略头，实际从未生效。
        #   与 `self.mode`（确实在 `model` 顶层）和 `GlobalCritic.pooling`
        #   （正确地读了 `config["critic"]`）并排看，层级错位一目了然。
        enc_cfg = config.get("encoder", {})
        activation = enc_cfg.get("activation", "relu")
        dropout = float(enc_cfg.get("dropout", 0.0))
        self.mode = config.get("mode", "mixed")
        # Both modes score an edge from its two endpoint node embeddings plus
        # the physical edge embedding. In demand_edge mode the encoder still
        # keeps physical-link messages out of the node representation, so the
        # node embeddings carry the dynamic demand signal without being
        # polluted by raw link rates; the actor can therefore see pending
        # demand even on slots where relay_importance is empty.
        edge_scorer_input_dim = hidden_dim * 3
        self.attention_only = bool(config["actor"].get("attention_only", False))
        self.edge_scorer = None if self.attention_only else build_mlp(
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
        # Opt-in per-arc demand-conditioned residual (see ``DemandResidual``).
        # Disabled by default so every existing arm stays bit-identical; when
        # enabled its last layer is zero-initialised, so the first forward pass
        # still equals the base model and the BC warm start remains valid.
        self.demand_residual = None
        dr_cfg = config["actor"].get("demand_residual", {}) or {}
        if dr_cfg.get("enabled", False) and not self.attention_only:
            self.demand_residual = DemandResidual(
                arc_dim=edge_scorer_input_dim,
                hidden_dim=hidden_dim,
                num_heads=int(dr_cfg.get("num_heads", 4)),
                activation=activation,
                dropout=dropout,
            )
        self.path_edge_residual = None
        path_cfg = config["actor"].get("path_edge_attention", {}) or {}
        if path_cfg.get("enabled", False) and not self.attention_only:
            self.path_edge_residual = PathEdgeResidual(
                arc_dim=edge_scorer_input_dim,
                hidden_dim=hidden_dim,
                num_heads=int(path_cfg.get("num_heads", 4)),
                activation=activation,
                dropout=dropout,
            )
        self.pair_path_scorer = None
        pair_cfg = config["actor"].get("pair_path_attention", {}) or {}
        if pair_cfg.get("enabled", False):
            self.pair_path_scorer = PairPathAttentionScorer(
                hidden_dim=hidden_dim,
                num_heads=int(pair_cfg.get("num_heads", 4)),
                activation=activation,
                dropout=dropout,
                arc_query=bool(pair_cfg.get("arc_query", False)),
                logit_temperature=float(pair_cfg.get("logit_temperature", 1.0)),
            )
        if self.attention_only and self.pair_path_scorer is None:
            raise ValueError("attention_only requires pair_path_attention.enabled")
        self.invalid_logit_value = invalid_logit_value
        self.temperature = float(config["actor"].get("temperature", 1.0))
        # Learnable STOP logit for the sequential global matching sampler: at
        # every decision the policy chooses among the still-feasible arcs and
        # STOP, so it can express "activate nothing this slot" instead of
        # being forced to fill every feasible port.
        self.stop_logit = nn.Parameter(torch.zeros(()))
        self.two_stage_enabled = bool(config["actor"].get("two_stage", {}).get("enabled", False))
        self.two_stage_phase = str(config["actor"].get("two_stage", {}).get("phase", "joint"))
        if self.two_stage_enabled and self.two_stage_phase not in {"service_only", "prepare_only", "joint"}:
            raise ValueError(f"invalid v3.1 training phase: {self.two_stage_phase}")
        self.prepare = PreparationHead(hidden_dim, activation) if self.two_stage_enabled else None
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
        demand_emb: torch.Tensor | None = None,
        physical_emb: torch.Tensor | None = None,
        path_edge_mask: torch.Tensor | None = None,
        physical_path_tokens: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[str], list[int], dict[str, torch.Tensor]]:
        # A cached observation can be evaluated by a second model (e.g. a
        # baseline/attention comparison). Its maps are model-local, not part
        # of the observation's candidate-plan cache.
        self._ensure_static(action_space)
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
            edge_scores = (torch.zeros(pair_emb.size(0), device=device)
                           if self.attention_only else self.edge_scorer(pair_emb).squeeze(-1))
            if self.demand_residual is not None:
                edge_scores = edge_scores + self.demand_residual(pair_emb, demand_emb)
            if self.path_edge_residual is not None:
                edge_scores = edge_scores + self.path_edge_residual(
                    pair_emb, physical_emb, path_edge_mask
                )
            if self.pair_path_scorer is not None:
                if physical_path_tokens is None or demand_emb is None:
                    raise ValueError("pair path attention requires physical and demand tokens")
                attention_scores = self.pair_path_scorer(
                    pair_emb, pos_t // 2, physical_path_tokens, demand_emb,
                    obs.pair_path_edge_masks, obs.pair_path_weights,
                    torch.as_tensor(obs.edge_index[:len(obs.physical_edge_ids) * 2:2],
                                    dtype=torch.long, device=device),
                )
                edge_scores = attention_scores if self.attention_only else edge_scores + attention_scores
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
        edge_map = _edge_score_map(plan, self._node_idx, edge_scores)
        arc_embeddings = pair_emb if edge_srcs.size else node_emb.new_zeros((0, 3 * node_emb.size(-1)))
        return logits, masked, list(obs.node_ids), lengths, edge_map, arc_embeddings


class GlobalCritic(nn.Module):
    def __init__(self, hidden_dim: int, config: dict):
        super().__init__()
        # 同 `SharedNodeActor`：`config` 是**整个 `model` 段**，
        # 激活/dropout 须从 `model.encoder` 取，不能从顶层取。
        # 注意本类紧邻的下一行就读了 `config["critic"]` —— 两种层级
        # 写法并排，正是这处错位的证据。
        enc_cfg = config.get("encoder", {})
        activation = enc_cfg.get("activation", "relu")
        dropout = float(enc_cfg.get("dropout", 0.0))
        self.pooling = config["critic"].get("pooling", "mean")
        # Opt-in: append a per-type MAX pool next to the mean pools. Default
        # False leaves every existing arm's input width (and therefore its
        # behavior) bit-identical.
        self.pool_include_max = bool(config["critic"].get("pool_include_max", False))
        if self.pooling == "typed_mean":
            # 节点平均 + 物理边平均 + 逻辑边平均 + 3 个图规模计数。
            value_input_dim = hidden_dim * 3 + 3
        elif self.pooling == "mean":
            value_input_dim = hidden_dim * 2
        else:
            raise NotImplementedError(f"Unsupported critic pooling: {self.pooling}")
        if self.pool_include_max:
            value_input_dim += hidden_dim * (3 if self.pooling == "typed_mean" else 1)
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
            if self.pool_include_max:
                node_mx = node_emb.max(dim=0).values
                phys_mx = (edge_emb_directed[:num_physical_directed].max(dim=0).values
                           if num_physical_directed > 0 else torch.zeros(hidden, device=device))
                dem_mx = (edge_emb_directed[num_physical_directed:].max(dim=0).values
                          if num_demand_directed > 0 else torch.zeros(hidden, device=device))
                graph_emb = torch.cat([graph_emb, node_mx, phys_mx, dem_mx], dim=-1)
        else:
            if edge_emb_directed.shape[0] == 0:
                # Mask-first filtering may leave the graph with no physical links.
                edge_pool = torch.zeros(hidden, device=device)
            else:
                edge_pool = edge_emb_directed.mean(dim=0)
            graph_emb = torch.cat([node_pool, edge_pool], dim=-1)
            if self.pool_include_max:
                node_mx = node_emb.max(dim=0).values
                edge_mx = (edge_emb_directed.max(dim=0).values
                           if edge_emb_directed.shape[0] > 0 else torch.zeros(hidden, device=device))
                graph_emb = torch.cat([graph_emb, node_mx, edge_mx], dim=-1)
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
            fuse_demand_to_node=bool(model_cfg["encoder"].get("fuse_demand_to_node", True)),
        )
        self.actor = SharedNodeActor(hidden_dim, model_cfg, invalid_logit_value)
        n_nodes = len(action_space.node_ids)
        if n_nodes <= hidden_dim:
            identity = torch.eye(n_nodes, hidden_dim)
        else:
            generator = torch.Generator().manual_seed(31)
            identity = torch.randn(n_nodes, hidden_dim, generator=generator)
            identity = torch.nn.functional.normalize(identity, dim=-1)
        self.register_buffer("endpoint_identity", identity, persistent=False)
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
                # edge_h 只在 h_rows 非空时拼接（见下方第二个调用点的说明）。
                if h_rows:
                    edge_features_directed = torch.cat(
                        [edge_features_directed, torch.cat(h_rows, dim=0)], dim=-1
                    )
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
        # Demand tokens are the tail of the merged edge tensor (physical edges
        # come first, see ``GraphEncoder.merge_edges``). Only needed by the
        # optional demand residual; slicing costs nothing when it is off.
        demand_emb = None
        if self.actor.demand_residual is not None or self.actor.pair_path_scorer is not None:
            n_phys = int(tensors.num_physical_directed)
            # The GNN keeps both directions for message passing, but a GS
            # pair is one demand unit for attention. Pool its two directions.
            directed = edge_emb[n_phys:]
            demand_emb = directed.reshape(-1, 2, directed.size(-1)).mean(dim=1)
        physical_emb = None
        path_edge_mask = None
        if self.actor.path_edge_residual is not None:
            n_phys = int(tensors.num_physical_directed)
            physical_emb = edge_emb[:n_phys:2]
            mask = obs.request_path_edge_mask or [False] * len(obs.physical_edge_ids)
            path_edge_mask = torch.as_tensor(mask, dtype=torch.bool, device=edge_emb.device)
        physical_path_tokens = None
        if self.actor.pair_path_scorer is not None:
            n_phys = int(tensors.num_physical_directed)
            endpoints = tensors.edge_index[:, :n_phys:2]
            physical_path_tokens = torch.cat([
                self.endpoint_identity[endpoints[0]], self.endpoint_identity[endpoints[1]], edge_emb[:n_phys:2]
            ], dim=-1)
        logits, logits_padded, node_order, lengths, edge_scores, arc_embeddings = self.actor(
            obs, node_emb, edge_emb, self.action_space,
            build_logits_dict=build_logits_dict,
            demand_emb=demand_emb,
            physical_emb=physical_emb,
            path_edge_mask=path_edge_mask,
            physical_path_tokens=physical_path_tokens,
        )
        return ActorCriticOutput(
            logits=logits,
            value=self.critic(node_emb, edge_emb, tensors.num_physical_directed),
            logits_padded=logits_padded,
            logits_node_order=node_order,
            logits_lengths=lengths,
            edge_scores=edge_scores,
            arc_embeddings=arc_embeddings,
        )

    def batched_forward(
        self,
        obs_list: list[GraphObservation],
        device: torch.device | str = "cpu",
        want_edge_maps: bool = True,
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
                # ⚠ history_dim == 0 时**绝不能拼**：h_rows 为空 ⟹ edge_h 是 0 行
                # 空张量，而 `torch.cat` 要求除 dim=1 外尺寸一致 ⟹ RuntimeError。
                # 这个组合（只开节点历史、不开边历史）此前**从未被跑过**，是潜伏 bug：
                # features.yaml 里 demand_edge 显式 true 且 node 全 false，
                # 所以以前打开 enabled 总是顺带把 history_dim 抬到 64，永远走不到这里。
                if h_rows:
                    edge_features_directed = torch.cat(
                        [edge_features_directed, torch.cat(h_rows, dim=0)], dim=-1
                    )
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
        # Offset of each graph's demand block inside `demand_emb`. `perm` lays
        # out all physical edges first (graph by graph) then all demand edges
        # (also graph by graph, same order), so the demand blocks are
        # contiguous and start at the running sum of `n_demand`.
        demand_off: list[int] = []
        dem_acc = 0
        for n in n_demand:
            demand_off.append(dem_acc)
            dem_acc += n
        edge_index_p = edge_index_all[:, perm]
        edge_features_p = edge_features_all[perm]

        node_emb = self.encoder.node_proj(node_features_all)
        phys_emb, demand_emb = self.encoder.project_edges(edge_features_p, num_phys_total)
        node_emb, phys_emb, demand_emb = self.encoder.run_layers(
            node_emb, edge_index_p, phys_emb, demand_emb, num_phys_total
        )
        edge_emb = self.encoder.merge_edges(phys_emb, demand_emb)

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
            edge_scores = (torch.zeros(pair_emb.size(0), device=device)
                           if self.actor.attention_only else self.actor.edge_scorer(pair_emb).squeeze(-1))
            if self.actor.demand_residual is not None:
                # Per-graph blocks: an arc must only attend over its OWN graph's
                # demand tokens. `pair_emb` and `demand_emb` are both laid out
                # graph-by-graph, so slicing by the recorded offsets keeps the
                # two aligned without materialising a block-diagonal mask.
                resid_parts: list[torch.Tensor] = []
                for i in range(len(plans)):
                    n_arc_i = int(plans[i][0].size)
                    if n_arc_i == 0:
                        continue
                    arc_lo, arc_hi = edge_score_off[i], edge_score_off[i] + n_arc_i
                    dem_lo = demand_off[i]
                    dem_hi = dem_lo + n_demand[i]
                    directed = demand_emb[dem_lo:dem_hi]
                    resid_parts.append(
                        self.actor.demand_residual(
                            pair_emb[arc_lo:arc_hi],
                            directed.reshape(-1, 2, directed.size(-1)).mean(dim=1),
                        )
                    )
                if resid_parts:
                    edge_scores = edge_scores + torch.cat(resid_parts, dim=0)
            if self.actor.path_edge_residual is not None:
                resid_parts = []
                phys_off = 0
                for i in range(len(plans)):
                    n_arc_i = int(plans[i][0].size)
                    n_phys_i = n_phys[i]
                    graph_phys = phys_emb[phys_off:phys_off + n_phys_i:2]
                    phys_off += n_phys_i
                    if n_arc_i == 0:
                        continue
                    arc_lo, arc_hi = edge_score_off[i], edge_score_off[i] + n_arc_i
                    mask = obs_list[i].request_path_edge_mask or [False] * len(obs_list[i].physical_edge_ids)
                    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
                    resid_parts.append(self.actor.path_edge_residual(
                        pair_emb[arc_lo:arc_hi], graph_phys, mask_t
                    ))
                if resid_parts:
                    edge_scores = edge_scores + torch.cat(resid_parts, dim=0)
            if self.actor.pair_path_scorer is not None:
                resid_parts = []
                phys_off = 0
                for i, plan in enumerate(plans):
                    n_arc_i = int(plan[0].size)
                    n_phys_i = n_phys[i]
                    graph_phys = phys_emb[phys_off:phys_off + n_phys_i:2]
                    phys_off += n_phys_i
                    if n_arc_i == 0:
                        continue
                    arc_lo = edge_score_off[i]
                    arc_hi = arc_lo + n_arc_i
                    dem_lo = demand_off[i]
                    dem_hi = dem_lo + n_demand[i]
                    graph_demand = demand_emb[dem_lo:dem_hi]
                    graph_demand = graph_demand.reshape(-1, 2, graph_demand.size(-1)).mean(dim=1)
                    endpoints = torch.as_tensor(
                        obs_list[i].edge_index[:n_phys_i:2], dtype=torch.long, device=device
                    )
                    if endpoints.numel():
                        path_tokens = torch.cat([
                            self.endpoint_identity[endpoints[:, 0]],
                            self.endpoint_identity[endpoints[:, 1]],
                            graph_phys,
                        ], dim=-1)
                    else:
                        path_tokens = graph_phys.new_zeros((0, 3 * graph_phys.size(-1)))
                    arc_pos = torch.as_tensor(plan[2] // 2, dtype=torch.long, device=device)
                    resid_parts.append(self.actor.pair_path_scorer(
                        pair_emb[arc_lo:arc_hi], arc_pos, path_tokens, graph_demand,
                        obs_list[i].pair_path_edge_masks, obs_list[i].pair_path_weights,
                        endpoints,
                    ))
                if resid_parts:
                    attention_scores = torch.cat(resid_parts, dim=0)
                    edge_scores = (attention_scores if self.actor.attention_only
                                   else edge_scores + attention_scores)
        else:
            edge_scores = torch.zeros((0,), dtype=torch.float32, device=device)
        if idle_parts:
            idle_t = torch.from_numpy(np.concatenate(idle_parts)).to(device)
            idle_scores = self.actor.idle_scorer(node_emb[idle_t]).squeeze(-1)
        else:
            idle_scores = torch.zeros((0,), dtype=torch.float32, device=device)

        # Per-graph edge_id -> raw score maps for the priority-matching
        # resolver (see _edge_score_map), plus the raw index arrays the matching
        # sampler consumes directly.
        edge_score_maps: list[dict[str, torch.Tensor]] = []
        edge_arrays: list[tuple[np.ndarray, np.ndarray, torch.Tensor]] = []
        arc_embeddings: list[torch.Tensor] = []
        for i, plan in enumerate(plans):
            n_edge_i = int(plan[0].size)
            graph_scores = (
                edge_scores[edge_score_off[i] : edge_score_off[i] + n_edge_i]
                if n_edge_i
                else edge_scores.new_zeros((0,))
            )
            # The arc dict costs one Python iteration per legal candidate; skip
            # it when the caller samples from `edge_arrays` instead.
            if want_edge_maps:
                edge_score_maps.append(
                    _edge_score_map(
                        plan,
                        self.actor._node_idx,
                        graph_scores,
                    )
                )
            edge_arrays.append((plan[0], plan[1], graph_scores))
            arc_embeddings.append(pair_emb[edge_score_off[i]:edge_score_off[i] + n_edge_i])

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
        hidden = node_emb.size(1)
        pooling = self.critic.pooling
        value_inputs: list[torch.Tensor] = []
        n_graphs = len(tensors_list)
        n_nodes = [int(t.node_features.size(0)) for t in tensors_list]
        # Segment means instead of per-graph slices -- see _segment_sum. An empty
        # type yields an all-zero row, which is exactly the placeholder the
        # previous code built by hand.
        node_sum, node_cnt = _segment_sum(node_emb, _segment_ids(n_nodes, device), n_graphs)
        phys_sum, phys_cnt = _segment_sum(phys_emb, _segment_ids(n_phys, device), n_graphs)
        dem_sum, dem_cnt = _segment_sum(demand_emb, _segment_ids(n_demand, device), n_graphs)
        include_max = self.critic.pool_include_max
        if include_max:
            node_mx_all = _segment_max(node_emb, _segment_ids(n_nodes, device), n_graphs)
            phys_mx_all = _segment_max(phys_emb, _segment_ids(n_phys, device), n_graphs)
            dem_mx_all = _segment_max(demand_emb, _segment_ids(n_demand, device), n_graphs)
        for i in range(len(tensors_list)):
            node_pool = node_sum[i] / node_cnt[i].clamp(min=1.0)
            if pooling == "typed_mean":
                physical_pool = phys_sum[i] / phys_cnt[i].clamp(min=1.0)
                demand_pool = dem_sum[i] / dem_cnt[i].clamp(min=1.0)
                scale_counts = torch.log1p(
                    torch.tensor(
                        [n_nodes[i], n_phys[i] // 2, n_demand[i] // 2],
                        dtype=torch.float32,
                        device=edge_emb.device,
                    )
                )
                parts = [node_pool, physical_pool, demand_pool, scale_counts]
                if include_max:
                    parts += [node_mx_all[i], phys_mx_all[i], dem_mx_all[i]]
                value_inputs.append(torch.cat(parts, dim=-1))
            else:
                n_edge_i = n_phys[i] + n_demand[i]
                edge_pool = (phys_sum[i] + dem_sum[i]) / max(n_edge_i, 1)
                parts = [node_pool, edge_pool]
                if include_max:
                    n_edge_all = n_phys[i] + n_demand[i]
                    if n_edge_all > 0:
                        # Combined max over physical+demand rows for this graph.
                        edge_mx = torch.maximum(phys_mx_all[i], dem_mx_all[i])
                    else:
                        edge_mx = torch.zeros(hidden, device=edge_emb.device)
                    parts += [node_mx_all[i], edge_mx]
                value_inputs.append(torch.cat(parts, dim=-1))
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
            edge_arrays=edge_arrays,
            arc_embeddings=arc_embeddings,
        )


def _nodes_from_edge_id(edge_id: str) -> tuple[str, str]:
    edge_name = edge_id[2:] if edge_id.startswith("E_") else edge_id
    if "__" not in edge_name:
        raise ValueError(f"Cannot infer edge endpoints from edge id {edge_id!r}.")
    return edge_name.split("__", 1)
