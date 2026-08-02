from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.graph_builder import GraphObservation
from qkd_rl.models.mlp import build_mlp, masked_logits


@dataclass
class GraphTensors:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features_directed: torch.Tensor
    node_ids: list[str]
    edge_ids: list[str]


@dataclass
class ActorCriticOutput:
    logits: dict[str, torch.Tensor]
    value: torch.Tensor


def observation_to_tensors(obs: GraphObservation, device: torch.device | str = "cpu") -> GraphTensors:
    node_features = torch.tensor(obs.node_features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(obs.edge_index, dtype=torch.long, device=device).t().contiguous()
    edge_features = torch.tensor(obs.edge_features, dtype=torch.float32, device=device)
    edge_features_directed = edge_features.repeat_interleave(2, dim=0)
    return GraphTensors(
        node_features=node_features,
        edge_index=edge_index,
        edge_features_directed=edge_features_directed,
        node_ids=obs.node_ids,
        edge_ids=obs.edge_ids,
    )


class EdgeConditionedGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, activation: str, dropout: float, layer_norm: bool):
        super().__init__()
        self.message_mlp = build_mlp(
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

    def forward(self, node_emb: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        messages = self.message_mlp(torch.cat([node_emb[src], edge_attr], dim=-1))
        aggregated = torch.zeros_like(node_emb)
        aggregated.index_add_(0, dst, messages)
        degree = torch.zeros(node_emb.size(0), 1, dtype=node_emb.dtype, device=node_emb.device)
        degree.index_add_(0, dst, torch.ones(messages.size(0), 1, dtype=node_emb.dtype, device=node_emb.device))
        aggregated = aggregated / degree.clamp_min(1.0)
        updated = self.update(torch.cat([node_emb, aggregated], dim=-1))
        return self.norm(updated)


class GraphEncoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, config: dict):
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        activation = config.get("activation", "relu")
        dropout = float(config.get("dropout", 0.0))
        layer_norm = bool(config.get("layer_norm", True))
        self.residual = bool(config.get("residual", True))
        self.node_proj = build_mlp(node_dim, [hidden_dim], hidden_dim, activation, dropout)
        self.edge_proj = build_mlp(edge_dim, [hidden_dim], hidden_dim, activation, dropout)
        self.layers = nn.ModuleList(
            [
                EdgeConditionedGraphLayer(hidden_dim, hidden_dim, activation, dropout, layer_norm)
                for _ in range(int(config["num_layers"]))
            ]
        )

    def forward(self, tensors: GraphTensors) -> tuple[torch.Tensor, torch.Tensor]:
        node_emb = self.node_proj(tensors.node_features)
        edge_emb = self.edge_proj(tensors.edge_features_directed)
        for layer in self.layers:
            next_node_emb = layer(node_emb, tensors.edge_index, edge_emb)
            node_emb = node_emb + next_node_emb if self.residual else next_node_emb
        return node_emb, edge_emb


class SharedNodeActor(nn.Module):
    def __init__(self, hidden_dim: int, config: dict, invalid_logit_value: float):
        super().__init__()
        activation = config.get("activation", "relu")
        dropout = float(config.get("dropout", 0.0))
        self.edge_scorer = build_mlp(
            hidden_dim * 3,
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

    def forward(
        self,
        obs: GraphObservation,
        node_emb: torch.Tensor,
        edge_emb_directed: torch.Tensor,
        action_space: NodeActionSpace,
    ) -> dict[str, torch.Tensor]:
        node_index = {node_id: idx for idx, node_id in enumerate(obs.node_ids)}
        edge_index_by_pair: dict[tuple[str, str], int] = {}
        for edge_pos, edge_id in enumerate(obs.physical_edge_ids):
            src, dst = _nodes_from_edge_id(edge_id)
            edge_index_by_pair[tuple(sorted((src, dst)))] = edge_pos * 2

        logits: dict[str, torch.Tensor] = {}
        for node_id in obs.node_ids:
            scores: list[torch.Tensor] = []
            src_idx = node_index[node_id]
            for action in obs.action_candidates[node_id]:
                if action == NodeActionSpace.IDLE:
                    scores.append(self.idle_scorer(node_emb[src_idx]).squeeze(-1))
                    continue
                dst_idx = node_index[action]
                edge_id = action_space.action_to_edge(node_id, action)
                if edge_id is None:
                    scores.append(torch.tensor(self.invalid_logit_value, device=node_emb.device))
                    continue
                edge_pos = edge_index_by_pair[tuple(sorted((node_id, action)))]
                pair_emb = torch.cat([node_emb[src_idx], node_emb[dst_idx], edge_emb_directed[edge_pos]], dim=-1)
                scores.append(self.edge_scorer(pair_emb).squeeze(-1))
            raw_logits = torch.stack(scores)
            mask = torch.tensor(obs.action_masks[node_id], dtype=torch.bool, device=node_emb.device)
            logits[node_id] = masked_logits(raw_logits, mask, self.invalid_logit_value)
        return logits


class GlobalCritic(nn.Module):
    def __init__(self, hidden_dim: int, config: dict):
        super().__init__()
        activation = config.get("activation", "relu")
        dropout = float(config.get("dropout", 0.0))
        self.pooling = config["critic"].get("pooling", "mean")
        self.value_head = build_mlp(
            hidden_dim * 2,
            list(config["critic"]["hidden_dims"]),
            1,
            activation,
            dropout,
        )

    def forward(self, node_emb: torch.Tensor, edge_emb_directed: torch.Tensor) -> torch.Tensor:
        if self.pooling != "mean":
            raise NotImplementedError(f"Unsupported critic pooling: {self.pooling}")
        graph_emb = torch.cat([node_emb.mean(dim=0), edge_emb_directed.mean(dim=0)], dim=-1)
        return self.value_head(graph_emb).squeeze(-1)


class GraphMAPPOActorCritic(nn.Module):
    def __init__(self, action_space: NodeActionSpace, config: dict):
        super().__init__()
        model_cfg = config["model"]
        feature_dims = config["features"]["dims"]
        node_dim = int(feature_dims["node_dim_resolved"])
        edge_dim = int(feature_dims["edge_dim_resolved"])
        hidden_dim = int(model_cfg["encoder"]["hidden_dim"])
        invalid_logit_value = float(model_cfg["distribution"]["invalid_logit_value"])
        self.action_space = action_space
        self.encoder = GraphEncoder(node_dim, edge_dim, model_cfg["encoder"])
        self.actor = SharedNodeActor(hidden_dim, model_cfg, invalid_logit_value)
        self.critic = GlobalCritic(hidden_dim, model_cfg)

    def forward(self, obs: GraphObservation, device: torch.device | str = "cpu") -> ActorCriticOutput:
        tensors = observation_to_tensors(obs, device)
        node_emb, edge_emb = self.encoder(tensors)
        return ActorCriticOutput(
            logits=self.actor(obs, node_emb, edge_emb, self.action_space),
            value=self.critic(node_emb, edge_emb),
        )


def _nodes_from_edge_id(edge_id: str) -> tuple[str, str]:
    edge_name = edge_id[2:] if edge_id.startswith("E_") else edge_id
    if "__" not in edge_name:
        raise ValueError(f"Cannot infer edge endpoints from edge id {edge_id!r}.")
    return edge_name.split("__", 1)
