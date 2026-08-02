from __future__ import annotations

import torch
from torch import nn


def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Sequential:
    activation_cls = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
    dims = [input_dim] + list(hidden_dims)
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation_cls())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor, invalid_value: float) -> torch.Tensor:
    return torch.where(mask.bool(), logits, torch.full_like(logits, invalid_value))

