"""Shared per-entity history encoder (LSTM).

A single LSTM processes the history windows of every node, every physical link
and every GS-GS demand pair, so the LSTM parameters are shared across all
entities (``critic`` and ``actor`` use the same encoder object). Because the
three entity types have different channel counts, each type gets its own input
projection to the shared hidden dimension.

Sequences are left-padded on cold start: ``history_valid`` carries the actual
number of observed steps and is used to pack the LSTM input, so partially
filled windows are not treated as zeros inside the recurrence.
"""

from __future__ import annotations

import torch
from torch import nn

from qkd_rl.env.graph_builder import GraphObservation
from qkd_rl.models.mlp import build_mlp


def _channel_flags(cfg: dict, names: list[str], defaults: list[bool]) -> dict[str, bool]:
    return {name: bool(cfg.get(name, default)) for name, default in zip(names, defaults)}


class HistoryEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        he = config["features"]["history_encoder"]
        self.hidden_dim = int(he.get("hidden_dim", 128))
        self.num_layers = int(he.get("num_layers", 1))
        self.seq_len = int(he.get("seq_len", 240))

        node_cfg = he.get("node", {})
        phys_cfg = he.get("physical_edge", {})
        demand_cfg = he.get("demand_edge", {})
        self.node_channels = sum(
            bool(node_cfg.get(key, default))
            for key, default in (
                ("include_arrived", True),
                ("include_served", True),
                ("include_failed", True),
                ("include_qkp_total", True),
            )
        )
        self.phys_channels = sum(
            bool(phys_cfg.get(key, default))
            for key, default in (
                ("include_qkp_level", True),
                ("include_available", True),
                ("include_activated", False),
            )
        )
        self.demand_channels = (
            int(config["features"]["demand_edge"].get("wait_bucket_count", 10))
            if demand_cfg.get("include_pending_wait_buckets", False)
            else 0
        )

        # Per-type projections unify different channel counts to hidden_dim;
        # the LSTM itself is shared by all entity types.
        self.node_proj = (
            build_mlp(self.node_channels, [self.hidden_dim], self.hidden_dim, "relu", 0.0)
            if self.node_channels > 0
            else None
        )
        self.phys_proj = (
            build_mlp(self.phys_channels, [self.hidden_dim], self.hidden_dim, "relu", 0.0)
            if self.phys_channels > 0
            else None
        )
        self.demand_proj = (
            build_mlp(self.demand_channels, [self.hidden_dim], self.hidden_dim, "relu", 0.0)
            if self.demand_channels > 0
            else None
        )
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)

    def _encode(
        self,
        proj: nn.Module,
        sequences: list[list[list[float]]],
        valid: list[int],
        device: torch.device | str,
    ) -> torch.Tensor | None:
        """Encode one entity type.

        Args:
            sequences: per-entity windows ``(N, seq_len, C)`` (may be shorter
                on cold start).
            valid: per-entity number of observed steps.

        Returns:
            ``(N, hidden_dim)`` embeddings or ``None`` when the batch is empty.
        """
        if not sequences:
            return None
        device = torch.device(device)
        x = torch.tensor(sequences, dtype=torch.float32, device=device)
        n_entities, seq_len, _ = x.shape
        if n_entities == 0:
            return None
        if all(v <= 0 for v in valid):
            # Cold start: nothing observed yet, skip the LSTM entirely.
            return torch.zeros((n_entities, self.hidden_dim), dtype=torch.float32, device=device)
        lengths = torch.as_tensor(valid, dtype=torch.long, device="cpu").clamp(1, seq_len)

        x = proj(x)  # (N, seq_len, hidden_dim)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=seq_len)

        idx = torch.arange(n_entities, device=device)
        last = out[idx, lengths.to(device) - 1]
        zero = torch.as_tensor(valid, dtype=torch.float32, device=device).unsqueeze(-1) > 0
        return last * zero

    def forward(
        self,
        obs: GraphObservation,
        device: torch.device | str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Encode all entity types.

        Returns:
            ``(node_h, physical_h, demand_h)`` with shapes ``(N, H)``,
            ``(E_phys, H)``, ``(E_demand, H)``; ``None`` when a type batch is
            empty (e.g. no physical link passed the mask).
        """
        node_valid = obs.history_valid[0] if obs.history_valid else []
        phys_valid = obs.history_valid[1] if len(obs.history_valid) > 1 else []
        demand_valid = obs.history_valid[2] if len(obs.history_valid) > 2 else []
        node_h = (
            self._encode(self.node_proj, obs.node_history, node_valid, device)
            if self.node_proj is not None
            else None
        )
        phys_h = (
            self._encode(self.phys_proj, obs.physical_edge_history, phys_valid, device)
            if self.phys_proj is not None
            else None
        )
        demand_h = (
            self._encode(self.demand_proj, obs.demand_edge_history, demand_valid, device)
            if self.demand_proj is not None
            else None
        )
        return node_h, phys_h, demand_h
