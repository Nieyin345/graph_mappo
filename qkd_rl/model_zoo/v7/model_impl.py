"""v7: v3 service scorer plus a separate remaining-port preparation scorer."""

from __future__ import annotations

import torch
from torch import nn

from qkd_rl.model_zoo.v3.model_impl import GraphMAPPOActorCritic as V3ActorCritic
from qkd_rl.rl.models.mlp import build_mlp


class PreparationHead(nn.Module):
    """Score an arc from its v3 embedding and the pair-history stock-gap prior."""

    def __init__(self, hidden_dim: int, activation: str):
        super().__init__()
        self.stop_logit = nn.Parameter(torch.zeros(()))
        self.prior_scale = nn.Parameter(torch.tensor(4.0))
        self.scorer = build_mlp(3 * hidden_dim + 1, [hidden_dim], 1, activation, 0.0)
        last = [module for module in self.scorer.modules() if isinstance(module, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, arc_embeddings: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        learned = self.scorer(torch.cat([arc_embeddings, priors[:, None]], dim=-1)).squeeze(-1)
        return learned + self.prior_scale * priors


class GraphMAPPOActorCritic(V3ActorCritic):
    """Keep v3 stage one intact and expose differentiable stage-two scores."""

    def __init__(self, action_space, config):
        super().__init__(action_space, config)
        hidden = int(config["model"]["encoder"]["hidden_dim"])
        activation = str(config["model"]["encoder"].get("activation", "relu"))
        self.actor.two_stage_enabled = True
        self.actor.prepare = PreparationHead(hidden, activation)

    def forward(self, obs, device="cpu", build_logits_dict=True, want_edge_map=True):
        output = super().forward(
            obs,
            device=device,
            build_logits_dict=build_logits_dict,
            want_edge_map=want_edge_map,
        )
        edge_scores = output.edge_scores or {}
        arcs = list(edge_scores)
        plan = getattr(obs, "_actor_plan", None)
        if plan is None:
            plan = self.actor._build_plan(obs, self.action_space)
            setattr(obs, "_actor_plan", plan)
        physical_positions = (plan[2] // 2).tolist()
        priors_raw = obs.preparation_edge_values or [0.0] * len(obs.physical_edge_ids)
        priors = torch.as_tensor(
            [priors_raw[pos] if pos < len(priors_raw) else 0.0 for pos in physical_positions],
            dtype=output.arc_embeddings.dtype,
            device=output.arc_embeddings.device,
        )
        prep_values = self.actor.prepare(output.arc_embeddings, priors)
        output.preparation_scores = {
            arc: score for arc, score in zip(arcs, prep_values.unbind(0))
        }
        service_physical = set()
        for current, mask in zip(obs.pair_current, obs.pair_path_edge_masks):
            if current:
                service_physical.update(i for i, selected in enumerate(mask) if selected)
        output.service_arcs = {
            arc for arc, pos in zip(arcs, physical_positions) if pos in service_physical
        }
        return output

