from __future__ import annotations

from dataclasses import dataclass

import torch

from qkd_rl.env.graph_builder import GraphObservation
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


@dataclass
class PolicyStep:
    actions: dict[str, str]
    action_scores: dict[str, dict[str, float]]
    log_probs: dict[str, torch.Tensor]
    entropies: dict[str, torch.Tensor]
    value: torch.Tensor


class MAPPOPolicy:
    def __init__(self, model: GraphMAPPOActorCritic, device: torch.device | str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def act(self, obs: GraphObservation, deterministic: bool = False) -> PolicyStep:
        output = self.model(obs, self.device)
        actions: dict[str, str] = {}
        action_scores: dict[str, dict[str, float]] = {}
        log_probs: dict[str, torch.Tensor] = {}
        entropies: dict[str, torch.Tensor] = {}

        for node_id, logits in output.logits.items():
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = int(torch.argmax(logits).item()) if deterministic else int(dist.sample().item())
            action_name = obs.action_candidates[node_id][action_idx]
            actions[node_id] = action_name
            action_scores[node_id] = {
                candidate: float(score.detach().cpu().item())
                for candidate, score in zip(obs.action_candidates[node_id], logits)
            }
            selected = torch.tensor(action_idx, dtype=torch.long, device=self.device)
            log_probs[node_id] = dist.log_prob(selected)
            entropies[node_id] = dist.entropy()

        return PolicyStep(
            actions=actions,
            action_scores=action_scores,
            log_probs=log_probs,
            entropies=entropies,
            value=output.value,
        )

