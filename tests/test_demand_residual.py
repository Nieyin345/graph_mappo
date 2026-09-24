"""Attention contracts, including nonzero residuals and separate batched graphs."""
import copy

import torch

from qkd_rl.core.types import KeyRequest
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.rl.models.graph_mappo import DemandResidual, GraphMAPPOActorCritic
from tests.helpers import point_config_to_h5


def test_all_masked_demand_returns_zero_without_nan():
    layer = DemandResidual(24, 8, 2, "relu", 0.0)
    torch.nn.init.normal_(layer.out[-1].weight)
    output = layer(torch.randn(3, 24), torch.randn(4, 8), torch.zeros(4, dtype=torch.bool))
    assert torch.isfinite(output).all()
    assert torch.equal(output, torch.zeros_like(output))


def test_attention_zero_init_gradient_and_cross_graph_isolation():
    torch.manual_seed(7)
    config = point_config_to_h5(load_default_config("."))
    config["features"]["edge"]["include_stocked_edges"] = True
    env = build_env_from_config(config)
    env.reset(seed=7)
    env.requests.add_arrivals([
        KeyRequest("a", "GS_001", "GS_002", 100.0, env.t, env.t + 30),
        KeyRequest("b", "GS_001", "GS_003", 200.0, env.t, env.t + 30),
    ])
    obs = env._build_observation()
    assert len(obs.demand_edge_ids) == 2
    config["model"]["actor"]["demand_residual"]["enabled"] = False
    base = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    config["model"]["actor"]["demand_residual"]["enabled"] = True
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert missing and all("demand_residual" in key for key in missing)
    assert not unexpected
    baseline = base(obs)
    initial = model(obs)
    for arc in baseline.edge_scores:
        assert torch.equal(baseline.edge_scores[arc], initial.edge_scores[arc])
    torch.stack(list(initial.edge_scores.values())).sum().backward()
    residual = model.actor.demand_residual
    assert residual.out[-1].weight.grad.abs().sum() > 0
    # Activate the residual: zero-init alone would make isolation vacuous.
    torch.nn.init.normal_(residual.out[-1].weight, std=0.1)
    other = copy.deepcopy(obs)
    # Forward input caches must not hide the intervention.
    for attr in ("_tensors_cache", "_actor_plan"):
        if hasattr(other, attr):
            delattr(other, attr)
    other.edge_features[len(other.physical_edge_ids):] += 2.0
    reference = model.batched_forward([obs, obs], "cpu")
    perturbed = model.batched_forward([obs, other], "cpu")
    a0 = torch.stack(list(reference.edge_score_maps[0].values()))
    a1 = torch.stack(list(perturbed.edge_score_maps[0].values()))
    b0 = torch.stack(list(reference.edge_score_maps[1].values()))
    b1 = torch.stack(list(perturbed.edge_score_maps[1].values()))
    torch.testing.assert_close(a0, a1, rtol=0, atol=0)
    assert not torch.allclose(b0, b1)
    model.zero_grad()
    (a1.sum() + b1.sum()).backward()
    for name, parameter in residual.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
