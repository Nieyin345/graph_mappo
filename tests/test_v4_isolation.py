"""v4's pair-isolation, exact path-union and inventory-only-link contract."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qkd_rl.env.factory import build_env_from_config
from qkd_rl.model_zoo import build_model
from qkd_rl.model_zoo.v4.graph_builder import GraphBuilder
from qkd_rl.model_zoo.v4.model_impl import (
    GraphEncoder, GraphTensors, PairPathAttentionScorer,
)
from qkd_rl.rl.algos.policy import MAPPOPolicy
from scripts.experiment import resolve_experiment
from tests.helpers import point_config_to_h5


def _config():
    return resolve_experiment("v4", seed=42, updates=1, overrides=[], run_name="test_v4")


def test_v4_config_strictly_isolates_demand_from_physical_graph():
    cfg = _config()
    assert cfg["features"]["edge"]["include_stocked_edges"] is True
    assert cfg["features"]["edge"]["include_relay_importance"] is False
    assert cfg["features"]["edge"]["include_req_hop"] is False
    assert cfg["features"]["edge"]["include_on_pending_path"] is False
    assert all(cfg["features"]["node"][name] is False for name in (
        "include_demand_in", "include_demand_out", "include_queue_pressure", "include_recent_demand"
    ))
    assert cfg["features"]["demand_edge"]["include_pair_history"] is False
    assert cfg["features"]["history_encoder"]["enabled"] is False
    assert cfg["model"]["encoder"]["fuse_demand_to_node"] is False
    assert cfg["model"]["actor"]["pair_path_attention"]["max_hops"] == 4
    assert cfg["train"]["episodes_per_update"] == 8
    assert cfg["train"]["rollout_steps"] == 240


def test_path_mask_is_union_of_all_simple_routes_not_backtracking_walks():
    # The s-a-x dead end satisfies v3's reverse distance-sum walk bound via
    # s-a-x-a-t but belongs to NO simple s-t path. Alternative 2/3/4-hop
    # routes must all contribute their edges; the fifth-hop route must not.
    edge_pairs = [
        ("s", "a"), ("a", "t"),
        ("s", "b"), ("b", "c"), ("c", "t"),
        ("a", "x"),  # stock-only dead end: do not add to the path mask
        ("s", "d"), ("d", "e"), ("e", "f"), ("f", "t"),
        ("s", "g"), ("g", "h"), ("h", "i"), ("i", "j"), ("j", "t"),
    ]
    builder = GraphBuilder.__new__(GraphBuilder)
    builder.node_index = {node: i for i, node in enumerate({n for pair in edge_pairs for n in pair})}
    builder.node_index["isolated"] = len(builder.node_index)
    edges = [SimpleNamespace(src=u, dst=v) for u, v in edge_pairs]
    mask, reverse, unreachable = builder._pair_shortest_path_masks(
        edges, [("s", "t"), ("t", "s"), ("x", "isolated")], max_hops=4
    )
    assert mask == [True] * 5 + [False] + [True] * 4 + [False] * 5
    assert reverse == mask
    assert unreachable == [False] * len(edges)
    # One stock-only edge on the 2-hop path still participates in context.
    assert mask[1] is True
    with pytest.raises(ValueError, match="max_hops"):
        builder._pair_shortest_path_masks(edges, [("s", "t")])


def test_pair_local_attention_never_sees_other_pairs_context():
    torch.manual_seed(10)
    hidden = 8
    scorer = PairPathAttentionScorer(hidden, 2, "relu", 0.0, arc_query=True)
    scorer.eval()
    arcs = torch.randn(3, 3 * hidden)
    phys = torch.randn(3, 3 * hidden, requires_grad=True)
    demands = torch.randn(2, hidden, requires_grad=True)
    # Physical edge #1 has only stored keys; it is never a generation arc.
    positions = torch.tensor([0, 0, 2])
    # Edges in different masks share endpoint 2; global attention would leak.
    endpoints = torch.tensor([[0, 1], [1, 2], [2, 3]])
    masks = [[True, True, False], [False, False, True]]
    scores = scorer(arcs, positions, phys, demands, masks, [1.0, 1.0], endpoints)
    changed_phys = torch.cat([phys[:2], phys[2:] + 50], dim=0)
    changed_demand = torch.cat([demands[:1], demands[1:] + 50], dim=0)
    changed = scorer(arcs, positions, changed_phys, changed_demand, masks, [1.0, 1.0], endpoints)
    torch.testing.assert_close(scores[:2], changed[:2], rtol=0, atol=1e-6)
    grad_phys, grad_demand = torch.autograd.grad(scores[0], (phys, demands))
    torch.testing.assert_close(grad_phys[2], torch.zeros_like(grad_phys[2]))
    torch.testing.assert_close(grad_demand[1], torch.zeros_like(grad_demand[1]))
    assert grad_phys[:2].abs().sum() > 0
    assert grad_demand[0].abs().sum() > 0
    # A stocked rate-zero physical token, although not a candidate arc, is
    # still available as attention context for a legal rate-positive arc.
    assert grad_phys[1].abs().sum() > 0
    assert torch.isfinite(scores).all()


def test_physical_encoder_never_receives_other_demand_tokens():
    torch.manual_seed(18)
    enc = GraphEncoder(
        node_dim=5, edge_dim=4,
        config={"hidden_dim": 8, "num_layers": 2, "activation": "relu",
                "dropout": 0.0, "layer_norm": True, "residual": True},
        fuse_demand_to_node=False,
    ).eval()
    edge_index = torch.tensor([
        [0, 1, 1, 2, 0, 2, 0, 3],
        [1, 0, 2, 1, 2, 0, 3, 0],
    ])  # 2 physical undirected, then 2 bidirectional demand pairs
    feat = torch.randn(8, 4, requires_grad=True)
    graph = GraphTensors(
        node_features=torch.randn(4, 5), edge_index=edge_index,
        edge_features_directed=feat,
        node_ids=["s", "r", "t", "u"],
        edge_ids=["s-r", "r-t", "s-t-demand", "s-u-demand"],
        num_physical_directed=4,
    )
    nodes, tokens = enc(graph)
    other = GraphTensors(
        node_features=graph.node_features, edge_index=edge_index,
        edge_features_directed=torch.cat([feat[:6], feat[6:] + 100]),
        node_ids=graph.node_ids, edge_ids=graph.edge_ids,
        num_physical_directed=4,
    )
    other_nodes, other_tokens = enc(other)
    torch.testing.assert_close(nodes, other_nodes, rtol=0, atol=1e-6)
    torch.testing.assert_close(tokens[:6], other_tokens[:6], rtol=0, atol=1e-6)
    grad = torch.autograd.grad(tokens[4:6].sum() + nodes.sum(), feat)[0]
    torch.testing.assert_close(grad[6:], torch.zeros_like(grad[6:]))
    assert grad[4:6].abs().sum() > 0
    assert tokens[4:6].shape == (2, 8)
    with pytest.raises(ValueError, match="never passes demand"):
        GraphEncoder(5, 4, {"hidden_dim": 8, "num_layers": 1}, fuse_demand_to_node=True)


def test_rate_zero_stocked_edge_visible_but_never_actionable_and_batch_matches():
    cfg = point_config_to_h5(_config())
    cfg["qkp"]["initial_level"] = 0.0
    env = build_env_from_config(cfg)
    initial = env.reset(seed=7)
    edge = next(e for e in env.scenario.edges if e.edge_id not in initial.physical_edge_ids)
    env.qkp.add_keys(edge.edge_id, 100.0, env.t)
    obs = env._build_observation()
    assert edge.edge_id in obs.physical_edge_ids
    assert edge.edge_id not in obs.generation_edge_ids
    assert edge.dst not in obs.action_candidates[edge.src]
    assert edge.src not in obs.action_candidates[edge.dst]
    assert np.isfinite(obs.edge_features).all()
    assert len(obs.pair_path_edge_masks) == len(obs.demand_edge_ids)
    assert all(len(mask) == len(obs.physical_edge_ids) for mask in obs.pair_path_edge_masks)
    model = build_model(env.action_resolver.action_space, cfg)
    assert model.__class__.__module__.startswith("qkd_rl.model_zoo.v4")
    model.eval()
    single = model(obs)
    batch = model.batched_forward([obs], "cpu")
    assert (edge.src, edge.dst) not in single.edge_scores
    torch.testing.assert_close(single.value, batch.values[0], rtol=1e-5, atol=1e-5)
    for arc, score in single.edge_scores.items():
        torch.testing.assert_close(score, batch.edge_score_maps[0][arc], rtol=1e-5, atol=1e-5)
    loss = batch.values[0] + sum(batch.edge_score_maps[0].values(), batch.values[0].new_zeros(()))
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.encoder.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.actor.pair_path_scorer.parameters())
    policy = MAPPOPolicy(model)
    step = policy.act(obs, deterministic=True)
    env.step(step.actions, expected_matched_edges=step.matched_edges)


def test_v4_rejects_configs_that_reintroduce_cross_pair_demand_leak():
    cfg = point_config_to_h5(_config())
    env = build_env_from_config(cfg)
    env.reset(seed=3)
    cfg = deepcopy(cfg)
    cfg["features"]["edge"]["include_req_hop"] = True
    with pytest.raises(ValueError, match="include_req_hop"):
        build_model(env.action_resolver.action_space, cfg)
