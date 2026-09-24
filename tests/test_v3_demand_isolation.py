"""A request must not enter another request's GNN token when diffusion is off."""

import torch

from qkd_rl.model_zoo.v3.model_impl import EdgeConditionedGraphLayer


def test_isolated_demand_edges_do_not_mix_through_shared_endpoint():
    torch.manual_seed(11)
    isolated = EdgeConditionedGraphLayer(
        hidden_dim=8, edge_dim=8, activation="relu", dropout=0.0,
        layer_norm=True, fuse_demand_to_node=False,
    )
    mixed = EdgeConditionedGraphLayer(
        hidden_dim=8, edge_dim=8, activation="relu", dropout=0.0,
        layer_norm=True, fuse_demand_to_node=True,
    )
    mixed.load_state_dict(isolated.state_dict())
    node = torch.randn(3, 8)
    demand = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
    changed = demand.clone()
    changed[:2] += 2.0

    def other_request(layer, edges):
        return layer(node, edge_index, edges.new_zeros((0, 8)), edges, 0)[2][2:]

    torch.testing.assert_close(other_request(isolated, demand), other_request(isolated, changed))
    assert not torch.allclose(other_request(mixed, demand), other_request(mixed, changed))
