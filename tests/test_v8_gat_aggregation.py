"""v8 encoder: optional GAT-style (GATv2, multi-head) neighbour aggregation.

Journal experiment (b): a GAT encoder baseline. ``model.encoder.aggregation``
switches EdgeConditionedGraphLayer's per-type neighbour aggregation from the
mean to a per-destination softmax attention over the incoming messages.

Pinned here:
  1. zero-initialised attention vectors => uniform weights => EXACTLY the mean
     layer (so the shared BC warm start stays valid and the arm starts from the
     control's function), and building the attention parameters does not
     advance the global RNG (all shared weights identical to the mean layer's);
  2. with non-zero attention the output differs from the mean (the switch is
     not a no-op) and the attention vectors receive gradient at init;
  3. the softmax is per destination node: a disjoint-union batch equals the
     graphs run one by one, and edge order does not matter;
  4. an unknown aggregation name is an error.
"""

from __future__ import annotations

import pytest
import torch

from qkd_rl.model_zoo.v8.model_impl import EdgeConditionedGraphLayer

H = 16


def _layer(aggregation, seed=0):
    torch.manual_seed(seed)
    return EdgeConditionedGraphLayer(
        H, H, "relu", 0.0, True, aggregation=aggregation, attention_heads=4
    )


def _graph(n_nodes, n_phys, n_dem, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, H, generator=g)
    ei_p = torch.randint(0, n_nodes, (2, n_phys), generator=g)
    ei_d = torch.randint(0, n_nodes, (2, n_dem), generator=g)
    phys = torch.randn(n_phys, H, generator=g)
    dem = torch.randn(n_dem, H, generator=g)
    return x, ei_p, ei_d, phys, dem


def _run(layer, x, ei_p, ei_d, phys, dem):
    ei = torch.cat([ei_p, ei_d], dim=1)
    return layer(x, ei, phys, dem, ei_p.size(1))


def _randomise_attention(layer, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if name.startswith("att_vec"):
                p.copy_(torch.randn(p.shape, generator=g) * 2.0)


def test_zero_init_attention_equals_mean_and_keeps_rng():
    mean = _layer("mean")
    probe_mean = torch.nn.Linear(4, 4)        # whatever is built AFTER the layer
    att = _layer("attention")
    probe_att = torch.nn.Linear(4, 4)
    assert torch.equal(probe_mean.weight, probe_att.weight)   # global RNG not advanced
    shared = mean.state_dict()
    att_sd = att.state_dict()
    for k, v in shared.items():
        assert torch.equal(att_sd[k], v), k           # fork_rng: same init
    assert any(k.startswith("att_") for k in att_sd)
    g = _graph(12, 40, 15, seed=3)
    for a, b in zip(_run(mean, *g), _run(att, *g)):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_nonzero_attention_changes_output_and_gets_gradient():
    att = _layer("attention")
    g = _graph(12, 40, 15, seed=3)
    node, _, _ = _run(att, *g)
    node.pow(2).sum().backward()
    grads = [p.grad for n, p in att.named_parameters() if n.startswith("att_vec")]
    assert grads and all(gr is not None and gr.abs().sum() > 0 for gr in grads)
    mean_node, _, _ = _run(_layer("mean"), *g)
    _randomise_attention(att)
    with torch.no_grad():
        node2, _, _ = _run(att, *g)
    assert (node2 - mean_node).abs().max() > 1e-3


def test_batch_equals_single_and_edge_order_invariant():
    att = _layer("attention")
    _randomise_attention(att)
    a = _graph(9, 30, 10, seed=5)
    b = _graph(7, 25, 8, seed=6)
    with torch.no_grad():
        out_a = _run(att, *a)
        out_b = _run(att, *b)
        na = a[0].size(0)
        batch = (
            torch.cat([a[0], b[0]]),
            torch.cat([a[1], b[1] + na], dim=1),
            torch.cat([a[2], b[2] + na], dim=1),
            torch.cat([a[3], b[3]]),
            torch.cat([a[4], b[4]]),
        )
        out = _run(att, *batch)
    torch.testing.assert_close(out[0], torch.cat([out_a[0], out_b[0]]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[1], torch.cat([out_a[1], out_b[1]]), atol=1e-5, rtol=1e-5)
    # permuting the physical edges permutes their outputs and leaves nodes alone
    perm = torch.randperm(a[1].size(1), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out_p = _run(att, a[0], a[1][:, perm], a[2], a[3][perm], a[4])
    torch.testing.assert_close(out_p[0], out_a[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_p[1], out_a[1][perm], atol=1e-5, rtol=1e-5)


def test_unknown_aggregation_rejected():
    with pytest.raises(ValueError):
        _layer("sum")
