"""The pair attention temperature must affect its actual scoring path."""

import torch

from qkd_rl.model_zoo.v3.model_impl import PairPathAttentionScorer


def test_temperature_changes_pair_attention_scores():
    torch.manual_seed(23)
    base = PairPathAttentionScorer(4, 1, "relu", 0.0, arc_query=True)
    warm = PairPathAttentionScorer(4, 1, "relu", 0.0, arc_query=True, logit_temperature=2.0)
    warm.load_state_dict(base.state_dict())
    kwargs = dict(
        arc_emb=torch.randn(2, 12),
        arc_physical_pos=torch.tensor([0, 1]),
        physical_tokens=torch.randn(2, 12),
        demand_emb=torch.randn(1, 4),
        pair_masks=[[True, True]],
        pair_weights=[1.0],
    )
    assert not torch.allclose(base(**kwargs), warm(**kwargs))


def test_temperature_must_be_positive():
    try:
        PairPathAttentionScorer(4, 1, "relu", 0.0, logit_temperature=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero attention temperature should be rejected")
