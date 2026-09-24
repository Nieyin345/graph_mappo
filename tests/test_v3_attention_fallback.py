"""Off-path arcs must keep the scorer's fallback scale."""

import torch

from qkd_rl.model_zoo.v3.model_impl import PairPathAttentionScorer


def test_off_path_arc_is_not_divided_by_zero_pair_weight():
    scorer = PairPathAttentionScorer(
        hidden_dim=4, num_heads=1, activation="relu", dropout=0.0, arc_query=True
    )
    with torch.no_grad():
        for parameter in scorer.out.parameters():
            parameter.zero_()
        scorer.out[-1].bias.fill_(1.0)
    scores = scorer(
        arc_emb=torch.zeros(2, 12),
        arc_physical_pos=torch.tensor([0, 1]),
        physical_tokens=torch.zeros(2, 12),
        demand_emb=torch.zeros(1, 4),
        pair_masks=[[True, False]],
        pair_weights=[1.0],
    )
    torch.testing.assert_close(scores, torch.ones(2))
