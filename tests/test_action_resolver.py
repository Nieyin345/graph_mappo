from __future__ import annotations

from qkd_rl.core.types import Edge, LinkType
from qkd_rl.env.action_resolver import ActionResolver
from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.state import EnvState
from qkd_rl.link.rate_provider import EdgeWindow
from qkd_rl.algos.policy import _masked_log_prob_entropy

import torch


def _resolver(mode: str) -> ActionResolver:
    action_space = NodeActionSpace(
        ["A", "B", "C"],
        [
            Edge("e_ab", "A", "B", LinkType.GS_HAP),
            Edge("e_bc", "B", "C", LinkType.HAP_SAT),
        ],
    )
    return ActionResolver(action_space, {"mode": mode})


def _weighted_resolver(mode: str = "max_weight_matching") -> ActionResolver:
    action_space = NodeActionSpace(
        ["A", "B", "C", "D"],
        [
            Edge("e_ab", "A", "B", LinkType.GS_HAP),
            Edge("e_ac", "A", "C", LinkType.GS_HAP),
            Edge("e_bd", "B", "D", LinkType.GS_HAP),
        ],
    )
    return ActionResolver(action_space, {"mode": mode})


def _all_valid_masks(resolver: ActionResolver) -> dict[str, list[bool]]:
    return {
        node_id: [True] * len(resolver.action_space.candidates_for_node(node_id))
        for node_id in resolver.action_space.node_ids
    }


def _state() -> EnvState:
    return EnvState(
        t=0,
        qkp_snapshot={},
        pending_requests=[],
        edge_windows={
            "e_ab": EdgeWindow("e_ab", [1.0], [True], LinkType.GS_HAP),
            "e_bc": EdgeWindow("e_bc", [3.0], [True], LinkType.HAP_SAT),
        },
        last_activated_edges=[],
    )


def _weighted_state() -> EnvState:
    return EnvState(
        t=0,
        qkp_snapshot={},
        pending_requests=[],
        edge_windows={
            "e_ab": EdgeWindow("e_ab", [1.0], [True], LinkType.GS_HAP),
            "e_ac": EdgeWindow("e_ac", [1.0], [True], LinkType.GS_HAP),
            "e_bd": EdgeWindow("e_bd", [1.0], [True], LinkType.GS_HAP),
        },
        last_activated_edges=[],
    )


def test_mutual_choice_requires_both_endpoints_to_choose_each_other():
    resolver = _resolver("mutual_choice")
    result = resolver.resolve(
        {"A": "B", "B": "A", "C": "B"},
        _state(),
        _all_valid_masks(resolver),
    )

    assert result.activated_edges == ["e_ab"]


    assert result.rejected_actions == {"C": "B"}
    assert result.conflict_count == 1


def test_max_candidates_per_node_is_a_hard_cap():
    action_space = NodeActionSpace(
        ["A", "B", "C", "D", "E"],
        [
            Edge("e_ab", "A", "B", LinkType.GS_HAP),
            Edge("e_ac", "A", "C", LinkType.GS_HAP),
            Edge("e_ad", "A", "D", LinkType.GS_HAP),
            Edge("e_be", "B", "E", LinkType.GS_HAP),
        ],
    )
    resolver = ActionResolver(
        action_space,
        {"mode": "max_weight_matching", "max_candidates_per_node": 2},
    )
    candidates = [
        (3.0, 1.0, "e_ab", "A", "B"),
        (2.0, 1.0, "e_ac", "A", "C"),
        (1.0, 1.0, "e_ad", "A", "D"),
        (0.9, 1.0, "e_be", "B", "E"),
    ]
    kept = resolver._prune_matching_candidates(candidates)
    kept_by_node: dict[str, int] = {}
    for _score, _rate, _edge_id, src, dst in kept:
        kept_by_node[src] = kept_by_node.get(src, 0) + 1
        kept_by_node[dst] = kept_by_node.get(dst, 0) + 1
    assert kept_by_node["A"] <= 2
    assert kept_by_node["B"] <= 2


def test_priority_matching_uses_actor_scores_to_break_conflicts():
    resolver = _resolver("priority_matching")
    result = resolver.resolve(
        {"A": "B", "B": "A", "C": "B"},
        _state(),
        _all_valid_masks(resolver),
        action_scores={"A": {"B": 1.0}, "B": {"A": 0.5}, "C": {"B": 2.0}},
    )

    assert result.activated_edges == ["e_bc"]
    assert result.rejected_actions == {"A": "B", "B": "A"}


def test_priority_scores_are_invariant_to_per_node_logit_offsets():
    """Equivalent categorical policies must not resolve the same proposals differently."""
    resolver = _resolver("priority_matching")
    actions = {"A": "B", "B": "A", "C": "B"}
    indices = torch.tensor([1, 1, 1])
    logits = torch.tensor([[0.0, 1.0], [0.0, 0.5], [0.0, 2.0]])
    shifted = logits.clone()
    shifted[2] += 100.0

    scores, _ = _masked_log_prob_entropy(logits, indices)
    shifted_scores, _ = _masked_log_prob_entropy(shifted, indices)
    result = resolver.resolve(
        actions,
        _state(),
        _all_valid_masks(resolver),
        action_scores={
            "A": {"B": float(scores[0])},
            "B": {"A": float(scores[1])},
            "C": {"B": float(scores[2])},
        },
    )
    shifted_result = resolver.resolve(
        actions,
        _state(),
        _all_valid_masks(resolver),
        action_scores={
            "A": {"B": float(shifted_scores[0])},
            "B": {"A": float(shifted_scores[1])},
            "C": {"B": float(shifted_scores[2])},
        },
    )

    assert result.activated_edges == shifted_result.activated_edges


def test_greedy_rate_matching_can_be_selected_from_config():
    resolver = _resolver("greedy_rate_matching")
    result = resolver.resolve(
        {"A": "B", "B": "A", "C": "B"},
        _state(),
        _all_valid_masks(resolver),
    )

    assert result.activated_edges == ["e_bc"]


def test_max_weight_matching_prefers_global_optimum_over_single_best_edge():
    resolver = _weighted_resolver()
    result = resolver.resolve(
        {"A": "B", "B": "A", "C": "A", "D": "B"},
        _weighted_state(),
        _all_valid_masks(resolver),
        edge_scores={"e_ab": 10.0, "e_ac": 6.0, "e_bd": 6.0},
    )

    assert result.activated_edges == ["e_ac", "e_bd"]
    assert result.rejected_actions == {"A": "B", "B": "A"}


def test_max_weight_matching_rate_tiebreak_cannot_override_scores():
    resolver = _weighted_resolver()
    state = _weighted_state()
    state.edge_windows["e_ab"].rates[0] = 1.0e12
    state.edge_windows["e_ac"].rates[0] = 1.0
    state.edge_windows["e_bd"].rates[0] = 1.0

    result = resolver.resolve(
        {"A": "B", "B": "A", "C": "A", "D": "B"},
        state,
        _all_valid_masks(resolver),
        edge_scores={"e_ab": 1.0, "e_ac": 1.01, "e_bd": 1.01},
    )

    assert result.activated_edges == ["e_ac", "e_bd"]


def test_max_weight_matching_does_not_collapse_to_empty_when_scores_are_negative():
    action_space = NodeActionSpace(
        ["A", "B"],
        [Edge("e_ab", "A", "B", LinkType.GS_HAP)],
    )
    resolver = ActionResolver(action_space, {"mode": "max_weight_matching"})
    result = resolver.resolve(
        {"A": "B", "B": "A"},
        EnvState(
            t=0,
            qkp_snapshot={},
            pending_requests=[],
            edge_windows={"e_ab": EdgeWindow("e_ab", [1.0], [True], LinkType.GS_HAP)},
            last_activated_edges=[],
        ),
        _all_valid_masks(resolver),
        edge_scores={"e_ab": -10.0},
    )

    assert result.activated_edges == ["e_ab"]
