from __future__ import annotations

from qkd_rl.core.types import Edge, LinkType
from qkd_rl.env.action_resolver import ActionResolver
from qkd_rl.env.action_space import NodeActionSpace
from qkd_rl.env.state import EnvState
from qkd_rl.link.rate_provider import EdgeWindow


def _resolver(mode: str) -> ActionResolver:
    action_space = NodeActionSpace(
        ["A", "B", "C"],
        [
            Edge("e_ab", "A", "B", LinkType.GS_HAP),
            Edge("e_bc", "B", "C", LinkType.HAP_SAT),
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


def test_greedy_rate_matching_can_be_selected_from_config():
    resolver = _resolver("greedy_rate_matching")
    result = resolver.resolve(
        {"A": "B", "B": "A", "C": "B"},
        _state(),
        _all_valid_masks(resolver),
    )

    assert result.activated_edges == ["e_bc"]
