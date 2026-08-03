"""Tests for the action mask builder and candidate filtering at graph-build time."""

from __future__ import annotations

from qkd_rl.core.types import Edge, LinkType
from qkd_rl.env.action_space import NodeActionSpace
from tests.helpers import build_test_env
from qkd_rl.env.masks import ActionMaskBuilder
from qkd_rl.env.state import EnvState
from qkd_rl.link.rate_provider import EdgeWindow


class _StubQKP:
    def __init__(self, capacity_left: dict[str, float]):
        self.capacity_left = capacity_left

    def get_capacity_left(self, edge_id: str) -> float:
        return self.capacity_left.get(edge_id, 1.0e9)


def _fixture(rate: float = 1.0, available: bool = True):
    action_space = NodeActionSpace(
        ["A", "B"],
        [Edge("e_ab", "A", "B", LinkType.GS_HAP)],
    )
    window = EdgeWindow("e_ab", [rate, rate], [available, available], LinkType.GS_HAP)
    state = EnvState(
        t=0,
        qkp_snapshot={},
        pending_requests=[],
        edge_windows={"e_ab": window},
        last_activated_edges=[],
    )
    return action_space, state


def _builder(action_space, min_rate: float = 1.0, allowed=None, **overrides) -> ActionMaskBuilder:
    config = {
        "mask_unavailable_edges": True,
        "mask_below_min_rate": True,
        "mask_full_qkp_edges": False,
        "mask_disallowed_link_types": False,
    }
    config.update(overrides)
    return ActionMaskBuilder(
        action_space,
        config,
        min_link_rate=min_rate,
        allowed_link_types=allowed or ["gs_hap"],
    )


def test_idle_is_always_legal():
    action_space, state = _fixture()
    builder = _builder(action_space)
    assert builder.is_action_legal("A", NodeActionSpace.IDLE, state, _StubQKP({}), None) is True


def test_unavailable_edge_is_masked():
    action_space, state = _fixture(available=False)
    builder = _builder(action_space)
    assert builder.is_action_legal("A", "B", state, _StubQKP({}), None) is False


def test_below_min_rate_edge_is_masked():
    action_space, state = _fixture(rate=0.5)
    builder = _builder(action_space, min_rate=1.0)
    assert builder.is_action_legal("A", "B", state, _StubQKP({}), None) is False


def test_disallowed_link_type_is_masked():
    action_space, state = _fixture()
    builder = _builder(action_space, allowed=["gs_sat"], mask_disallowed_link_types=True)
    assert builder.is_action_legal("A", "B", state, _StubQKP({}), None) is False


def test_full_qkp_edge_is_masked_when_enabled():
    action_space, state = _fixture()
    builder = _builder(action_space, mask_full_qkp_edges=True)
    assert builder.is_action_legal("A", "B", state, _StubQKP({"e_ab": 0.0}), None) is False


def test_edge_legal_when_all_checks_pass():
    action_space, state = _fixture()
    builder = _builder(action_space)
    assert builder.is_action_legal("A", "B", state, _StubQKP({}), None) is True


def test_graph_observation_candidates_only_contain_legal_actions():
    env = build_test_env(".")
    obs = env.reset()
    space = env.action_resolver.action_space
    fresh_masks = env.mask_builder.build(obs.state, env.qkp, env.requests)

    for node_id in obs.node_ids:
        full_candidates = space.candidates_for_node(node_id)
        expected_legal = [action for action, ok in zip(full_candidates, fresh_masks[node_id]) if ok]
        assert obs.action_candidates[node_id] == expected_legal
        assert all(obs.action_masks[node_id])
        assert len(obs.action_masks[node_id]) == len(obs.action_candidates[node_id])

