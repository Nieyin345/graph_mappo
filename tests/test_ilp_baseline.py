"""Unit tests for the MILP-optimal baseline.

Mirrors ``test_baseline_policies.py``: constructs a minimal
:class:`GraphObservation` directly, so the solver is tested in isolation from
the training stack.  The policy is a real MILP (binary link activation + path
choice, continuous per-request served amount), so the tests also pin the
partial-service, remaining-demand, switch-cost and capacity semantics.
"""
from __future__ import annotations

import pytest

from qkd_rl.baselines import ILPOptimalPolicy
from qkd_rl.core.types import KeyRequest, LinkType
from qkd_rl.env.graph_builder import GraphObservation
from qkd_rl.env.state import EnvState
from qkd_rl.link.rate_provider import EdgeWindow


def _make_obs(
    node_ids: list[str],
    edge_pairs: list[tuple[str, str]],
    demand: dict[tuple[str, str], float],
    levels: dict[str, float],
    rates: dict[str, float] | None = None,
    served: dict[str, float] | None = None,
    capacities: dict[str, float] | None = None,
    last_activated_edges: list[str] | None = None,
) -> GraphObservation:
    rates = rates or {}
    served = served or {}
    edge_ids = [f"E_{src}__{dst}" for src, dst in edge_pairs]
    windows = {
        edge_id: EdgeWindow(
            edge_id=edge_id,
            rates=[rates.get(edge_id, 10.0)],
            available=[True],
            link_type=LinkType.GS_SAT,
        )
        for edge_id in edge_ids
    }
    requests = [
        KeyRequest(
            request_id=f"req_{i}",
            src_gs=src,
            dst_gs=dst,
            amount=amount,
            arrival_t=0,
            deadline_t=100,
            served_amount=served.get(f"req_{i}", 0.0),
        )
        for i, ((src, dst), amount) in enumerate(demand.items())
    ]
    state = EnvState(
        t=0,
        qkp_snapshot=dict(levels),
        pending_requests=requests,
        edge_windows=windows,
        last_activated_edges=last_activated_edges or [],
        qkp_capacity=dict(capacities) if capacities else None,
    )
    candidates: dict[str, list[str]] = {}
    for node in node_ids:
        neighbours = sorted(
            {dst for src, dst in edge_pairs if src == node} | {src for src, dst in edge_pairs if dst == node}
        )
        candidates[node] = ["idle"] + neighbours
    return GraphObservation(
        node_features=[[0.0] for _ in node_ids],
        edge_index=[],
        edge_features=[[0.0] for _ in edge_ids],
        node_ids=node_ids,
        edge_ids=edge_ids,
        physical_edge_ids=edge_ids,
        demand_edge_ids=[],
        action_candidates=candidates,
        action_masks={node: [True] * len(cands) for node, cands in candidates.items()},
        state=state,
    )


def test_ilp_serves_single_edge_request() -> None:
    obs = _make_obs(
        node_ids=["GS1", "GS2"],
        edge_pairs=[("GS1", "GS2")],
        demand={("GS1", "GS2"): 50.0},
        levels={},  # no existing inventory: serving requires activation
    )
    policy = ILPOptimalPolicy(slot_seconds=60.0)
    outcome = policy.solve(obs)
    assert outcome.status == 0
    assert outcome.served_amount == 50.0
    assert outcome.activated_edges == ["E_GS1__GS2"]
    actions, scores = policy.act(obs)
    assert actions["GS1"] == "GS2"
    assert actions["GS2"] == "GS1"
    assert scores["GS1"]["GS2"] > 0.0


def test_ilp_matching_constraint_serves_best_request() -> None:
    # GS1 can activate at most one incident link, so of the two requests only
    # the larger one (GS1->GS3, amount 100) can be served.
    obs = _make_obs(
        node_ids=["GS1", "GS2", "GS3"],
        edge_pairs=[("GS1", "GS2"), ("GS1", "GS3")],
        demand={("GS1", "GS2"): 10.0, ("GS1", "GS3"): 100.0},
        levels={},
    )
    outcome = ILPOptimalPolicy(slot_seconds=60.0).solve(obs)
    assert outcome.served_amount == pytest.approx(100.0, abs=1e-3)
    assert outcome.served_request_ids == ["req_1"]
    assert outcome.activated_edges == ["E_GS1__GS3"]


def test_ilp_serves_multihop_from_existing_inventory() -> None:
    # GS1 -> SAT -> GS3 with enough pre-existing inventory on both links: the
    # request can be served without activating either link this slot.
    # Zero generation isolates the pure-inventory path: the request must be
    # served from the existing pool without any generation this slot.
    obs = _make_obs(
        node_ids=["GS1", "SAT1", "GS3"],
        edge_pairs=[("GS1", "SAT1"), ("SAT1", "GS3")],
        demand={("GS1", "GS3"): 100.0},
        levels={"E_GS1__SAT1": 200.0, "E_SAT1__GS3": 200.0},
        rates={"E_GS1__SAT1": 0.0, "E_SAT1__GS3": 0.0},
    )
    outcome = ILPOptimalPolicy(slot_seconds=60.0).solve(obs)
    assert outcome.served_amount == pytest.approx(100.0, abs=1e-3)
    # With no generation the secondary (stocking) objective is flat, so the
    # activation set is not unique; serving must not depend on it.
    actions, _ = ILPOptimalPolicy(slot_seconds=60.0).act(obs)
    assert actions["GS1"] == "idle"
    assert actions["SAT1"] == "idle"
    assert actions["GS3"] == "idle"


def test_ilp_act_returns_legal_actions() -> None:
    obs = _make_obs(
        node_ids=["GS1", "GS2", "GS3"],
        edge_pairs=[("GS1", "GS2"), ("GS1", "GS3")],
        demand={("GS1", "GS3"): 100.0},
        levels={},
    )
    policy = ILPOptimalPolicy(slot_seconds=60.0)
    actions, scores = policy.act(obs)
    for node in obs.node_ids:
        assert actions[node] in obs.action_candidates[node]
        assert set(scores[node]) == set(obs.action_candidates[node])
    assert actions["GS1"] == "GS3"
    assert actions["GS3"] == "GS1"
    assert actions["GS2"] == "idle"


def test_ilp_partial_service_up_to_bottleneck() -> None:
    # The two-hop path holds 30 keys on the first hop and 200 on the second:
    # the request (100) is partially served with only the bottleneck amount.
    obs = _make_obs(
        node_ids=["GS1", "SAT1", "GS3"],
        edge_pairs=[("GS1", "SAT1"), ("SAT1", "GS3")],
        demand={("GS1", "GS3"): 100.0},
        levels={"E_GS1__SAT1": 30.0, "E_SAT1__GS3": 200.0},
        # Zero generation isolates the pure inventory bottleneck: the MILP
        # must not be able to top up the weak hop by activating it.
        rates={"E_GS1__SAT1": 0.0, "E_SAT1__GS3": 0.0},
    )
    outcome = ILPOptimalPolicy(slot_seconds=60.0).solve(obs)
    assert outcome.served_amount == pytest.approx(30.0, abs=1e-3)
    assert outcome.flow_by_request == pytest.approx({"req_0": 30.0}, abs=1e-3)


def test_ilp_respects_remaining_demand() -> None:
    # The request already had 80 of its 100 served: only 20 remain, so the
    # MILP must cap the flow at 20 even though the link can generate more.
    obs = _make_obs(
        node_ids=["GS1", "GS2"],
        edge_pairs=[("GS1", "GS2")],
        demand={("GS1", "GS2"): 100.0},
        levels={},
        served={"req_0": 80.0},
    )
    outcome = ILPOptimalPolicy(slot_seconds=60.0).solve(obs)
    assert outcome.served_amount == pytest.approx(20.0, abs=1e-3)


def test_ilp_switch_decay_on_new_activation() -> None:
    # rate 10 * slot 60 = 600 keys per slot at full rate. A kept-active link
    # serves 600; a newly activated link generates at decay 0.5 -> 300.
    obs_keep = _make_obs(
        node_ids=["GS1", "GS2"],
        edge_pairs=[("GS1", "GS2")],
        demand={("GS1", "GS2"): 1000.0},
        levels={},
        last_activated_edges=["E_GS1__GS2"],
    )
    assert ILPOptimalPolicy(slot_seconds=60.0).solve(obs_keep).served_amount == pytest.approx(600.0, abs=1e-3)

    obs_switch = _make_obs(
        node_ids=["GS1", "GS2"],
        edge_pairs=[("GS1", "GS2")],
        demand={("GS1", "GS2"): 1000.0},
        levels={},
        last_activated_edges=[],
    )
    assert ILPOptimalPolicy(slot_seconds=60.0).solve(obs_switch).served_amount == pytest.approx(300.0, abs=1e-3)


def test_ilp_end_to_end_in_env() -> None:
    """Run the MILP policy against the real small env for a few steps."""
    from tests.helpers import build_test_env

    env = build_test_env()
    obs = env.reset(seed=7)
    policy = ILPOptimalPolicy(slot_seconds=60.0, time_limit_s=2.0)
    for _ in range(5):
        actions, _scores = policy.act(obs)
        for node in obs.node_ids:
            assert actions[node] in obs.action_candidates[node]
        obs, _reward, _term, _trunc, _info = env.step(actions)
        assert policy.last_outcome is not None


def test_ilp_capacity_caps_service() -> None:
    # The pool is already at capacity (200), so even though the activated link
    # would generate 600 more keys, the flow is capped at the capacity.
    obs = _make_obs(
        node_ids=["GS1", "GS2"],
        edge_pairs=[("GS1", "GS2")],
        demand={("GS1", "GS2"): 1000.0},
        levels={"E_GS1__GS2": 200.0},
        capacities={"E_GS1__GS2": 200.0},
        last_activated_edges=["E_GS1__GS2"],
    )
    outcome = ILPOptimalPolicy(slot_seconds=60.0).solve(obs)
    assert outcome.served_amount == pytest.approx(200.0, abs=1e-3)