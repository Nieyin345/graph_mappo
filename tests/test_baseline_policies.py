"""Unit tests for the independent baseline policies.

These tests never import the training stack (no policy/rollout/PPO code),
so baselines can be extended or swapped without touching the trainer.
"""
from __future__ import annotations

from pathlib import Path

from qkd_rl.baselines import (
    GreedyDemandPolicy,
    GreedyMatchingPolicy,
    GreedyQKPPolicy,
    GreedyRatePolicy,
    GreedyRelayPolicy,
    RandomPolicy,
)
from qkd_rl.core.types import KeyRequest, LinkType
from qkd_rl.env.graph_builder import GraphObservation
from qkd_rl.env.state import EnvState
from tests.helpers import build_test_env
from qkd_rl.link.rate_provider import EdgeWindow

ROOT = Path(__file__).resolve().parents[1]


def _minimal_obs(
    node_ids: list[str],
    edge_pairs: list[tuple[str, str]],
    demand: dict[tuple[str, str], float],
    levels: dict[str, float],
    inactive_pairs: list[tuple[str, str]] | None = None,
) -> GraphObservation:
    inactive_pairs = inactive_pairs or []
    all_pairs = edge_pairs + inactive_pairs
    edge_ids = [f"E_{src}__{dst}" for src, dst in all_pairs]
    active_edge_ids = [f"E_{src}__{dst}" for src, dst in edge_pairs]
    windows = {
        edge_id: EdgeWindow(
            edge_id=edge_id,
            rates=[10.0],
            available=[edge_id in active_edge_ids],
            link_type=LinkType.GS_SAT,
        )
        for edge_id in edge_ids
    }
    pair_to_edge = {tuple(sorted(pair)): edge_id for pair, edge_id in zip(all_pairs, edge_ids)}
    requests = [
        KeyRequest(
            request_id=f"req_{i}",
            src_gs=src,
            dst_gs=dst,
            amount=amount,
            arrival_t=0,
            deadline_t=100,
        )
        for i, ((src, dst), amount) in enumerate(demand.items())
    ]
    state = EnvState(
        t=0,
        qkp_snapshot=dict(levels),
        pending_requests=requests,
        edge_windows=windows,
        last_activated_edges=[],
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
        physical_edge_ids=active_edge_ids,
        demand_edge_ids=[],
        action_candidates=candidates,
        action_masks={node: [True] * len(cands) for node, cands in candidates.items()},
        state=state,
    )


def test_greedy_demand_picks_largest_demand_pair() -> None:
    obs = _minimal_obs(
        node_ids=["GS1", "GS2", "GS3"],
        edge_pairs=[("GS1", "GS2"), ("GS1", "GS3")],
        demand={("GS1", "GS2"): 50.0, ("GS1", "GS3"): 10.0},
        levels={},
    )
    actions, scores = GreedyDemandPolicy().act(obs)
    assert actions["GS1"] == "GS2"
    assert scores["GS1"]["GS2"] > scores["GS1"]["GS3"]


def test_greedy_qkp_picks_highest_level_edge() -> None:
    obs = _minimal_obs(
        node_ids=["GS1", "GS2", "GS3"],
        edge_pairs=[("GS1", "GS2"), ("GS1", "GS3")],
        demand={},
        levels={"E_GS1__GS2": 3.0, "E_GS1__GS3": 9.0},
    )
    actions, scores = GreedyQKPPolicy().act(obs)
    assert actions["GS1"] == "GS3"
    assert scores["GS1"]["GS3"] > scores["GS1"]["GS2"]


def test_greedy_matching_submits_mutual_actions() -> None:
    obs = _minimal_obs(
        node_ids=["GS1", "GS2", "GS3"],
        edge_pairs=[("GS1", "GS2"), ("GS1", "GS3")],
        demand={},
        levels={"E_GS1__GS2": 5.0, "E_GS1__GS3": 5.0},
    )
    # With equal scores only one of GS1's two edges can be matched (a node
    # belongs to at most one activated link); the matched pair must agree and
    # the leftover node goes idle.
    actions, _scores = GreedyQKPPolicy().act(obs)
    assert actions["GS1"] in ("GS2", "GS3")
    assert actions[actions["GS1"]] == "GS1"
    leftover = next(node for node in ("GS2", "GS3") if node != actions["GS1"])
    assert actions[leftover] == "idle"


def test_greedy_matching_prefers_highest_scored_disjoint_edges() -> None:
    # A 4-node line GS1-GS2-GS3-GS4: the two strongest edges are disjoint and
    # both get activated; the weakest edge (GS1-GS2) is skipped only when it
    # conflicts with a better match.
    obs = _minimal_obs(
        node_ids=["GS1", "GS2", "GS3", "GS4"],
        edge_pairs=[("GS1", "GS2"), ("GS2", "GS3"), ("GS3", "GS4")],
        demand={},
        levels={"E_GS1__GS2": 1.0, "E_GS2__GS3": 9.0, "E_GS3__GS4": 8.0},
    )
    actions, _scores = GreedyQKPPolicy().act(obs)
    assert actions["GS2"] == "GS3"
    assert actions["GS3"] == "GS2"
    assert actions["GS1"] == "idle"
    assert actions["GS4"] == "idle"


def test_greedy_matching_combined_policy_runs_in_real_env() -> None:
    env = build_test_env(ROOT)
    obs = env.reset(seed=11)
    policy = GreedyMatchingPolicy()
    actions, scores = policy.act(obs)
    for node in obs.node_ids:
        assert actions[node] in obs.action_candidates[node]
        assert set(scores[node]) == set(obs.action_candidates[node])
    # Mutual consistency for every non-idle action.
    for node, action in actions.items():
        if action != "idle":
            assert actions[action] == node


def test_baseline_policies_return_legal_actions_in_real_env() -> None:
    policies = [RandomPolicy(seed=0), GreedyRatePolicy(), GreedyDemandPolicy(), GreedyQKPPolicy()]
    for policy in policies:
        env = build_test_env(ROOT)
        obs = env.reset(seed=42)
        actions, scores = policy.act(obs)
        for node in obs.node_ids:
            assert actions[node] in obs.action_candidates[node]
            assert set(scores[node]) == set(obs.action_candidates[node])


def test_baseline_policies_complete_an_episode() -> None:
    env = build_test_env(ROOT)
    obs = env.reset(seed=7)
    for policy in [GreedyDemandPolicy(), GreedyQKPPolicy()]:
        done = False
        while not done:
            actions, scores = policy.act(obs)
            obs, _reward, terminated, truncated, _info = env.step(actions, scores)
            done = terminated or truncated
        summary = env.metrics.episode_summary()
        assert summary["arrived_keys"] >= 0.0
        # success_rate = served / arrived can exceed 1.0 when requests are
        # served from pre-existing QKP stock (small env starts with level>0).
        assert 0.0 <= summary["success_rate"] <= 2.0
        obs = env.reset(seed=7)


def test_greedy_relay_completes_half_stocked_path() -> None:
    # GS1 -> GS2 -> GS3 relay path; GS1-GS2 already holds keys, so the policy
    # must activate the completing edge GS2-GS3 instead of idling both.
    obs = _minimal_obs(
        node_ids=["GS1", "GS2", "GS3"],
        edge_pairs=[("GS1", "GS2"), ("GS2", "GS3")],
        demand={("GS1", "GS3"): 100.0},
        levels={"E_GS1__GS2": 5.0},
    )
    actions, scores = GreedyRelayPolicy().act(obs)
    assert actions["GS2"] == "GS3"
    assert actions["GS3"] == "GS2"
    assert actions["GS1"] == "idle"
    assert scores["GS2"]["GS3"] > scores["GS2"]["idle"]


def test_greedy_relay_prefers_demand_path_over_high_rate() -> None:
    # R1 carries the only 2-hop path for the pending GS-A <-> GS-B demand while
    # R2 offers a much higher rate but no demand path; the relay policy must
    # choose the demand path over raw rate.
    obs = _minimal_obs(
        node_ids=["A", "B", "R1", "R2"],
        edge_pairs=[("A", "R1"), ("B", "R1"), ("A", "R2")],
        demand={("A", "B"): 200.0},
        levels={},
    )
    obs.state.edge_windows["E_A__R2"] = obs.state.edge_windows["E_A__R2"].__class__(
        edge_id="E_A__R2",
        rates=[1000.0],
        available=[True],
        link_type=LinkType.GS_SAT,
    )
    obs.state.edge_windows["E_A__R1"] = obs.state.edge_windows["E_A__R1"].__class__(
        edge_id="E_A__R1",
        rates=[1.0],
        available=[True],
        link_type=LinkType.GS_SAT,
    )
    obs.state.edge_windows["E_B__R1"] = obs.state.edge_windows["E_B__R1"].__class__(
        edge_id="E_B__R1",
        rates=[1.0],
        available=[True],
        link_type=LinkType.GS_SAT,
    )
    actions, _scores = GreedyRelayPolicy().act(obs)
    # R1 (relay) may activate at most one link per slot, so it picks the
    # demand path edge A-R1 now; B-R1 completes the path on the next slot.
    assert actions["A"] == "R1"
    assert actions["R1"] == "A"
    assert actions["B"] == "idle"
    assert actions["R2"] == "idle"


def test_stocked_unavailable_edges_are_included_by_default() -> None:
    from qkd_rl.baselines.greedy_relay_diffusion import (
        GreedyRelayDiffusionPolicyV2,
        GreedyRelayDiffusionPolicyV3,
    )

    # A-R is the only selectable edge. B-R holds keys but is unavailable this
    # slot, so the pending A<->B demand is only visible to the BFS when the
    # stocked-unavailable edge is included in the potential graph.
    obs = _minimal_obs(
        node_ids=["A", "B", "R"],
        edge_pairs=[("A", "R")],
        demand={("A", "B"): 100.0},
        levels={"E_A__R": 0.0, "E_B__R": 50.0},
        inactive_pairs=[("B", "R")],
    )
    obs.state.qkp_capacity = {"E_A__R": 100.0, "E_B__R": 100.0}

    v2 = GreedyRelayDiffusionPolicyV2()
    v3 = GreedyRelayDiffusionPolicyV3()
    _, scores_v2 = v2.act(obs)
    _, scores_v3 = v3.act(obs)
    v2_disabled = GreedyRelayDiffusionPolicyV2(include_stocked_unavailable=False)
    _, scores_disabled = v2_disabled.act(obs)

    assert scores_v3["A"]["R"] > scores_v2["A"]["R"]
    assert scores_v2["A"]["R"] > scores_v2["A"]["idle"]
    assert v2.include_stocked_unavailable is True
    assert v2_disabled.include_stocked_unavailable is False
    assert v3.include_stocked_unavailable is True
    assert v3.importance_weight == 10.0


def test_pending_demand_uses_remaining_amount() -> None:
    from dataclasses import replace

    from qkd_rl.baselines._matching import pending_demand

    obs = _minimal_obs(
        node_ids=["A", "B"],
        edge_pairs=[("A", "B")],
        demand={("A", "B"): 100.0},
        levels={},
    )
    obs.state.pending_requests[0] = replace(obs.state.pending_requests[0], served_amount=40.0)
    pair_demand, incident = pending_demand(obs)
    assert pair_demand[("A", "B")] == 60.0
    assert incident["A"] == 60.0
    assert incident["B"] == 60.0
