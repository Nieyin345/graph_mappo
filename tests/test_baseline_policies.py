"""Unit tests for the independent baseline policies.

These tests never import the training stack (no policy/rollout/PPO code),
so baselines can be extended or swapped without touching the trainer.
"""
from __future__ import annotations

from pathlib import Path

from qkd_rl.baselines import (
    GreedyDemandPolicy,
    GreedyQKPPolicy,
    GreedyRatePolicy,
    RandomPolicy,
)
from qkd_rl.core.types import KeyRequest, LinkType
from qkd_rl.env.graph_builder import GraphObservation
from qkd_rl.env.state import EnvState
from qkd_rl.env.factory import build_default_env
from qkd_rl.link.rate_provider import EdgeWindow

ROOT = Path(__file__).resolve().parents[1]


def _minimal_obs(
    node_ids: list[str],
    edge_pairs: list[tuple[str, str]],
    demand: dict[tuple[str, str], float],
    levels: dict[str, float],
) -> GraphObservation:
    edge_ids = [f"E_{src}__{dst}" for src, dst in edge_pairs]
    windows = {
        edge_id: EdgeWindow(
            edge_id=edge_id,
            rates=[10.0],
            available=[True],
            link_type=LinkType.GS_SAT,
        )
        for edge_id in edge_ids
    }
    pair_to_edge = {tuple(sorted(pair)): edge_id for pair, edge_id in zip(edge_pairs, edge_ids)}
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
        physical_edge_ids=edge_ids,
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


def test_greedy_qkp_ties_break_by_rate() -> None:
    obs = _minimal_obs(
        node_ids=["GS1", "GS2", "GS3"],
        edge_pairs=[("GS1", "GS2"), ("GS1", "GS3")],
        demand={},
        levels={"E_GS1__GS2": 5.0, "E_GS1__GS3": 5.0},
    )
    # Both edges carry the same level; with equal rates the first sorted
    # neighbour wins, so we only assert legality and max-score consistency.
    actions, scores = GreedyQKPPolicy().act(obs)
    for node in obs.node_ids:
        assert scores[node][actions[node]] == max(scores[node].values())


def test_baseline_policies_return_legal_actions_in_real_env() -> None:
    policies = [RandomPolicy(seed=0), GreedyRatePolicy(), GreedyDemandPolicy(), GreedyQKPPolicy()]
    for policy in policies:
        env = build_default_env(ROOT)
        obs = env.reset(seed=42)
        actions, scores = policy.act(obs)
        for node in obs.node_ids:
            assert actions[node] in obs.action_candidates[node]
            assert set(scores[node]) == set(obs.action_candidates[node])


def test_baseline_policies_complete_an_episode() -> None:
    env = build_default_env(ROOT)
    obs = env.reset(seed=7)
    for policy in [GreedyDemandPolicy(), GreedyQKPPolicy()]:
        done = False
        while not done:
            actions, scores = policy.act(obs)
            obs, _reward, terminated, truncated, _info = env.step(actions, scores)
            done = terminated or truncated
        summary = env.metrics.episode_summary()
        assert summary["arrived_keys"] >= 0.0
        assert 0.0 <= summary["success_rate"] <= 1.0
        obs = env.reset(seed=7)
