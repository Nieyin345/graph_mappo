from __future__ import annotations

import math

import pytest

from qkd_rl.core.types import Edge, KeyRequest, Node, NodeType, LinkType
from qkd_rl.core.config import ConfigValidator
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.env.factory import load_default_config
from qkd_rl.env.relay_importance import compute_relay_importance
from tests.helpers import ROOT, build_test_env, point_config_to_h5
from qkd_rl.env.request import RequestHistoryTracker


def live_relay_importance(env, config, physical_edge_ids=None, all_edge_ids=None):
    """Call the production relay-importance function with the env's live state.

    ``GraphBuilder.build_edge_features`` uses exactly this call, so these tests
    exercise the code path the model and the reward actually see.
    """
    relay_cfg = config["features"]["edge"]["relay_importance"]
    return compute_relay_importance(
        node_ids=env.graph_builder._node_ids_list,
        physical_edge_ids=(
            list(env.graph_builder._edge_list) if physical_edge_ids is None
            else list(physical_edge_ids)
        ),
        pending_requests=env.requests.get_pending(),
        qkp_snapshot=env.qkp.snapshot(),
        qkp_capacity=env.qkp.capacities,
        t=env.t,
        max_path_links=int(relay_cfg.get("max_path_links", 3)),
        hop_decay_factor=float(relay_cfg.get("hop_decay_factor", 0.25)),
        capacity_strength=float(relay_cfg.get("capacity_decay_strength", 1.0)),
        min_scarcity=float(relay_cfg.get("min_scarcity", 0.0)),
        wait_urgency_tau_ratio=float(relay_cfg.get("wait_urgency_tau_ratio", 0.8)),
        ignore_consumption=bool(relay_cfg.get("ignore_consumption", False)),
        include_stocked_unavailable=False,
        all_edge_ids=all_edge_ids,
    )


def _chain_topology():
    """S and D are GS endpoints; relays give candidate edges at 2/3/4 total hops.

    ``compute_relay_importance`` scores an edge with
    ``total = min(d(S,u),d(S,v)) + min(d(D,u),d(D,v)) + 1``, so the hop counts
    below follow from that definition (verified by hand on this graph).
    """
    nodes = [
        Node("S", NodeType.GS), Node("D", NodeType.GS),
        Node("rA", NodeType.HAP), Node("rB", NodeType.HAP),
        Node("rC", NodeType.HAP), Node("rF", NodeType.HAP),
    ]
    pairs = [
        ("S", "rA"), ("S", "rB"), ("D", "rA"), ("D", "rB"),
        ("rA", "rB"), ("rA", "rC"), ("rC", "rF"), ("D", "rF"),
    ]
    edges = [
        Edge(edge_id=f"E_{u}__{v}", src=u, dst=v, link_type=LinkType.HAP_SAT)
        for u, v in pairs
    ]
    return nodes, edges


def test_active_request_creates_gs_pair_demand_edge():
    config = load_default_config(".")
    env = build_test_env(".")
    env.reset()
    request = KeyRequest(
        request_id="REQ_TEST",
        src_gs="GS_001",
        dst_gs="GS_002",
        amount=250.0,
        arrival_t=env.t,
        deadline_t=env.t + 12,
    )
    env.requests.add_arrivals([request])
    env.request_history.record_arrivals([request], env.t)

    obs = env._build_observation()
    demand_edge_id = "D_GS_001__GS_002"
    demand_row_index = obs.edge_ids.index(demand_edge_id)
    physical_dim = config["features"]["dims"]["physical_edge_dim_resolved"]
    demand_dim = config["features"]["dims"]["demand_edge_dim_resolved"]

    assert demand_edge_id in obs.demand_edge_ids
    # edge_features is a numpy (n_edges, physical_dim + demand_dim) array.
    assert obs.edge_features.shape[1] == physical_dim + demand_dim
    assert bool((obs.edge_features[demand_row_index, :physical_dim] == 0.0).all())
    demand_cfg = config["features"]["demand_edge"]
    if demand_cfg.get("normalize_amount_log1p", False):
        reference = float(demand_cfg.get("normalize_amount_reference", 1000.0))
        expected = math.log1p(250.0) / math.log1p(reference)
    else:
        expected = 250.0 / float(demand_cfg.get("normalize_amount_by", 1000.0))
    assert obs.edge_features[demand_row_index, physical_dim] == pytest.approx(expected)


def test_request_history_tracks_pair_amounts_by_window():
    history = RequestHistoryTracker()
    old_request = KeyRequest("REQ_OLD", "GS_001", "GS_002", 10.0, 0, 12)
    recent_request = KeyRequest("REQ_RECENT", "GS_001", "GS_002", 20.0, 20, 32)

    history.record_arrivals([old_request], 0)
    history.record_arrivals([recent_request], 20)
    history.record_served([recent_request], 25)

    pair = ("GS_001", "GS_002")
    assert history.sum(pair, "arrived", t=30, window=15) == 20.0
    assert history.sum(pair, "arrived", t=30, window=60) == 30.0
    assert history.sum(pair, "served", t=30, window=15) == 20.0


def test_demand_edge_removed_after_success_or_expiry():
    config = point_config_to_h5(load_default_config(ROOT))
    config["requests"]["deadline_steps"] = 12
    config["features"]["demand_edge"]["build_mode"] = "active_pairs"
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    env.reset()

    demand_id = "D_GS_001__GS_002"
    tiny = KeyRequest("REQ_OK", "GS_001", "GS_002", 1.0, env.t, env.t + 12)
    env.requests.add_arrivals([tiny])
    obs = env._build_observation()
    assert demand_id in obs.demand_edge_ids

    env.requests.serve(env.qkp, env.routing, env.t)
    env.requests.expire(env.t)
    obs = env._build_observation()
    assert demand_id not in obs.demand_edge_ids

    big = KeyRequest("REQ_FAIL", "GS_001", "GS_002", 1.0e9, env.t, env.t + 12)
    env.requests.add_arrivals([big])
    env.t = 24
    obs = env._build_observation()
    assert demand_id in obs.demand_edge_ids
    env.requests.serve(env.qkp, env.routing, env.t)
    expired = env.requests.expire(env.t)
    assert len(expired) == 1
    obs = env._build_observation()
    assert demand_id not in obs.demand_edge_ids


def test_relay_importance_favors_scarce_hop_and_clears_when_path_full():
    """A relay edge loses importance once its pool fills up.

    GS_001/GS_002 both reach the single HAP and the single SAT, so
    ``E_GS_001__HAP_001`` and ``E_GS_002__HAP_001`` are the two 2-hop relay
    candidates for that pair.
    """
    config = point_config_to_h5(load_default_config(ROOT))
    config["features"]["edge"]["include_relay_importance"] = True
    config["qkp"]["initial_level"] = 0.0
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    env.reset()

    path = ["E_GS_001__HAP_001", "E_GS_002__HAP_001"]
    req = KeyRequest("REQ_RELAY", "GS_001", "GS_002", 100000.0, env.t, env.t + 12)
    env.requests.add_arrivals([req])

    importance_empty = live_relay_importance(env, config, physical_edge_ids=path)
    assert importance_empty.get(path[0], 0.0) > 0.0
    assert importance_empty.get(path[1], 0.0) > 0.0

    env.qkp.add_keys(path[0], env.qkp.get_capacity(path[0]), env.t)
    importance_one_full = live_relay_importance(env, config, physical_edge_ids=path)
    assert importance_one_full.get(path[1], 0.0) > importance_one_full.get(path[0], 0.0)

    env.qkp.add_keys(path[1], env.qkp.get_capacity(path[1]), env.t)
    importance_full = live_relay_importance(env, config, physical_edge_ids=path)
    assert importance_full.get(path[0], 0.0) == 0.0
    assert importance_full.get(path[1], 0.0) == 0.0


def test_demand_edge_mean_wait_time_tracks_queue_age():
    config = point_config_to_h5(load_default_config(ROOT))
    config["requests"]["deadline_steps"] = 100
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    env.reset()

    t = env.t
    old = KeyRequest("REQ_OLD", "GS_001", "GS_002", 20.0, t - 20, t + 80)
    recent = KeyRequest("REQ_RECENT", "GS_001", "GS_002", 10.0, t - 10, t + 90)
    env.requests.add_arrivals([old, recent])

    obs = env._build_observation()
    row = obs.edge_ids.index("D_GS_001__GS_002")
    physical_dim = config["features"]["dims"]["physical_edge_dim_resolved"]
    # pending_amount, pending_count, min_deadline, mean_deadline, mean_wait_time
    wait_col = physical_dim + 4
    assert obs.edge_features[row, wait_col] == pytest.approx(15.0 / 100.0)

    env.t += 10
    obs = env._build_observation()
    row = obs.edge_ids.index("D_GS_001__GS_002")
    assert obs.edge_features[row, wait_col] == pytest.approx(25.0 / 100.0)


def test_wait_buckets_move_forward_over_time():
    config = point_config_to_h5(load_default_config(ROOT))
    config["requests"]["deadline_steps"] = 100
    config["features"]["demand_edge"]["wait_bucket_count"] = 10
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    env.reset()

    req = KeyRequest("REQ_BUCKET", "GS_001", "GS_002", 100.0, env.t, env.t + 100)
    env.requests.add_arrivals([req])

    physical_dim = config["features"]["dims"]["physical_edge_dim_resolved"]
    # pending_amount, pending_count, min_deadline, mean_deadline, mean_wait,
    # priority_sum, then wait buckets.
    buckets_start = physical_dim + 6
    edge_id = "D_GS_001__GS_002"

    obs = env._build_observation()
    row = obs.edge_ids.index(edge_id)
    assert obs.edge_features[row, buckets_start] > 0.0
    assert obs.edge_features[row, buckets_start + 9] == 0.0

    env.t += 50
    obs = env._build_observation()
    row = obs.edge_ids.index(edge_id)
    assert obs.edge_features[row, buckets_start] == 0.0
    assert obs.edge_features[row, buckets_start + 5] > 0.0

    env.t += 45
    obs = env._build_observation()
    row = obs.edge_ids.index(edge_id)
    assert obs.edge_features[row, buckets_start + 5] == 0.0
    assert obs.edge_features[row, buckets_start + 9] > 0.0


def test_relay_importance_capacity_decay_strength_changes_ranking():
    """A stronger capacity exponent lowers a half-full relay edge's importance.

    initial_level=0 keeps the second 2-hop candidate empty (scarcity 1.0), so
    the normalized value of the half-filled edge is just 0.5**strength.
    """
    def importance_with_strength(strength: float) -> float:
        config = point_config_to_h5(load_default_config(ROOT))
        config["features"]["edge"]["include_relay_importance"] = True
        config["features"]["edge"]["relay_importance"]["capacity_decay_strength"] = strength
        config["qkp"]["initial_level"] = 0.0
        ConfigValidator().validate(config)
        env = build_env_from_config(config)
        env.reset()
        path = ["E_GS_001__HAP_001", "E_GS_002__HAP_001"]
        env.requests.add_arrivals(
            [KeyRequest("REQ_STRENGTH", "GS_001", "GS_002", 1000.0, env.t, env.t + 960)]
        )
        env.qkp.add_keys(path[0], env.qkp.get_capacity(path[0]) * 0.5, env.t)
        return float(live_relay_importance(env, config, physical_edge_ids=path).get(path[0], 0.0))

    weak = importance_with_strength(1.0)
    strong = importance_with_strength(2.0)
    assert weak > 0.0
    assert strong < weak


def test_relay_importance_hop_decay_uses_stronger_quarter_factor():
    """Edge weight decays by hop_decay_factor per extra hop beyond the minimum.

    Chain topology: S/D are the demand endpoints; ``E_S__rA`` sits on a 2-hop
    path, ``E_rA__rB`` on a 3-hop path, ``E_rC__rF`` on a 4-hop path. All pools
    are empty, so the only difference between their weights is the hop decay.
    """
    nodes, edges = _chain_topology()
    edge_ids = [edge.edge_id for edge in edges]
    capacities = {edge_id: 1000.0 for edge_id in edge_ids}
    request = KeyRequest("REQ_HOP", "S", "D", 1000.0, 0, 100)

    importance = compute_relay_importance(
        node_ids=[node.node_id for node in nodes],
        physical_edge_ids=edge_ids,
        pending_requests=[request],
        qkp_snapshot={edge_id: 0.0 for edge_id in edge_ids},
        qkp_capacity=capacities,
        t=0,
        max_path_links=4,
        hop_decay_factor=0.25,
        include_stocked_unavailable=False,
    )

    two_hop = importance["E_S__rA"]
    three_hop = importance["E_rA__rB"]
    four_hop = importance["E_rC__rF"]
    assert two_hop > three_hop > four_hop > 0.0
    assert three_hop / two_hop == pytest.approx(0.25)
    assert four_hop / three_hop == pytest.approx(0.25)


def test_relay_importance_is_scoped_to_active_edges() -> None:
    config = point_config_to_h5(load_default_config(ROOT))
    config["features"]["edge"]["include_relay_importance"] = True
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    env.reset()
    env.requests.add_arrivals(
        [KeyRequest("REQ_ACTIVE", "GS_001", "GS_002", 1000.0, env.t, env.t + 960)]
    )
    obs = env._build_observation()
    assert set(env.graph_builder.last_relay_importance) <= set(obs.physical_edge_ids)
