from __future__ import annotations

import math

import pytest

from qkd_rl.core.types import KeyRequest
from qkd_rl.core.config import ConfigValidator
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.env.factory import load_default_config
from tests.helpers import ROOT, build_test_env, point_config_to_h5
from qkd_rl.env.request import RequestHistoryTracker


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
    config = point_config_to_h5(load_default_config(ROOT))
    config["features"]["edge"]["include_relay_importance"] = True
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    env.reset()

    pair, path = next(
        (p, [e for e, total in entries if total == 2][:2])
        for p, entries in env.graph_builder._relay_edge_info.items()
        if any(total == 2 for _, total in entries)
    )
    req = KeyRequest("REQ_RELAY", pair[0], pair[1], 100000.0, env.t, env.t + 12)
    env.requests.add_arrivals([req])
    demand_stats = env.requests.stats_by_pair(
        env.t,
        env.request_history,
        None,
        wait_bucket_count=int(config["features"]["demand_edge"]["wait_bucket_count"]),
        deadline_steps=int(config["requests"]["deadline_steps"]),
    )

    importance_empty = env.graph_builder._relay_importance(
        env.requests, demand_stats, env_state=env._build_state()
    )
    assert importance_empty.get(path[0], 0.0) > 0.0
    assert importance_empty.get(path[1], 0.0) > 0.0

    env.qkp.add_keys(path[0], env.qkp.get_capacity(path[0]), env.t)
    importance_one_full = env.graph_builder._relay_importance(
        env.requests, demand_stats, env_state=env._build_state()
    )
    assert importance_one_full.get(path[1], 0.0) > importance_one_full.get(path[0], 0.0)

    env.qkp.add_keys(path[1], env.qkp.get_capacity(path[1]), env.t)
    importance_full = env.graph_builder._relay_importance(
        env.requests, demand_stats, env_state=env._build_state()
    )
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


def test_capacity_decay_strength_is_configurable():
    def importance_with_strength(strength: float) -> float:
        config = point_config_to_h5(load_default_config(ROOT))
        config["features"]["edge"]["include_relay_importance"] = True
        config["features"]["edge"]["relay_importance"]["capacity_decay_strength"] = strength
        ConfigValidator().validate(config)
        env = build_env_from_config(config)
        env.reset()
        pair, entries = next(
            (p, es) for p, es in env.graph_builder._relay_edge_info.items() if es
        )
        assert len(entries) >= 2
        edge_id = entries[0][0]
        req = KeyRequest("REQ_STRENGTH", pair[0], pair[1], 1000.0, env.t, env.t + 960)
        env.requests.add_arrivals([req])
        env.qkp.add_keys(edge_id, env.qkp.get_capacity(edge_id) * 0.5, env.t)
        stats = env.requests.stats_by_pair(
            env.t,
            env.request_history,
            None,
            wait_bucket_count=int(config["features"]["demand_edge"]["wait_bucket_count"]),
            deadline_steps=int(config["requests"]["deadline_steps"]),
        )
        importance = env.graph_builder._relay_importance(
            env.requests, stats, env_state=env._build_state()
        )
        return float(importance.get(edge_id, 0.0))

    weak = importance_with_strength(1.0)
    strong = importance_with_strength(2.0)
    assert weak > 0.0
    assert strong < weak


def test_relay_importance_hop_decay_uses_stronger_quarter_factor():
    config = point_config_to_h5(load_default_config(ROOT))
    config["features"]["edge"]["include_relay_importance"] = True
    config["features"]["edge"]["relay_importance"]["hop_decay_factor"] = 0.25
    config["qkp"]["initial_level"] = 0.0
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    env.reset()

    pair = ("GS_001", "GS_002")
    edge_ids = list(env.graph_builder._edge_list[:3])
    env.graph_builder._relay_edge_info[pair] = [
        (edge_ids[0], 2),
        (edge_ids[1], 3),
        (edge_ids[2], 4),
    ]
    req = KeyRequest("REQ_HOP_DECAY", pair[0], pair[1], 1000.0, env.t, env.t + 960)
    env.requests.add_arrivals([req])
    stats = env.requests.stats_by_pair(
        env.t,
        env.request_history,
        None,
        wait_bucket_count=int(config["features"]["demand_edge"]["wait_bucket_count"]),
        deadline_steps=int(config["requests"]["deadline_steps"]),
    )

    importance = env.graph_builder._relay_importance(
        env.requests, stats, env_state=env._build_state()
    )
    raw = [float(importance[edge_id]) for edge_id in edge_ids]

    assert raw[1] / raw[0] == pytest.approx(0.25)
    assert raw[2] / raw[1] == pytest.approx(0.25)


def test_relay_importance_is_scoped_to_active_edges() -> None:
    config = point_config_to_h5(load_default_config(ROOT))
    config["features"]["edge"]["include_relay_importance"] = True
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    env.reset()
    pair = next(iter(env.graph_builder._relay_edge_info))
    req = KeyRequest("REQ_ACTIVE", pair[0], pair[1], 1000.0, env.t, env.t + 960)
    env.requests.add_arrivals([req])
    obs = env._build_observation()
    assert set(env.graph_builder.last_relay_importance) <= set(obs.physical_edge_ids)

