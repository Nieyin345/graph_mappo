from __future__ import annotations

from qkd_rl.core.types import KeyRequest
from qkd_rl.env.factory import build_default_env, load_default_config
from qkd_rl.env.request import RequestHistoryTracker


def test_active_request_creates_gs_pair_demand_edge():
    config = load_default_config(".")
    env = build_default_env(".")
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
    assert len(obs.edge_features[demand_row_index]) == physical_dim + demand_dim
    assert obs.edge_features[demand_row_index][:physical_dim] == [0.0] * physical_dim
    assert obs.edge_features[demand_row_index][physical_dim] == 0.25


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

