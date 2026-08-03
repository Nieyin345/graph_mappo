"""Tests for request queue expiry/serving accounting, generation, and history windows."""

from __future__ import annotations

from qkd_rl.core.types import KeyRequest
from tests.helpers import build_test_env
from qkd_rl.env.request import RequestGenerator, RequestHistoryTracker


def test_deadline_reached_request_is_expired_not_failed_by_serve():
    env = build_test_env(".")
    env.reset()
    req = KeyRequest("REQ_X", "GS_001", "GS_002", 1.0e9, env.t, env.t - 1)
    env.requests.add_arrivals([req])

    serve_result = env.requests.serve(env.qkp, env.routing, env.t)
    expired = env.requests.expire(env.t)

    assert serve_result.failed_keys == 0.0
    assert len(expired) == 1
    assert expired[0].request_id == "REQ_X"
    assert len(env.requests.get_pending()) == 0


def test_servable_request_is_served_and_removed():
    env = build_test_env(".")
    env.reset()
    # A tiny request that the initial QKP can serve immediately.
    req = KeyRequest("REQ_Y", "GS_001", "GS_002", 1.0, env.t, env.t + 12)
    env.requests.add_arrivals([req])

    serve_result = env.requests.serve(env.qkp, env.routing, env.t)
    expired = env.requests.expire(env.t)

    assert serve_result.served_keys == 1.0
    assert len(expired) == 0


def test_request_generator_priority_modes():
    common = {"arrival_rate": 1.0, "amount_mean": 10.0, "deadline_steps": 12}
    uniform = RequestGenerator(["GS_001", "GS_002"], {**common, "priority_mode": "uniform"}, seed=0)
    reqs = uniform.generate(0)
    assert reqs and all(req.priority == 1.0 for req in reqs)

    random_gen = RequestGenerator(["GS_001", "GS_002"], {**common, "priority_mode": "random"}, seed=1)
    reqs_random = random_gen.generate(0)
    assert reqs_random and all(1.0 <= req.priority <= 3.0 for req in reqs_random)


def test_history_window_sums_and_node_sums():
    history = RequestHistoryTracker()
    history.record_arrivals([KeyRequest("R1", "GS_001", "GS_002", 10.0, 0, 12)], 0)
    history.record_arrivals([KeyRequest("R2", "GS_001", "GS_003", 20.0, 20, 32)], 20)
    history.record_served([KeyRequest("R2", "GS_001", "GS_003", 20.0, 20, 32)], 25)

    assert history.sum(("GS_001", "GS_002"), "arrived", t=30, window=15) == 0.0
    assert history.sum(("GS_001", "GS_002"), "arrived", t=30, window=60) == 10.0
    assert history.sum(("GS_001", "GS_003"), "served", t=30, window=15) == 20.0
    assert history.sum_for_node("GS_001", "arrived", t=30, window=60) == 30.0
    assert history.sum_for_node("GS_003", "arrived", t=30, window=60) == 20.0
    assert ("GS_001", "GS_003") in history.recent_pairs(t=30, windows=[15])
    assert ("GS_001", "GS_002") not in history.recent_pairs(t=30, windows=[15])
