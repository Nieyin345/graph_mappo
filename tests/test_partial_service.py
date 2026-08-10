"""Tests for per-hop partial key service (A-plan).

A request is served as much as the bottleneck hop of a usable path allows;
the remainder keeps its ``served_amount`` and is re-queued. All-or-nothing
semantics of ``can_serve``/``consume_for_request`` stay unchanged.
"""
from __future__ import annotations

from dataclasses import replace

from qkd_rl.core.types import Edge, KeyRequest, LinkType
from qkd_rl.env.qkp import LinkQKPPool
from qkd_rl.env.request import RequestQueue
from qkd_rl.env.routing import RoutingPolicy


def _make_env():
    edges = [
        Edge("E1", "GS_A", "GS_B", LinkType.GS_SAT),
        Edge("E2", "GS_B", "GS_C", LinkType.GS_SAT),
        Edge("E3", "GS_C", "GS_D", LinkType.GS_SAT),
    ]
    config = {
        "capacity": {"default": 1.0e6, "per_link_type": {}, "per_edge": {}},
        "initial_level": 0.0,
        "key_ttl_steps": 60,
        "overflow_policy": "discard",
    }
    qkp = LinkQKPPool(edges, config)
    qkp.reset(0)
    routing = RoutingPolicy(edges, {})
    return qkp, routing


def _req(amount: float = 10.0, deadline: int = 100) -> KeyRequest:
    return KeyRequest("REQ_1", "GS_A", "GS_C", amount, arrival_t=0, deadline_t=deadline)


def test_partial_serve_consumes_bottleneck_amount():
    qkp, routing = _make_env()
    qkp.add_keys("E1", 4.0, 0)
    qkp.add_keys("E2", 4.0, 0)
    req = _req(amount=10.0)

    consumed = routing.partial_consume_for_request(req, qkp, t=1)

    assert consumed == 4.0
    assert qkp.get_level("E1") == 0.0
    assert qkp.get_level("E2") == 0.0
    # Nothing left to serve a second time.
    assert routing.partial_consume_for_request(req, qkp, t=1) == 0.0


def test_partial_serve_uses_remaining_not_full_amount():
    qkp, routing = _make_env()
    qkp.add_keys("E1", 4.0, 0)
    qkp.add_keys("E2", 4.0, 0)
    req = replace(_req(amount=10.0), served_amount=8.0)  # previously partially served

    consumed = routing.partial_consume_for_request(req, qkp, t=1)

    assert consumed == 2.0
    assert qkp.get_level("E1") == 2.0
    assert qkp.get_level("E2") == 2.0


def test_queue_partial_service_accumulates_to_completion():
    qkp, routing = _make_env()
    queue = RequestQueue()
    req = _req(amount=10.0)
    queue.add_arrivals([req])

    qkp.add_keys("E1", 4.0, 0)
    qkp.add_keys("E2", 4.0, 0)
    step1 = queue.serve(qkp, routing, t=1)
    assert step1.served_keys == 4.0
    assert step1.served_requests == []
    pending = queue.get_pending()
    assert len(pending) == 1 and pending[0].served_amount == 4.0
    assert step1.waiting_keys == 6.0
    assert queue.demand_by_pair() == {("GS_A", "GS_C"): 6.0}

    qkp.add_keys("E1", 4.0, 0)
    qkp.add_keys("E2", 4.0, 0)
    step2 = queue.serve(qkp, routing, t=2)
    assert step2.served_keys == 4.0
    assert queue.get_pending()[0].served_amount == 8.0
    assert step2.waiting_keys == 2.0

    qkp.add_keys("E1", 4.0, 0)
    qkp.add_keys("E2", 4.0, 0)
    step3 = queue.serve(qkp, routing, t=3)
    assert step3.served_keys == 2.0  # only the remaining 2 keys are consumed
    assert len(step3.served_requests) == 1
    assert step3.served_requests[0].served_amount == 10.0
    assert queue.get_pending() == []


def test_no_positive_key_path_waits_without_failure():
    qkp, routing = _make_env()
    queue = RequestQueue()
    req = _req(amount=10.0, deadline=100)
    queue.add_arrivals([req])

    result = queue.serve(qkp, routing, t=1)
    expired = queue.expire(t=1)

    assert result.served_keys == 0.0
    assert result.failed_keys == 0.0
    assert expired == []
    assert len(queue.get_pending()) == 1
    assert queue.get_pending()[0].served_amount == 0.0


def test_partial_serve_then_expire_counts_remaining_only():
    qkp, routing = _make_env()
    queue = RequestQueue()
    req = replace(_req(amount=10.0, deadline=5), served_amount=6.0)
    queue.add_arrivals([req])

    # Fully served at the deadline step: completes, so nothing expires.
    qkp.add_keys("E1", 4.0, 0)
    qkp.add_keys("E2", 4.0, 0)
    result = queue.serve(qkp, routing, t=5)
    expired = queue.expire(t=5)

    assert result.served_keys == 4.0
    assert result.failed_keys == 0.0
    assert len(result.served_requests) == 1
    assert result.served_requests[0].served_amount == 10.0
    assert expired == []
    assert queue.get_pending() == []


def test_partial_serve_without_keys_then_expire_counts_remaining():
    qkp, routing = _make_env()
    queue = RequestQueue()
    req = replace(_req(amount=10.0, deadline=5), served_amount=6.0)
    queue.add_arrivals([req])

    result = queue.serve(qkp, routing, t=5)
    expired = queue.expire(t=5)

    assert result.served_keys == 0.0
    assert len(expired) == 1
    assert expired[0].served_amount == 6.0


def test_can_serve_keeps_all_or_nothing_semantics():
    qkp, routing = _make_env()
    req = _req(amount=10.0)

    qkp.add_keys("E1", 10.0, 0)
    qkp.add_keys("E2", 10.0, 0)
    assert routing.can_serve(req, qkp, t=1) is True
    assert routing.consume_for_request(req, qkp, t=1) is True
    assert qkp.get_level("E1") == 0.0

    # Bottleneck below the full amount: all-or-nothing fails, partial works.
    qkp.add_keys("E1", 10.0, 0)
    qkp.add_keys("E2", 5.0, 0)
    assert routing.can_serve(req, qkp, t=2) is False
    assert routing.consume_for_request(req, qkp, t=2) is False
    assert routing.partial_consume_for_request(req, qkp, t=2) == 5.0


def test_find_request_path_required_level_filter():
    qkp, routing = _make_env()
    qkp.add_keys("E1", 3.0, 0)
    qkp.add_keys("E2", 3.0, 0)
    req = _req(amount=10.0)

    # Full-amount BFS requires each hop >= 10 -> no path.
    assert routing.find_request_path(req, qkp=qkp) is None
    # Partial BFS only requires positive stock -> path exists.
    path = routing.find_request_path(req, qkp=qkp, required_level=0.0)
    assert path == ["E1", "E2"]

def test_hop_distance_precomputed_and_disconnected():
    qkp, routing = _make_env()
    assert routing.hop_distance("GS_A", "GS_C") == 2
    assert routing.hop_distance("GS_A", "GS_D") == 3
    assert routing.hop_distance("GS_A", "GS_A") == 0
    assert routing.hop_distance("GS_A", "NOPE") == RoutingPolicy.DISCONNECTED


def test_serve_prioritizes_fewer_hops_on_shared_bottleneck():
    """Two requests share edge E1; the 1-hop request is served (completed)
    first, the 2-hop request gets only the leftover partial keys."""
    edges = [
        Edge("E1", "GS_A", "GS_B", LinkType.GS_SAT),
        Edge("E2", "GS_B", "GS_C", LinkType.GS_SAT),
    ]
    config = {
        "capacity": {"default": 1.0e6, "per_link_type": {}, "per_edge": {}},
        "initial_level": 0.0,
        "key_ttl_steps": 60,
        "overflow_policy": "discard",
    }
    qkp = LinkQKPPool(edges, config)
    qkp.reset(0)
    routing = RoutingPolicy(edges, {})
    qkp.add_keys("E1", 12.0, 0)
    qkp.add_keys("E2", 12.0, 0)

    queue = RequestQueue()
    two_hop = KeyRequest("REQ_2HOP", "GS_A", "GS_C", 10.0, 0, 100)
    one_hop = KeyRequest("REQ_1HOP", "GS_A", "GS_B", 10.0, 0, 100)
    # Insert the 2-hop request first to prove ordering is by hops, not FIFO.
    queue.add_arrivals([two_hop, one_hop])

    result = queue.serve(qkp, routing, t=1)

    assert [r.request_id for r in result.served_requests] == ["REQ_1HOP"]
    pending = queue.get_pending()
    assert [r.request_id for r in pending] == ["REQ_2HOP"]
    assert pending[0].served_amount == 2.0  # only the E1 leftover after REQ_1HOP


def test_serve_prioritizes_smaller_remaining_when_hops_equal():
    """Same deadline and hop count: the smaller remaining demand is served
    first so more requests complete."""
    # Single shared edge: 10 keys total, one request needs 3, the other 10.
    edges = [
        Edge("E1", "GS_A", "GS_B", LinkType.GS_SAT),
    ]
    config = {
        "capacity": {"default": 1.0e6, "per_link_type": {}, "per_edge": {}},
        "initial_level": 0.0,
        "key_ttl_steps": 60,
        "overflow_policy": "discard",
    }
    qkp = LinkQKPPool(edges, config)
    qkp.reset(0)
    routing = RoutingPolicy(edges, {})
    qkp.add_keys("E1", 10.0, 0)

    queue = RequestQueue()
    small = KeyRequest("REQ_SMALL", "GS_A", "GS_B", 3.0, 0, 100)
    big = KeyRequest("REQ_BIG", "GS_A", "GS_B", 10.0, 0, 100)
    queue.add_arrivals([big, small])

    result = queue.serve(qkp, routing, t=1)

    assert [r.request_id for r in result.served_requests] == ["REQ_SMALL"]
    pending = queue.get_pending()
    assert [r.request_id for r in pending] == ["REQ_BIG"]
    assert pending[0].served_amount == 7.0
