"""Tests for reward normalization robustness and the generation/clip knobs."""

from __future__ import annotations

import pytest

from qkd_rl.env.reward import RewardFunction


def _reward_fn(
    normalize_window: int = 5,
    normalize_floor: float = 1000.0,
    clip_abs: float = 0.0,
    generated_weight: float = 0.0,
    overflow_weight: float = 0.1,
    waiting_stock_weight: float = 0.01,
    served_reference: float = 0.0,
) -> RewardFunction:
    return RewardFunction(
        {
            "served_weight": 1.0,
            "served_reference": served_reference,
            "generated_weight": generated_weight,
            "failed_weight": 2.0,
            "waiting_weight": 0.05,
            "overflow_weight": overflow_weight,
            "expired_key_weight": 0.1,
            "conflict_weight": 0.01,
            "normalize_by_arrived_demand": True,
            "normalize_window": normalize_window,
            "normalize_floor": normalize_floor,
            "clip_abs": clip_abs,
            "waiting_stock_weight": waiting_stock_weight,
        }
    )


def _compute(
    fn: RewardFunction,
    arrived: float,
    served_keys: float = 0.0,
    failed_keys: float = 0.0,
    waiting_keys: float = 0.0,
    waiting_delta: float = 0.0,
    added_keys: float = 0.0,
    overflow_keys: float = 0.0,
):
    from qkd_rl.env.action_resolver import ResolvedAction
    from qkd_rl.env.qkp import LinkQKPPool
    from qkd_rl.env.request import ServeResult
    from qkd_rl.env.routing import AllocationResult

    return fn.compute(
        serve_result=ServeResult([], [], [], served_keys, waiting_keys, failed_keys),
        allocation=AllocationResult(added_keys=added_keys, overflow_keys=overflow_keys),
        expired_requests=[],
        expired_keys=0.0,
        resolved_action=ResolvedAction(activated_edges=[], rejected_actions={}, illegal_actions={}, conflict_count=0),
        qkp=LinkQKPPool([], {"capacity": {"default": 1.0}, "initial_level": 0.0}),
        arrived_keys=arrived,
        served_keys=served_keys,
        waiting_keys=waiting_keys,
        waiting_delta=waiting_delta,
    )


def test_zero_arrival_slot_does_not_blow_up_penalty() -> None:
    fn = _reward_fn(normalize_window=5)
    for _ in range(5):
        _compute(fn, arrived=100000.0, failed_keys=0.0)
    normal_penalty = _compute(fn, arrived=100000.0, failed_keys=50000.0).failed_penalty
    zero_penalty = _compute(fn, arrived=0.0, failed_keys=50000.0).failed_penalty
    assert zero_penalty == pytest.approx(normal_penalty, rel=0.5)


def test_window_mean_matches_expected_scale() -> None:
    fn = _reward_fn(normalize_window=5)
    for _ in range(5):
        _compute(fn, arrived=100000.0, failed_keys=0.0)
    detail = _compute(fn, arrived=100000.0, failed_keys=50000.0)
    assert detail.failed_penalty == pytest.approx(1.0, rel=0.6)


def test_normalize_floor_caps_cold_start_amplification() -> None:
    # After reset the window is empty; a zero-arrival first step used to fall
    # back to denom=1. The floor caps the amplification: denom >= floor.
    fn = _reward_fn(normalize_window=3, normalize_floor=1000.0)
    fn.reset()
    detail = _compute(fn, arrived=0.0, failed_keys=50000.0)
    assert detail.failed_penalty == pytest.approx(2.0 * 50000.0 / 1000.0, rel=0.01)


def test_reset_clears_window_with_floor_1() -> None:
    fn = _reward_fn(normalize_window=3, normalize_floor=1.0)
    for _ in range(3):
        _compute(fn, arrived=100000.0, failed_keys=0.0)
    fn.reset()
    detail = _compute(fn, arrived=0.0, failed_keys=50000.0)
    assert detail.failed_penalty == pytest.approx(100000.0, rel=0.6)


def test_fixed_served_reference_is_time_invariant() -> None:
    fn = _reward_fn(served_reference=100000.0)
    fn.reset()
    early = _compute(fn, arrived=0.0, served_keys=20000.0)
    for _ in range(20):
        _compute(fn, arrived=100000.0, served_keys=0.0)
    late = _compute(fn, arrived=100000.0, served_keys=20000.0)
    assert early.served_reward == pytest.approx(late.served_reward, rel=1e-9)


def test_generated_reward_scales_with_added_keys() -> None:
    fn = _reward_fn(generated_weight=0.01)
    for _ in range(5):
        _compute(fn, arrived=100000.0, failed_keys=0.0)
    detail = _compute(fn, arrived=100000.0, added_keys=2000000.0)
    # denom ~= 100000 -> generated_reward ~= 0.01 * 2e6 / 1e5 = 0.2
    assert detail.generated_reward == pytest.approx(0.2, rel=0.6)
    # A full pool adds nothing -> no generation reward.
    detail0 = _compute(fn, arrived=100000.0, added_keys=0.0)
    assert detail0.generated_reward == pytest.approx(0.0, abs=1e-9)


def test_overflow_weight_zero_kills_raw_overflow_penalty() -> None:
    # The spike root cause: raw overflow (rate x slot minus capacity) must not
    # be penalized; wasting an activation on a full pool is neutral instead.
    fn = _reward_fn(overflow_weight=0.0)
    for _ in range(5):
        _compute(fn, arrived=100000.0, failed_keys=0.0)
    detail = _compute(fn, arrived=100000.0, overflow_keys=2_000_000_000.0)
    assert detail.overflow_penalty == pytest.approx(0.0, abs=1e-9)


def test_clip_abs_bounds_total_and_components() -> None:
    fn = _reward_fn(clip_abs=500.0)
    for _ in range(5):
        _compute(fn, arrived=100000.0, failed_keys=0.0)
    # Huge waiting increase gives 0.05 * 5e9 / 1e5 = 2500, clipped to 500.
    detail = _compute(fn, arrived=100000.0, waiting_keys=5_000_000_000.0, waiting_delta=5_000_000_000.0)
    assert detail.total >= -500.0 - 1e-6
    assert detail.total <= 500.0 + 1e-6
    assert detail.waiting_penalty >= -500.0 - 1e-6


def test_clip_disabled_by_default() -> None:
    fn = _reward_fn(clip_abs=0.0)
    for _ in range(5):
        _compute(fn, arrived=100000.0, failed_keys=0.0)
    detail = _compute(fn, arrived=100000.0, waiting_keys=5_000_000_000.0, waiting_delta=5_000_000_000.0)
    # 0.05 * 5e9 / 1e5 = 2500, unclipped.
    assert detail.waiting_penalty == pytest.approx(2500.0, rel=0.6)


def test_waiting_penalty_combines_backlog_and_growth() -> None:
    # The backlog-aware term keeps pressure on a large queue even when it
    # temporarily stops growing, while the delta term still provides the dense
    # signal that says "this step made the queue worse".
    fn = _reward_fn()
    for _ in range(5):
        _compute(fn, arrived=100000.0, failed_keys=0.0)
    no_growth = _compute(fn, arrived=100000.0, waiting_keys=5_000_000_000.0, waiting_delta=0.0)
    assert no_growth.waiting_penalty > 0.0
    growth = _compute(fn, arrived=100000.0, waiting_keys=5_000_000_000.0, waiting_delta=1_000_000.0)
    # 0.05 * 1e6 / 1e5 = 0.5 plus the smaller backlog term.
    assert growth.waiting_penalty > no_growth.waiting_penalty


def test_success_rate_reward_mode_tracks_cumulative_ratio() -> None:
    from qkd_rl.env.action_resolver import ResolvedAction
    from qkd_rl.env.qkp import LinkQKPPool
    from qkd_rl.env.request import ServeResult
    from qkd_rl.env.routing import AllocationResult

    fn = RewardFunction({"mode": "success_rate"})
    r1 = fn.compute(
        serve_result=ServeResult([], [], [], 20.0, 0.0, 0.0),
        allocation=AllocationResult(added_keys=0.0, overflow_keys=0.0),
        expired_requests=[],
        expired_keys=0.0,
        resolved_action=ResolvedAction(activated_edges=[], rejected_actions={}, illegal_actions={}, conflict_count=0),
        qkp=LinkQKPPool([], {"capacity": {"default": 1.0}, "initial_level": 0.0}),
        arrived_keys=100.0,
        served_keys=20.0,
    )
    r2 = fn.compute(
        serve_result=ServeResult([], [], [], 50.0, 0.0, 0.0),
        allocation=AllocationResult(added_keys=0.0, overflow_keys=0.0),
        expired_requests=[],
        expired_keys=0.0,
        resolved_action=ResolvedAction(activated_edges=[], rejected_actions={}, illegal_actions={}, conflict_count=0),
        qkp=LinkQKPPool([], {"capacity": {"default": 1.0}, "initial_level": 0.0}),
        arrived_keys=100.0,
        served_keys=50.0,
    )
    assert r1.total == pytest.approx(0.2)
    assert r2.total == pytest.approx(0.15)
    assert (r1.total + r2.total) == pytest.approx(0.35)


def test_dense_importance_reward_uses_added_keys_and_importance() -> None:
    from qkd_rl.env.action_resolver import ResolvedAction
    from qkd_rl.env.qkp import LinkQKPPool
    from qkd_rl.env.request import ServeResult
    from qkd_rl.env.routing import AllocationResult

    def compute(fn: RewardFunction, added: float, importance: float):
        return fn.compute(
            serve_result=ServeResult([], [], [], 0.0, 0.0, 0.0),
            allocation=AllocationResult(added_keys=added, overflow_keys=0.0),
            expired_requests=[],
            expired_keys=0.0,
            resolved_action=ResolvedAction([], {}, {}, 0),
            qkp=LinkQKPPool([], {"capacity": {"default": 1.0}, "initial_level": 0.0}),
            arrived_keys=0.0,
            added_by_edge={"e1": added},
            relay_importance={"e1": importance},
        )

    high_fn = RewardFunction(
        {
            "mode": "success_rate",
            "dense_generation_importance_weight": 1.0,
            "dense_low_importance_penalty": 0.0,
            "dense_reference": 1000.0,
        }
    )
    assert compute(high_fn, 1000.0, 1.0).total == pytest.approx(1.0)

    low_fn = RewardFunction(
        {
            "mode": "success_rate",
            "dense_generation_importance_weight": 1.0,
            "dense_low_importance_penalty": 1.0,
            "dense_reference": 1000.0,
        }
    )
    assert compute(low_fn, 1000.0, 0.0).total == pytest.approx(-1.0)


def test_success_rate_reward_penalizes_failure_waiting_and_switch() -> None:
    from qkd_rl.core.types import KeyRequest
    from qkd_rl.env.action_resolver import ResolvedAction
    from qkd_rl.env.qkp import LinkQKPPool
    from qkd_rl.env.request import ServeResult
    from qkd_rl.env.routing import AllocationResult

    fn = RewardFunction(
        {
            "mode": "success_rate",
            "failed_weight": 1.0,
            "waiting_weight": 0.1,
            "waiting_stock_weight": 0.01,
            "switch_weight": 0.5,
            "penalty_reference": 1000.0,
        }
    )
    expired = [KeyRequest("REQ_FAIL", "A", "B", 1000.0, 0, 10)]
    result = fn.compute(
        serve_result=ServeResult([], [], [], 0.0, 1000.0, 0.0),
        allocation=AllocationResult(added_keys=0.0, overflow_keys=0.0),
        expired_requests=expired,
        expired_keys=0.0,
        resolved_action=ResolvedAction([], {}, {}, 0),
        qkp=LinkQKPPool([], {"capacity": {"default": 1.0}, "initial_level": 0.0}),
        arrived_keys=0.0,
        waiting_keys=1000.0,
        waiting_delta=1000.0,
        switch_count=2,
    )
    assert result.failed_penalty == pytest.approx(1.0)
    assert result.waiting_penalty == pytest.approx(0.11)
    # switch is a link-toggle count and keeps O(1) scale: 0.5 * 2 = 1.0.
    assert result.switch_penalty == pytest.approx(1.0)
    assert result.total == pytest.approx(-2.11)
