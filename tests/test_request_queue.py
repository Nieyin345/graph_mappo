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
    common = {"arrival_rate": 3.0, "amount_mean": 10.0, "deadline_steps": 12}
    uniform = RequestGenerator(["GS_001", "GS_002"], {**common, "priority_mode": "uniform"}, seed=0)
    reqs = [r for _ in range(200) for r in uniform.generate(_)]
    assert reqs and all(req.priority == 1.0 for req in reqs)

    random_gen = RequestGenerator(["GS_001", "GS_002"], {**common, "priority_mode": "random"}, seed=1)
    reqs_random = [r for _ in range(200) for r in random_gen.generate(_)]
    assert reqs_random and all(1.0 <= req.priority <= 3.0 for req in reqs_random)


def test_request_generator_poisson_load():
    """arrival_rate is the expected requests per slot (Poisson)."""
    gen = RequestGenerator(
        ["GS_001", "GS_002", "GS_003", "GS_004"],
        {"arrival_rate": 6.0, "amount_mean": 1000.0, "deadline_steps": 12},
        seed=0,
    )
    totals = [len(gen.generate(t)) for t in range(3000)]
    mean = sum(totals) / len(totals)
    assert 5.0 < mean < 7.0  # empirical Poisson(6) mean
    assert max(totals) > 1  # multiple requests per slot do occur
    amounts = [r.amount for t in range(200) for r in gen.generate(t)]
    assert sum(amounts) / len(amounts) > 500.0  # mean amount tracks amount_mean


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
import pytest

def test_request_generator_hourly_pattern():
    """hourly_weights shape the Poisson rate: night hours << peak hours."""
    from qkd_rl.env.request import RequestGenerator

    hourly_weights = [0.15, 0.10, 0.10, 0.10, 0.10, 0.15, 0.30, 0.60, 1.00, 1.20,
                      1.10, 1.00, 0.90, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.20,
                      1.00, 0.80, 0.50, 0.25]
    cfg = {"arrival_rate": 10.0, "amount_mean": 100.0, "deadline_steps": 12,
           "steps_per_hour": 60, "hourly_weights": hourly_weights,
           "pair_hotness": "uniform"}
    gen = RequestGenerator(["GS_001", "GS_002", "GS_003", "GS_004"], cfg, seed=0)

    # hour 0 weight 0.15 vs hour 9 weight 1.20, 60 slots each
    night = sum(len(gen.generate(t)) for t in range(0, 60))
    peak = sum(len(gen.generate(t)) for t in range(9 * 60, 9 * 60 + 60))

    assert night > 0 and peak > night * 2.0  # expected ratio ~8x
    # peak mean per slot should be far above the daily average (10/slot)
    assert peak / 60.0 > 12.0


def test_request_generator_zipf_hotness():
    """zipf pair hotness concentrates demand; uniform does not."""
    from qkd_rl.env.request import RequestGenerator

    gs = ["GS_001", "GS_002", "GS_003", "GS_004"]

    def pair_counts(cfg, steps=1200):
        gen = RequestGenerator(gs, cfg, seed=7)
        counts = {}
        for t in range(steps):
            for req in gen.generate(t):
                key = tuple(sorted((req.src_gs, req.dst_gs)))
                counts[key] = counts.get(key, 0) + 1
        vals = sorted(counts.values())
        return vals[-1], vals[0]

    base = {"arrival_rate": 20.0, "amount_mean": 10.0, "deadline_steps": 12}
    zipf_cfg = {**base, "pair_hotness": "zipf", "pair_zipf_s": 1.0, "pair_seed": 0}
    uniform_cfg = {**base, "pair_hotness": "uniform"}

    top, bottom = pair_counts(zipf_cfg)
    assert top > bottom * 2.5  # hot pair carries most demand

    top_u, bottom_u = pair_counts(uniform_cfg)
    assert top_u < bottom_u * 1.4  # roughly flat


def test_request_generator_rejects_unknown_pair_hotness():
    from qkd_rl.env.request import RequestGenerator

    cfg = {"arrival_rate": 1.0, "amount_mean": 10.0, "pair_hotness": "banana"}
    with pytest.raises(ValueError):
        RequestGenerator(["GS_001", "GS_002"], cfg, seed=0)


def test_request_generator_yaml_config_drives_pattern():
    """The shipped env_full.yaml request block must construct and show the
    intra-day + zipf pattern (yaml -> code contract for the new keys)."""
    from pathlib import Path
    import yaml
    from qkd_rl.env.request import RequestGenerator

    cfg = yaml.safe_load((Path("configs/env_full.yaml")).read_text(encoding="utf-8"))["requests"]
    assert cfg["pair_hotness"] == "zipf" and len(cfg["hourly_weights"]) == 24

    gen = RequestGenerator(["GS_001", "GS_002", "GS_003", "GS_004"], cfg, seed=0)
    night = sum(len(gen.generate(t)) for t in range(0, 60))
    peak = sum(len(gen.generate(t)) for t in range(9 * 60, 9 * 60 + 60))
    assert peak > night * 2.0

    pairs = {}
    for t in range(300):
        for req in gen.generate(t):
            key = tuple(sorted((req.src_gs, req.dst_gs)))
            pairs[key] = pairs.get(key, 0) + 1
    vals = sorted(pairs.values())
    assert vals[-1] > vals[0] * 2.0

