"""Calibration script must run on the real H5 path and yield sane numbers."""

from __future__ import annotations

from scripts.calibrate_demand import greedy_match_stats, suggest_config
from tests.helpers import SMALL_H5_DIR, ensure_small_h5


def test_greedy_match_stats_runs_on_small_h5() -> None:
    ensure_small_h5(SMALL_H5_DIR)
    stats = greedy_match_stats(SMALL_H5_DIR, slots=50, seed=0)
    # 6 nodes (4 GS + 1 HAP + 1 SAT) => at most 3 activated edges per slot.
    assert stats["activated_edges_per_slot"].max() <= 3
    assert stats["total_bits_per_slot"].min() >= 0.0
    assert stats["per_edge_bits"].size > 0
    assert stats["available_edges_per_slot"].min() >= 0


def test_suggest_config_sane_quantities() -> None:
    ensure_small_h5(SMALL_H5_DIR)
    stats = greedy_match_stats(SMALL_H5_DIR, slots=100, seed=1)
    cfg = suggest_config(stats, load_ratio=0.5, ttl_steps=60)
    assert cfg["requests"]["type"] == "poisson"
    assert cfg["requests"]["arrival_rate"] >= 1
    assert cfg["requests"]["amount_mean"] > 0
    assert cfg["qkp"]["capacity_default"] > 0
    # Request load must be of the same order of magnitude as generated load.
    total_p50 = cfg["summary"]["generated_bits_per_slot_p50"]
    offered = cfg["requests"]["arrival_rate"] * cfg["requests"]["amount_mean"]
    assert 0.3 * total_p50 <= offered <= 1.5 * total_p50
