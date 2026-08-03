"""Inspect the active RateProvider and compute rate statistics.

Checks the provider interface (get_rate / get_edge_window /
get_all_edge_windows), samples the rate distribution for training
normalization, and writes ``rate_stats.json``.

Usage (run from the project root):

    python scripts/inspect_rate_provider.py                 # default (h5)
    python scripts/inspect_rate_provider.py --provider h5   # H5 dataset
    python scripts/inspect_rate_provider.py --samples 2000 --output stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from qkd_rl.core.config import ConfigValidator
from qkd_rl.data.scenario_builder import ScenarioBuilder
from qkd_rl.env.factory import load_default_config
from qkd_rl.link.rate_provider import build_rate_provider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="h5", choices=["h5"], help="Rate provider (H5 dataset only).")
    parser.add_argument("--scenario", default=None, choices=["small", "full"], help="Override scenario mode.")
    parser.add_argument("--dataset-dir", default=None, help="H5 dataset directory (provider=h5).")
    parser.add_argument("--samples", type=int, default=1000, help="Number of sampled time slots.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Output path for rate_stats.json.")
    args = parser.parse_args()

    config = load_default_config(ROOT)
    if args.provider:
        config["rate_provider"]["provider"] = args.provider
    if args.scenario:
        config["scenario"]["mode"] = args.scenario
    if args.dataset_dir:
        config["rate_provider"]["h5"]["dataset_dir"] = args.dataset_dir
    ConfigValidator().validate(config)

    scenario_builder = ScenarioBuilder(config)
    scenario_mode = config["scenario"].get("mode", "small")
    if scenario_mode == "small":
        scenario = scenario_builder.build_small()
    else:
        scenario = scenario_builder.build_full()

    provider = build_rate_provider(config, scenario.edges, seed=args.seed)
    print(
        f"provider={config['rate_provider']['provider']} "
        f"nodes={len(scenario.nodes)} edges={len(scenario.edges)} "
        f"horizon={int(config['features']['edge']['prediction_horizon'])}"
    )

    start_t = int(config["rate_provider"]["time"]["start_index"])
    end_t = int(config["rate_provider"]["time"]["end_index"])
    provider_end = getattr(provider, "time_bounds", None)
    if provider_end is not None:
        end_t = min(end_t, provider_end[1])
    if end_t <= start_t:
        raise ValueError(f"Empty sampling range: [{start_t}, {end_t})")

    rng = np.random.RandomState(args.seed)
    times = sorted(rng.randint(start_t, end_t, size=args.samples))

    per_edge: dict[str, dict] = {edge_id: {"mean": 0.0, "std": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "zero_count": 0} for edge_id in scenario.edge_ids}
    all_rates: list[float] = []
    nan_count = 0
    negative_count = 0
    zero_only_edges: list[str] = []

    # Use the batched path the env actually consumes.
    first_window = provider.get_all_edge_windows(times[0])
    window_len = len(next(iter(first_window.values())).rates)
    for t in times:
        windows = provider.get_all_edge_windows(t)
        assert set(windows) == set(scenario.edge_ids), "provider must cover every scenario edge"
        for edge_id, window in windows.items():
            rates = window.rates
            stats = per_edge[edge_id]
            stats["zero_count"] += sum(1 for r in rates if r <= 0.0)
            for r in rates:
                if np.isnan(r):
                    nan_count += 1
                elif r < 0.0:
                    negative_count += 1
                else:
                    all_rates.append(r)

    # Spot-check the scalar interface on a few edges/times.
    for edge_id in scenario.edge_ids[:3]:
        for t in times[:5]:
            provider.get_rate(edge_id, t)
            provider.is_available(edge_id, t)
    sample_window = provider.get_edge_window(scenario.edge_ids[0], times[0])
    assert len(sample_window.rates) == window_len

    if all_rates:
        arr = np.asarray(all_rates, dtype=np.float64)
        dist = {
            "count": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }
    else:
        dist = {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    for edge_id, stats in per_edge.items():
        if stats["zero_count"] >= len(times) * window_len:
            zero_only_edges.append(edge_id)
        stats.pop("zero_count", None)

    result = {
        "provider": config["rate_provider"]["provider"],
        "node_count": len(scenario.nodes),
        "edge_count": len(scenario.edges),
        "sampled_times": len(times),
        "window_length": window_len,
        "nan_count": nan_count,
        "negative_count": negative_count,
        "zero_only_edge_count": len(zero_only_edges),
        "zero_only_edges_sample": zero_only_edges[:20],
        "rate_distribution": dist,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    output = Path(args.output) if args.output else Path(config["project"]["output_dir"]) / "rate_stats.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()