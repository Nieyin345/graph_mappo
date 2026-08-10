"""Calibrate request intensity and QKP capacities against real H5 rates.

The environment must be sized from the links that are actually activated by
the agents (a greedy matching over usable links), not from the all-link
average: most links are never selected (global frac_zero ~0.87) and the
activated subset is much faster.

Usage:
    conda run -n pytorch python scripts/calibrate_demand.py \
        --dataset dataset/global --slots 600 --load-ratio 0.65 --ttl-steps 1920

Outputs per-slot matching statistics and a ready-to-paste YAML block.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "dataset" / "global"


def load_registry(dataset_dir: Path) -> dict[int, tuple[int, int, str]]:
    links: dict[int, tuple[int, int, str]] = {}
    with (dataset_dir / "link_registry.csv").open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            links[int(row["link_id"])] = (int(row["node_u"]), int(row["node_v"]), row["link_type"])
    return links


def greedy_match_stats(
    dataset_dir: Path,
    slots: int = 600,
    seed: int = 0,
    min_rate_bps: float = 1.0,
) -> dict:
    """Sample slots, build a greedy rate-sorted matching, and return stats.

    Returns dict with arrays: activated_edges_per_slot, per_edge_bits,
    total_bits_per_slot, available_edges_per_slot.
    """
    dataset_dir = Path(dataset_dir)
    links = load_registry(dataset_dir)
    h5_path = dataset_dir / "link_data.h5"
    with h5py.File(h5_path, "r") as f:
        kmax = f["k_max"]
        los = f["los"]
        T, L = kmax.shape
        theta = int(f.attrs.get("theta_sec", 60))
        rng = np.random.RandomState(seed)
        ts = sorted(rng.choice(max(T - 6, 1), min(slots, T), replace=False))

        n_act = []
        per_edge_bits = []
        total_bits = []
        avail_edges = []
        for t in ts:
            r = kmax[t]
            l = los[t]
            usable = np.where((r >= min_rate_bps) & (l == 1))[0]
            avail_edges.append(len(usable))
            order = usable[np.argsort(-r[usable])]
            used = set()
            chosen = []
            for e in order:
                u, v, _ = links[e]
                if u in used or v in used:
                    continue
                used.add(u)
                used.add(v)
                chosen.append(e)
            n_act.append(len(chosen))
            step_total = 0.0
            for e in chosen:
                bits = float(r[e]) * theta
                per_edge_bits.append(bits)
                step_total += bits
            total_bits.append(step_total)

    return {
        "theta_sec": theta,
        "available_edges_per_slot": np.array(avail_edges, dtype=np.float32),
        "activated_edges_per_slot": np.array(n_act, dtype=np.float32),
        "per_edge_bits": np.array(per_edge_bits, dtype=np.float32),
        "total_bits_per_slot": np.array(total_bits, dtype=np.float32),
    }


def percentile(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p)) if x.size else 0.0


def suggest_config(stats: dict, load_ratio: float = 0.65, ttl_steps: int = 1920) -> dict:
    """Derive requests/qkp settings from the activated-link statistics."""
    total_p50 = percentile(stats["total_bits_per_slot"], 50)
    edge_p50 = percentile(stats["per_edge_bits"], 50)
    lambda_req = max(1.0, round(total_p50 * load_ratio / 500000.0))
    amount_mean = round(total_p50 * load_ratio / lambda_req)
    capacity = max(1000.0, round(edge_p50 * ttl_steps))
    return {
        "requests": {
            "type": "poisson",
            "arrival_rate": lambda_req,
            "amount_mean": amount_mean,
            "amount_max": round(5.0 * amount_mean),
            "deadline_steps": 960,
        },
        "qkp": {
            "initial_level": 0.0,
            "key_ttl_steps": ttl_steps,
            "capacity_default": capacity,
        },
        "summary": {
            "generated_bits_per_slot_p50": round(total_p50),
            "generated_bits_per_slot_mean": round(float(np.mean(stats["total_bits_per_slot"]))),
            "per_edge_bits_p50": round(edge_p50),
            "activated_edges_per_slot_p50": round(percentile(stats["activated_edges_per_slot"], 50)),
            "activated_edges_per_slot_mean": round(float(np.mean(stats["activated_edges_per_slot"]))),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate requests/qkp from real H5 rates.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--slots", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-ratio", type=float, default=0.65)
    parser.add_argument("--ttl-steps", type=int, default=1920)
    args = parser.parse_args()

    stats = greedy_match_stats(args.dataset, slots=args.slots, seed=args.seed)
    s = stats
    print(f"theta_sec            : {s['theta_sec']}")
    print(f"available edges/slot : mean={np.mean(s['available_edges_per_slot']):.1f} "
          f"p50={percentile(s['available_edges_per_slot'], 50):.0f}")
    print(f"activated edges/slot : mean={np.mean(s['activated_edges_per_slot']):.1f} "
          f"p50={percentile(s['activated_edges_per_slot'], 50):.0f} "
          f"p90={percentile(s['activated_edges_per_slot'], 90):.0f}")
    print(f"per-edge bits/slot   : mean={np.mean(s['per_edge_bits']):.0f} "
          f"p50={percentile(s['per_edge_bits'], 50):.0f} "
          f"p90={percentile(s['per_edge_bits'], 90):.0f}")
    print(f"total bits/slot      : mean={np.mean(s['total_bits_per_slot']):.0f} "
          f"p50={percentile(s['total_bits_per_slot'], 50):.0f} "
          f"p10={percentile(s['total_bits_per_slot'], 10):.0f}")
    print()
    print("Suggested config:")
    print(__import__("yaml").safe_dump(suggest_config(stats, args.load_ratio, args.ttl_steps), sort_keys=False))


if __name__ == "__main__":
    main()
