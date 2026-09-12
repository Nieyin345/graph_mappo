"""Collect and persist PG-Phased expert trajectories to disk.

Runs the finalized three-stage heuristic (``PathScoreGreedy(phased=True)`` +
``ServeProbe``) over the training window as one independent episode per day,
and writes each day as a single gzip-pickled file holding, per step:

- the minimal model-input fields of the observation (node/edge features,
  edge index, ids, action candidates/masks);
- the actually-executed directed arcs (``env.last_matched_arcs``), i.e. the
  exact matching the policy clones.

Day metrics (served keys, arrived keys, success rate, completed requests,
activated edges) are appended to ``metrics.json`` so the expert's final success
rate is available for comparison without re-running the heuristic.

After collection, ``supervised_train_pg_phased.py --data-dir <out>`` trains the
behavior-cloning policy directly from the stored trajectories without re-running
the heuristic. Days are collected in parallel (one process per worker) because
per-day episodes are independent.

Usage:
    python scripts/rl/collect_pg_phased_trajectories.py --out-dir outputs/trajs_pg_phased
    python scripts/rl/collect_pg_phased_trajectories.py --workers 8 --start-day 0 --end-day 295
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qkd_rl.baselines.path_greedy import PathScoreGreedy
from qkd_rl.baselines.serve_probe import ServeProbe
from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config

DAY_STEPS = 1440


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/supervised_train.yaml")
    parser.add_argument("--out-dir", type=str, default="outputs/trajs_pg_phased")
    parser.add_argument("--start-day", type=int, default=None)
    parser.add_argument("--end-day", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def build_base_config(config_path: str) -> dict:
    """Per-day independent episodes over the global training window."""
    raw = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8")) or {}
    ev = raw.get("evaluation", {}) or {}
    global_raw = yaml.safe_load((ROOT / "configs" / "global.yaml").read_text(encoding="utf-8")) or {}
    train_global = global_raw.get("global", {}).get("training", {}) or {}
    window = train_global.get("window", {}) or {}
    start_day = int(window.get("start_day", 0))
    end_day = int(window.get("end_day", 0))
    episode_steps = int(ev.get("episode_steps", 0) or 0) or DAY_STEPS
    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config["rate_provider"]["provider"] = "h5"
    config["env"]["episode_start_mode"] = "random_day"
    config["env"]["episode_steps"] = episode_steps
    config["env"]["activation_window_start_day"] = start_day
    config["env"]["activation_window_days"] = end_day - start_day
    config["scenario"]["time_limit"]["days"] = end_day + max(1, math.ceil(episode_steps / DAY_STEPS))
    config["project"]["output_dir"] = "outputs"
    ConfigValidator().validate(config)
    return config


def _obs_record(obs) -> dict:
    """Minimal model-input fields of one observation (state not needed for
    the model forward and is the heaviest part of the object)."""
    return {
        "node_features": np.asarray(obs.node_features, dtype=np.float32),
        "edge_index": np.asarray(obs.edge_index, dtype=np.int64),
        "edge_features": np.asarray(obs.edge_features, dtype=np.float32),
        "node_ids": list(obs.node_ids),
        "edge_ids": list(obs.edge_ids),
        "physical_edge_ids": list(obs.physical_edge_ids),
        "demand_edge_ids": list(obs.demand_edge_ids),
        "action_candidates": obs.action_candidates,
        "action_masks": obs.action_masks,
        "raw_action_masks": obs.raw_action_masks,
        "flat_action_masks": obs.flat_action_masks,
    }


def collect_day(day: int, base_config: dict, out_dir: Path) -> dict:
    """Run one independent PG-Phased episode (day) and persist it to disk."""
    config = copy.deepcopy(base_config)
    config["env"]["episode_start_day"] = day
    env = build_env_from_config(config)
    expert = PathScoreGreedy(
        weights=(1.0, 10.0, 1.0, 0.5, 0.2),
        phased=True,
        principles=False,
        router=ServeProbe(env),
    )
    obs = env.reset(seed=day)
    records = []
    total_served = 0.0
    n_act = 0
    steps = 0
    for _ in range(int(config["env"]["episode_steps"])):
        actions, _scores = expert.act(obs)
        obs_next, _reward, terminated, truncated, info = env.step(actions)
        rec = _obs_record(obs)
        rec["arcs"] = list(env.last_matched_arcs)
        records.append(rec)
        total_served += float(info.get("served_keys", 0.0))
        n_act += len(getattr(env, "last_activated_edges", []) or [])
        steps += 1
        obs = obs_next
        if terminated or truncated:
            break
    arrived = float(env.metrics.arrived_keys)
    sr = total_served / arrived if arrived > 0 else 0.0
    sm = env.metrics.episode_summary()
    with gzip.open(out_dir / f"day_{day:04d}.pkl.gz", "wb") as fh:
        pickle.dump(records, fh, protocol=4)
    return {
        "day": day,
        "steps": steps,
        "served_keys": total_served,
        "arrived_keys": arrived,
        "success_rate": sr,
        "activated_per_step": n_act / max(1, steps),
        "completed_requests": sm.get("completed_requests", 0),
        "records": len(records),
        "file": str(out_dir / f"day_{day:04d}.pkl.gz"),
    }


def main() -> None:
    args = parse_args()
    base_config = build_base_config(args.config)
    start_day = int(args.start_day if args.start_day is not None else base_config["env"]["activation_window_start_day"])
    end_day = int(args.end_day if args.end_day is not None else start_day + base_config["env"]["activation_window_days"])
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    days = list(range(start_day, end_day))
    metrics = []
    workers = max(1, min(args.workers, len(days)))
    print(f"[collect] days {start_day}->{end_day} ({len(days)} days), workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(collect_day, day, base_config, out_dir): day for day in days}
        for fut in as_completed(futures):
            day = futures[fut]
            try:
                m = fut.result()
            except Exception as exc:  # keep going; report the failed day
                print(f"day={day} FAILED: {exc}", flush=True)
                metrics.append({"day": day, "failed": str(exc)})
                continue
            metrics.append(m)
            print(
                f"day={m['day']:>3} steps={m['steps']:>4} SR={m['success_rate']:.4f} "
                f"served={m['served_keys']:.0f}/{m['arrived_keys']:.0f} "
                f"done={m['completed_requests']} avg_act={m['activated_per_step']:.2f}",
                flush=True,
            )
    metrics.sort(key=lambda m: m["day"])
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    srs = [m["success_rate"] for m in metrics if "success_rate" in m]
    if srs:
        print(f"[collect] done {len(srs)}/{len(metrics)} days, avg SR={sum(srs) / len(srs):.4f}", flush=True)
        print(f"[collect] metrics: {out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
