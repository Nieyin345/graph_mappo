"""Evaluator: run one or several policies over seeds/episodes and persist metrics.

Raw per-episode and (optionally) per-step records are written as CSV so every
figure is reproducible from data, independent of the plotting code.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Callable


@dataclass
class EpisodeRecord:
    policy: str
    episode: int
    seed: int
    steps: int
    total_reward: float
    arrived_keys: float
    served_keys: float
    failed_keys: float
    success_rate: float
    conflict_count: int


class Evaluator:
    """Run policies inside the QKD env.

    Parameters
    ----------
    env_builder:
        Callable taking a seed and returning a freshly-reset ``QKDEnv``.
    """

    def __init__(self, env_builder: Callable[[int], object]):
        self.env_builder = env_builder

    def run_policy(
        self,
        policy,
        policy_name: str,
        num_episodes: int = 5,
        seeds: list[int] | None = None,
        collect_steps: bool = False,
    ) -> tuple[list[EpisodeRecord], list[dict] | None]:
        """Run one policy and return episode records + optional step records."""
        episodes: list[EpisodeRecord] = []
        step_rows: list[dict] = []
        for ep in range(num_episodes):
            seed = seeds[ep % len(seeds)] if seeds else (ep + 1) * 1000
            env = self.env_builder(seed)
            obs = env.reset(seed=seed)
            total_reward = 0.0
            done = False
            step = 0
            while not done:
                actions, scores = self._act(policy, obs)
                obs, reward, terminated, truncated, info = env.step(actions, scores)
                total_reward += reward
                done = terminated or truncated
                if collect_steps:
                    row = self._step_row(ep, seed, step, reward, info)
                    step_rows.append(row)
                step += 1
            summary = env.metrics.episode_summary()
            episodes.append(
                EpisodeRecord(
                    policy=policy_name,
                    episode=ep,
                    seed=seed,
                    steps=step,
                    total_reward=round(total_reward, 6),
                    arrived_keys=summary["arrived_keys"],
                    served_keys=summary["served_keys"],
                    failed_keys=summary["failed_keys"],
                    success_rate=summary["success_rate"],
                    conflict_count=summary["conflict_count"],
                )
            )
        return episodes, (step_rows if collect_steps else None)

    def compare(
        self,
        policies: dict[str, object],
        num_episodes: int = 5,
        seeds: list[int] | None = None,
        collect_steps: bool = False,
    ) -> tuple[list[EpisodeRecord], dict[str, list[dict]]]:
        """Run several policies and return all episode records + step records per policy."""
        all_episodes: list[EpisodeRecord] = []
        all_steps: dict[str, list[dict]] = {}
        for name, policy in policies.items():
            episodes, steps = self.run_policy(
                policy, name, num_episodes=num_episodes, seeds=seeds, collect_steps=collect_steps
            )
            all_episodes.extend(episodes)
            if steps:
                all_steps[name] = steps
        return all_episodes, all_steps

    @staticmethod
    def _act(policy, obs):
        result = policy.act(obs)
        if isinstance(result, tuple):
            return result[0], result[1] if len(result) > 1 else None
        actions = getattr(result, "actions", None)
        scores = getattr(result, "action_scores", None)
        return actions, scores

    @staticmethod
    def _step_row(ep, seed, step, reward, info) -> dict:
        return {
            "episode": ep,
            "seed": seed,
            "step": step,
            "reward": round(float(reward), 6),
            "served_keys": info.get("served_keys", 0.0),
            "failed_keys": info.get("failed_keys", 0.0),
            "waiting_keys": info.get("waiting_keys", 0.0),
            "generated_keys": info.get("generated_keys", 0.0),
            "conflict_count": info.get("conflict_count", 0),
            "qkp_utilization": info.get("qkp_utilization", 0.0),
        }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def write_episodes_csv(records: list[EpisodeRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return path


def write_steps_csv(step_rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(step_rows[0].keys()))
        writer.writeheader()
        writer.writerows(step_rows)
    return path


def aggregate_episodes(records: list[EpisodeRecord]) -> dict:
    """Per-policy mean/std over episodes."""
    by_policy: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        by_policy.setdefault(record.policy, []).append(record)
    agg: dict[str, dict] = {}
    for name, items in sorted(by_policy.items()):
        fields = ["total_reward", "served_keys", "failed_keys", "success_rate", "conflict_count", "steps"]
        agg[name] = {
            "n_episodes": len(items),
            **{
                f"{field}_mean": mean(getattr(item, field) for item in items)
                for field in fields
            },
            **{
                f"{field}_std": stdev(getattr(item, field) for item in items)
                if len(items) > 1
                else 0.0
                for field in fields
            },
        }
    return agg


def write_summary_json(records: list[EpisodeRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(aggregate_episodes(records), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
