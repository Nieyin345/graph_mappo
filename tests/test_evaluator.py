"""Tests for the evaluation module: episode records, persistence, aggregation."""
from __future__ import annotations

from pathlib import Path

import pytest

from qkd_rl.baselines.random_policy import RandomPolicy
from tests.helpers import build_test_env
import json

from qkd_rl.evaluation import Evaluator, plot_learning_curve, plot_policy_comparison, read_train_history
from qkd_rl.evaluation.evaluator import (
    EpisodeRecord,
    aggregate_episodes,
    merge_eval_summary,
    write_episodes_csv,
    write_steps_csv,
    write_summary_json,
)
from qkd_rl.evaluation.plots import _rolling_smooth
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _build_env(seed: int):
    env = build_test_env(ROOT)
    env.reset(seed=seed)
    return env


def test_evaluator_records_episode_and_steps(tmp_path: Path) -> None:
    evaluator = Evaluator(env_builder=_build_env)
    episodes, steps = evaluator.run_policy(
        RandomPolicy(seed=0), "random", num_episodes=1, seeds=[7], collect_steps=True
    )
    assert len(episodes) == 1
    record = episodes[0]
    assert record.policy == "random"
    assert record.steps > 0
    assert record.arrived_keys > 0
    assert record.success_rate >= 0.0
    assert steps is not None and len(steps) == record.steps
    assert set(steps[0]) >= {"step", "reward", "served_keys", "failed_keys", "qkp_utilization"}


def test_persistence_and_aggregation(tmp_path: Path) -> None:
    evaluator = Evaluator(env_builder=_build_env)
    episodes, _ = evaluator.run_policy(RandomPolicy(seed=0), "random", num_episodes=2, seeds=[1, 2])
    ep_path = write_episodes_csv(episodes, tmp_path / "episodes.csv")
    steps_path = write_steps_csv(
        [{"episode": 0, "seed": 1, "step": 0, "reward": 1.0, "served_keys": 0.0}],
        tmp_path / "steps.csv",
    )
    summary_path = write_summary_json(episodes, tmp_path / "summary.json")
    assert ep_path.exists() and steps_path.exists() and summary_path.exists()
    assert ep_path.read_text(encoding="utf-8").splitlines()[0].startswith("policy,")

    agg = aggregate_episodes(episodes)
    assert "random" in agg
    assert agg["random"]["n_episodes"] == 2
    assert 0.0 <= agg["random"]["success_rate_mean"] <= 1.0


def test_plot_policy_comparison_outputs_vector(tmp_path: Path) -> None:
    evaluator = Evaluator(env_builder=_build_env)
    episodes, _ = evaluator.run_policy(RandomPolicy(seed=0), "random", num_episodes=1, seeds=[3])
    figures = plot_policy_comparison(episodes, tmp_path / "figures" / "comparison")
    names = {path.suffix for path in figures}
    assert {".svg", ".pdf", ".png"} <= names
    svg = tmp_path / "figures" / "comparison.svg"
    assert svg.exists()
    assert "<image" not in svg.read_text(encoding="utf-8")


def _write_history(path: Path, n_updates: int = 60) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for update in range(1, n_updates + 1):
            record = {
                "update": update,
                "actor_loss": 1.0 / update,
                "critic_loss": 0.5 / update,
                "entropy": 0.2,
                "kl": 0.01,
                "mean_reward": 10.0 * (1 - 1 / update),
                "mean_return": 10.0,
                "mean_success_rate": 0.5 * (1 - 1 / update),
                "mean_served_keys": 20.0 * (1 - 1 / update),
                "elapsed_s": 0.1,
            }
            handle.write(json.dumps(record) + "\n")
        handle.write(json.dumps({"eval": {"mean_reward": 1.0}}) + "\n")


def test_read_train_history_skips_eval_rows(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write_history(path)
    rows = read_train_history(path)
    assert len(rows) == 60
    assert all("update" in row and "mean_reward" in row for row in rows)


def test_plot_learning_curve_outputs_vector(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write_history(path)
    histories = [read_train_history(path) for _ in range(3)]
    figures = plot_learning_curve(histories, tmp_path / "figures" / "learning_curve", title="demo")
    names = {figure.suffix for figure in figures}
    assert {".svg", ".pdf", ".png"} <= names
    svg = tmp_path / "figures" / "learning_curve.svg"
    assert svg.exists()
    assert "<image" not in svg.read_text(encoding="utf-8")


def test_rolling_smooth_short_series_does_not_create_triangle() -> None:
    values = [0.09] * 20
    smoothed = _rolling_smooth(values, window=20)
    np.testing.assert_allclose(smoothed, values, atol=1e-6)


def test_compare_can_override_env_builder_per_policy() -> None:
    class _Metrics:
        def episode_summary(self):
            return {"arrived_keys": 1.0, "served_keys": 1.0, "failed_keys": 0.0,
                    "success_rate": 1.0, "conflict_count": 0}

    class _Env:
        def __init__(self, marker: str):
            self.marker = marker
            self.metrics = _Metrics()
        def reset(self, seed=None, start_seed=None):
            return {"marker": self.marker}
        def step(self, actions, scores=None):
            return {"marker": self.marker}, 0.0, True, False, {}

    class _Policy:
        def __init__(self, expected: str):
            self.expected = expected
        def act(self, obs):
            assert obs["marker"] == self.expected
            return {}, {}

    evaluator = Evaluator(lambda seed: _Env("default"))
    episodes, _ = evaluator.compare(
        {"base": _Policy("default"), "rl": _Policy("checkpoint")},
        num_episodes=1, seeds=[7],
        env_builders={"rl": lambda seed: _Env("checkpoint")},
    )
    assert [ep.policy for ep in episodes] == ["base", "rl"]

def _sample_episode() -> EpisodeRecord:
    return EpisodeRecord(
        policy="random", episode=0, seed=7, steps=1, total_reward=1.0,
        arrived_keys=10.0, served_keys=8.0, failed_keys=2.0,
        success_rate=0.8, conflict_count=0,
    )


def test_merge_eval_summary_refuses_to_overwrite_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    original = "{not valid json"
    path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to overwrite unreadable evaluation summary"):
        merge_eval_summary([_sample_episode()], path)
    assert path.read_text(encoding="utf-8") == original


def test_merge_eval_summary_warns_on_malformed_historical_episode(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps({"policies": {"random": {"runs": [{"episode_log": [{"bad": "shape"}]}]}}}),
        encoding="utf-8",
    )
    with pytest.warns(RuntimeWarning, match="skipped 1 malformed episode_log entries"):
        merge_eval_summary([_sample_episode()], path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    random_blob = payload["policies"]["random"]
    assert random_blob["invalid_episode_logs"] == 1
    assert random_blob["n_episodes"] == 1
