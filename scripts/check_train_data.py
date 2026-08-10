"""Reusable rollout/PPO intermediate-data sanity checker.

Runs a short serial rollout, then verifies that:

- matched edges are unique and endpoint-disjoint;
- every stored matched edge exists in the recomputed edge scores;
- ``joint_log_prob`` matches a fresh evaluation from the stored observation;
- reward / value / returns / advantages / observation features are finite;
- one PPO update produces finite losses and changes parameters.

Usage (from the project root):

    python scripts/check_train_data.py --rollout-steps 12 --episodes 1
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qkd_rl.algos.mappo_trainer import MAPPOTrainer
from qkd_rl.algos.policy import MAPPOPolicy
from qkd_rl.core.config import ConfigValidator, deep_merge, load_config
from qkd_rl.env.factory import build_env_from_config, load_default_config
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


def _check(buffer, policy) -> list[str]:
    problems: list[str] = []
    endpoints = policy._endpoints()
    for i, step in enumerate(buffer.steps):
        edges = list(step.matched_edges or [])
        if len(set(edges)) != len(edges):
            problems.append(f"step {i}: duplicate matched_edges")
        used: set[str] = set()
        for edge_id in edges:
            if edge_id not in endpoints:
                problems.append(f"step {i}: unknown edge {edge_id}")
                continue
            src, dst = endpoints[edge_id]
            if src in used or dst in used:
                problems.append(f"step {i}: endpoint conflict {edge_id}")
            used.update((src, dst))

        output = policy.model(step.obs, policy.device, build_logits_dict=False)
        edge_scores = output.edge_scores or {}
        missing = [edge_id for edge_id in edges if edge_id not in edge_scores]
        if missing:
            problems.append(f"step {i}: edges missing from edge_scores {missing}")
        lp, _ent = policy.log_prob_entropy_for_matching(edge_scores, edges)
        if not torch.isfinite(step.joint_log_prob):
            problems.append(f"step {i}: non-finite joint_log_prob")
        elif not torch.allclose(lp.detach().cpu(), step.joint_log_prob.detach().cpu(), atol=1e-5):
            problems.append(f"step {i}: joint_log_prob mismatch {lp} vs {step.joint_log_prob}")

        for name, value in [
            ("reward", step.reward),
            ("value", step.value),
            ("returns", step.returns),
            ("advantages", step.advantages),
        ]:
            if value is None or not bool(torch.isfinite(torch.as_tensor(value)).all()):
                problems.append(f"step {i}: non-finite {name}")

        for name, arr in [
            ("node_features", step.obs.node_features),
            ("edge_features", step.obs.edge_features),
        ]:
            array = np.asarray(arr, dtype=np.float32)
            if array.size and not np.isfinite(array).all():
                problems.append(f"step {i}: non-finite {name}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-steps", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config = deep_merge(
        config,
        {
            "train": {
                "num_updates": 1,
                "rollout_steps": args.rollout_steps,
                "episodes_per_update": args.episodes,
                "n_rollout_workers": 1,
                "rollout_batch": False,
                "ppo": {"epochs": 1, "minibatch_size": 32, "batch_chunk": 32},
                "logging": {"log_interval": 0, "checkpoint_interval": 100000, "eval_interval": 0},
            },
            "project": {"output_dir": "_diag", "run_name": "check_train_data"},
        },
    )
    ConfigValidator().validate(config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.TemporaryDirectory(prefix="qkd_train_data_check_") as tmp:
        config["project"]["output_dir"] = tmp
        env = build_env_from_config(config)
        model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
        policy = MAPPOPolicy(model, device)
        trainer = MAPPOTrainer(env, policy, config, tmp, device=device)
        buffer = trainer.collect_rollout()
        print("steps:", len(buffer.steps))

        problems = _check(buffer, policy)
        if problems:
            print("PROBLEMS")
            print("\n".join(problems[:20]))
            sys.exit(1)

        params_before = [p.detach().clone() for p in model.parameters()]
        stats = trainer.update(buffer)
        for name, value in [
            ("actor_loss", stats.actor_loss),
            ("critic_loss", stats.critic_loss),
            ("entropy", stats.entropy),
            ("kl", stats.kl),
            ("mean_reward", stats.mean_reward),
            ("mean_success_rate", stats.mean_success_rate),
        ]:
            if not np.isfinite(float(value)):
                problems.append(f"non-finite stat {name}: {value}")
        if not any(
            not torch.equal(a, b) for a, b in zip(params_before, model.parameters())
        ):
            problems.append("parameters did not change after PPO update")

        if problems:
            print("PROBLEMS")
            print("\n".join(problems[:20]))
            sys.exit(1)

        print("OK")
        print(
            "actor_loss=",
            round(float(stats.actor_loss), 6),
            "critic_loss=",
            round(float(stats.critic_loss), 6),
            "entropy=",
            round(float(stats.entropy), 4),
            "kl=",
            round(float(stats.kl), 6),
            "success_rate=",
            round(float(stats.mean_success_rate), 4),
        )


if __name__ == "__main__":
    main()
