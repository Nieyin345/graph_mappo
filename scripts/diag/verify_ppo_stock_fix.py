"""Bounded server verification with real H5 data and an existing checkpoint.

This is a correctness smoke, not an estimate of policy performance. Each arm
starts from the same checkpoint; outputs must go to a new directory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qkd_rl.core.config import ConfigValidator
from qkd_rl.env.factory import build_env_from_config
from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer
from qkd_rl.rl.algos.policy import MAPPOPolicy
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    records = []
    for stocked in (False, True):
        torch.manual_seed(42)
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        config["runtime"]["device"] = "cpu"
        config["rate_provider"]["h5"]["dataset_dir"] = str(args.dataset_dir.resolve())
        actor_cfg = config["model"].setdefault("actor", {})
        actor_cfg.setdefault("demand_residual", {"enabled": False, "num_heads": 4})
        actor_cfg["demand_residual"]["enabled"] = True
        edge_cfg = config["features"].setdefault("edge", {})
        edge_cfg["include_stocked_edges"] = stocked
        config["env"].update(episode_steps=args.steps, continuous=False)
        config["train"].update(rollout_steps=args.steps, episodes_per_update=2,
                               n_rollout_workers=2, episode_steps_fixed=True, replay_days=0)
        config["train"]["ppo"].update(minibatch_size=32, batch_chunk=16)
        if args.actor_lr is not None:
            config["train"]["optimizer"]["actor_lr"] = args.actor_lr
        if args.target_kl is not None:
            config["train"]["ppo"]["target_kl"] = args.target_kl
        config["seed"].update(global_seed=42, env_seed=42)
        ConfigValidator().validate(config)
        output = args.output / ("attention_stocked" if stocked else "attention_joint")
        output.mkdir()
        (output / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        env = build_env_from_config(config)
        model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
        policy = MAPPOPolicy(model)
        trainer = MAPPOTrainer(env, policy, config, output)
        try:
            trainer.load_checkpoint(args.checkpoint)
            for _ in range(args.updates):
                started = time.monotonic()
                buffer = trainer.collect_rollout()
                # Independent worker collection vs parent recomputation must
                # start at ratio=1 before any optimization step.
                samples = buffer.steps[::max(1, len(buffer.steps)//8)][:8]
                with torch.no_grad():
                    evaluated = policy.evaluate_actions_batched(
                        [s.obs for s in samples], [s.actions for s in samples],
                        [s.matched_edges for s in samples])
                    errors = [abs(float(lp[s.obs.node_ids[0]] - s.mean_log_prob))
                              for s, (lp, _, _) in zip(samples, evaluated)]
                assert max(errors) < 2e-3, errors
                before = model.actor.demand_residual.out[-1].weight.detach().clone()
                stats = trainer.update(buffer)
                record = {"stocked": stocked, "stats": asdict(stats),
                          "max_initial_log_prob_error": max(errors),
                          "attention_weight_delta": float((
                              model.actor.demand_residual.out[-1].weight.detach()-before).abs().max()),
                          "elapsed_s": time.monotonic()-started}
                assert all(torch.isfinite(torch.tensor(v)) for v in asdict(stats).values()
                           if isinstance(v, (int, float)))
                assert record["attention_weight_delta"] > 0, "Attention received no update"
                records.append(record)
                (args.output / "summary.json").write_text(
                    json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
                print(json.dumps(record, allow_nan=False), flush=True)
        finally:
            trainer.shutdown()


if __name__ == "__main__":
    main()
