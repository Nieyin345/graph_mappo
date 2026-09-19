"""Time the rollout at a given n_rollout_workers, on the canonical config.

The rollout is now the dominant cost (86% of a round on the real config), and
it is embarrassingly parallel *across episodes* -- `RolloutWorkerPool.collect`
hands whole episodes to worker processes. The canonical preset runs 8 episodes
on 4 workers, i.e. 2 episodes each, on a 48-core box. This measures what more
workers actually buy, rather than assuming.

    python .tmp/sweep_workers.py --workers 4 --repeats 2

One worker count per process: the pool is built lazily on the first
collect_rollout and is not meant to be resized in place.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer  # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "tgm", ROOT / "scripts" / "rl" / "train_graph_mappo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--num-updates", type=int, default=0, help="0 = rollout only")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tgm = load_train_module()
    ns = argparse.Namespace(
        mode="random_episode",
        # The canonical preset -- same merge order the real runs use, so
        # episodes_per_update=8 and rollout_steps=1440 land in the config.
        configs=["train_mappo.yaml"],
        seed=0,
        num_updates=1,
        run_name=f"wsw_{args.workers}",
        checkpoint=None,
        device=args.device,
    )
    config = tgm.build_config(ns)
    config["train"]["n_rollout_workers"] = args.workers

    print(
        f"rollout_steps={config['train']['rollout_steps']} "
        f"episodes={config['train']['episodes_per_update']} "
        f"workers={config['train']['n_rollout_workers']} "
        f"epochs={config['train']['ppo']['epochs']} "
        f"minibatch={config['train']['ppo']['minibatch_size']} "
        f"chunk={config['train']['ppo'].get('batch_chunk')}",
        flush=True,
    )

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)
    out = ROOT / "outputs" / f"wsw_{args.workers}"
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out, device=args.device)
    ckpt = ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)

    roll, upd = [], []
    for i in range(args.repeats + 1):
        t0 = time.perf_counter()
        buffer = trainer.collect_rollout()
        dt = time.perf_counter() - t0
        if i:  # skip the first: it also builds the worker pool
            roll.append(dt)
        print(f"  rollout {i}: {dt:.2f} s ({len(buffer.steps)} steps)", flush=True)
        if args.num_updates:
            t0 = time.perf_counter()
            trainer.update(buffer)
            du = time.perf_counter() - t0
            if i:
                upd.append(du)
            print(f"  update  {i}: {du:.2f} s", flush=True)

    roll.sort()
    med = roll[len(roll) // 2]
    print(f"RESULT workers={args.workers} rollout_median={med:.2f} steps={len(buffer.steps)}")
    if upd:
        upd.sort()
        print(f"RESULT workers={args.workers} update_median={upd[len(upd) // 2]:.2f}")


if __name__ == "__main__":
    main()
