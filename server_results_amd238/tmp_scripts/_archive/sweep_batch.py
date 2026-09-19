"""Does a bigger PPO minibatch actually speed up the update?

The intuition is that 125 GB of RAM should buy large batches and therefore
speed. That is only true if the update is tensor-throughput bound -- and it is
measurably NOT: the update runs at ~3% of CPU peak (9.5 GFLOP/s against 300+
available on 48 cores), because the cost is the NUMBER of small operators and
their serial dependency chains, not the size of the tensors.

Batching does still remove per-minibatch fixed costs (optimizer step, the
Python batch loop, buffer sampling), so it may buy something. Measure it rather
than argue about it.

For each (minibatch_size, batch_chunk) this runs one real PPO update on a fixed
rollout buffer and reports wall time, time per rollout step, and -- the number
that matters -- time per rollout step normalised by how much work the update
did.

Usage:
    python .tmp/sweep_batch.py --device cpu --steps 240 --episodes 8
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
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--configs", nargs="+", default=["rl_algorithm.yaml", "train_diag_fast.yaml"])
    ap.add_argument("--checkpoint", default="outputs/supervised_pg_phased/supervised_pg_phased_latest.pt")
    # (minibatch_size, batch_chunk) pairs. batch_chunk is the block-diagonal
    # forward chunk; if it is smaller than the minibatch the forward is split,
    # which re-introduces the per-chunk overhead we are trying to remove.
    ap.add_argument(
        "--grid",
        default="256:256,512:512,1024:1024,1920:1920",
        help="comma-separated minibatch:chunk pairs",
    )
    args = ap.parse_args()

    grid = []
    for item in args.grid.split(","):
        mb, ch = item.split(":")
        grid.append((int(mb), int(ch)))

    tgm = load_train_module()
    ns = argparse.Namespace(
        mode="random_episode",
        configs=args.configs,
        seed=None,
        num_updates=1,
        run_name="batch_tmp",
        checkpoint=None,
        device=args.device,
    )
    config = tgm.build_config(ns)
    config["train"]["rollout_steps"] = args.steps
    config["train"]["episodes_per_update"] = args.episodes

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)
    out = ROOT / "outputs" / "batch_tmp"
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out, device=args.device)
    ckpt = ROOT / args.checkpoint
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)

    print(f"torch={torch.__version__} threads={torch.get_num_threads()} device={args.device}")
    print("collecting one rollout buffer (reused for every configuration)...")
    t0 = time.perf_counter()
    buffer = trainer.collect_rollout()
    print(f"  rollout: {time.perf_counter() - t0:.2f} s ({len(buffer.steps)} steps)\n")

    print(f"{'minibatch':>10} {'chunk':>7} {'batches':>8} {'update_s':>10} {'ms/step':>9} {'s/batch':>9}")
    for mb, ch in grid:
        trainer.config["train"]["ppo"]["minibatch_size"] = mb
        trainer.config["train"]["ppo"]["batch_chunk"] = ch
        # Warmup so the thread pool and allocator are hot before timing.
        trainer.update(buffer)
        t0 = time.perf_counter()
        trainer.update(buffer)
        upd = time.perf_counter() - t0
        n_batches = max(1, -(-len(buffer.steps) // mb))
        print(
            f"{mb:10d} {ch:7d} {n_batches:8d} {upd:10.2f} "
            f"{1000 * upd / len(buffer.steps):9.2f} {upd / n_batches:9.3f}"
        )


if __name__ == "__main__":
    main()
