"""Interleaved A/B of batch_chunk, to separate a real effect from drift.

The first sweep ran each chunk once, in order, on the same buffer:
    512 -> 95.07 s, 256 -> 90.44 s, 128 -> 67.02 s
and then, at the end, chunk=512 again -> 74.43 s. Same configuration, 22%
apart, so the single-run numbers cannot be compared as-is: the first update of
a process is measurably slower than later ones (allocator growth, page faults
on the freshly built buffer), and that one-off cost sat on chunk=512.

This alternates the chunk sizes across rounds and reports per-chunk medians,
so the warm-up cost is spread over every configuration instead of landing on
whichever one ran first.

    python .tmp/sweep_chunk_ab.py --chunks 512,128,64 --rounds 2
"""
from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from collections import defaultdict
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
    ap.add_argument("--chunks", default="512,128,64")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=0,
                    help="extra updates to discard before timing")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    chunks = [int(c) for c in args.chunks.split(",")]
    tgm = load_train_module()
    ns = argparse.Namespace(
        mode="random_episode", configs=["train_mappo.yaml"], seed=0,
        num_updates=1, run_name="chunk_ab", checkpoint=None, device=args.device,
    )
    config = tgm.build_config(ns)
    print(f"torch threads = {torch.get_num_threads()}", flush=True)

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)
    out = ROOT / "outputs" / "chunk_ab"
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out, device=args.device)
    ckpt = ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)

    t0 = time.perf_counter()
    buffer = trainer.collect_rollout()
    print(f"rollout: {time.perf_counter() - t0:.2f} s\n", flush=True)

    # One untimed update first, so the one-off allocator/page-fault cost is paid
    # before any measurement rather than landing on whichever chunk runs first.
    trainer.config["train"]["ppo"]["batch_chunk"] = chunks[0]
    t0 = time.perf_counter()
    trainer.update(buffer)
    print(f"  warm-up (untimed, chunk={chunks[0]}): {time.perf_counter() - t0:.2f} s\n",
          flush=True)

    times: dict[int, list[float]] = defaultdict(list)
    for r in range(args.rounds):
        for chunk in chunks:
            trainer.config["train"]["ppo"]["batch_chunk"] = chunk
            t0 = time.perf_counter()
            stats = trainer.update(buffer)
            dt = time.perf_counter() - t0
            times[chunk].append(dt)
            print(f"  round {r + 1}  chunk={chunk:<5d} {dt:7.2f} s  nb={stats.n_minibatches}",
                  flush=True)

    print("\n=== medians ===")
    base = statistics.median(times[chunks[0]])
    for chunk in chunks:
        med = statistics.median(times[chunk])
        print(f"  chunk={chunk:<5d} {med:7.2f} s   {base / med:.2f}x vs chunk={chunks[0]}")
    print("CHUNK_AB_DONE")


if __name__ == "__main__":
    main()
