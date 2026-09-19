"""Sweep the PPO update over torch threads and batch_chunk, on the real config.

The update is now the biggest phase of a round (~95 s of 167 s), and it is the
one phase that parallelises, so its shape matters twice over: once for single-run
speed, and once for deciding how many arms fit side by side on the node.

Two things are being probed:

* **batch_chunk** (512 in train_mappo.yaml). The chunk is how many steps go into
  one block-diagonal model forward, so at ~393 directed edges per step a chunk
  of 512 makes an edge tensor of ~200k x 128 float32 -- about 103 MB, which is
  the size of this CPU's entire L3. Halving it should fit in cache. Since the
  slice-backward fix removed the memsets, what is left is dominated by memory
  traffic, which is exactly what cache residency buys.
* **threads**. If the update saturates below 48 threads, an arm costs fewer cores
  than it is given, and several arms can share the node.

    python .tmp/sweep_update.py --threads 48,24,16 --chunks 512,256,128
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
    ap.add_argument("--threads", default="48,24,16")
    ap.add_argument("--chunks", default="512,256,128")
    ap.add_argument("--verify-chunk", type=int, default=512,
                    help="chunk to hold fixed while sweeping threads")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tgm = load_train_module()
    ns = argparse.Namespace(
        mode="random_episode",
        configs=["train_mappo.yaml"],
        seed=0,
        num_updates=1,
        run_name="sweep_upd",
        checkpoint=None,
        device=args.device,
    )
    config = tgm.build_config(ns)
    ppo = config["train"]["ppo"]
    print(
        f"epochs={ppo['epochs']} minibatch={ppo['minibatch_size']} "
        f"chunk={ppo.get('batch_chunk')} rollout={config['train']['rollout_steps']} "
        f"episodes={config['train']['episodes_per_update']}",
        flush=True,
    )
    print(f"torch default threads = {torch.get_num_threads()}", flush=True)

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)
    out = ROOT / "outputs" / "sweep_upd"
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out, device=args.device)
    ckpt = ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)

    t0 = time.perf_counter()
    buffer = trainer.collect_rollout()
    print(f"rollout: {time.perf_counter() - t0:.2f} s ({len(buffer.steps)} steps)\n", flush=True)

    def timed(tag: str) -> float:
        t0 = time.perf_counter()
        stats = trainer.update(buffer)
        dt = time.perf_counter() - t0
        print(
            f"  {tag:24s} {dt:7.2f} s   nb={stats.n_minibatches} kl={stats.kl:.5f}",
            flush=True,
        )
        return dt

    print("=== batch_chunk sweep (threads = default) ===", flush=True)
    for chunk in (int(c) for c in args.chunks.split(",")):
        trainer.config["train"]["ppo"]["batch_chunk"] = chunk
        timed(f"chunk={chunk}")

    print("\n=== thread sweep (chunk held at the default) ===", flush=True)
    trainer.config["train"]["ppo"]["batch_chunk"] = args.verify_chunk
    for n in (int(t) for t in args.threads.split(",")):
        torch.set_num_threads(n)
        timed(f"threads={n}")

    print("SUPD_DONE")


if __name__ == "__main__":
    main()
