"""Wall-clock one MAPPO update, no profiler in the way.

cProfile is useless at this scale: it adds several hundred percent on a workload
with millions of Python-level tensor ops, which made the A/B take >20 minutes per
side and told us nothing the op table had not. This just times the update.

    python .tmp/time_update.py --device cpu --steps 240 --episodes 8 --repeats 3
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
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0, help="0 = leave torch default")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    tgm = load_train_module()
    ns = argparse.Namespace(
        mode="random_episode",
        configs=["rl_algorithm.yaml", "train_diag_fast.yaml"],
        seed=0,
        num_updates=1,
        run_name="time_tmp",
        checkpoint=None,
        device=args.device,
    )
    config = tgm.build_config(ns)
    config["train"]["rollout_steps"] = args.steps
    config["train"]["episodes_per_update"] = args.episodes

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)
    out = ROOT / "outputs" / "time_tmp"
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out, device=args.device)
    ckpt = ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)

    t0 = time.perf_counter()
    buffer = trainer.collect_rollout()
    print(f"rollout: {time.perf_counter() - t0:.2f} s ({len(buffer.steps)} steps)", flush=True)

    times = []
    for i in range(args.repeats):
        t0 = time.perf_counter()
        trainer.update(buffer)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  update {i + 1}: {dt:.2f} s", flush=True)

    times.sort()
    print(f"UPDATE_MEDIAN {times[len(times) // 2]:.2f}  (threads={torch.get_num_threads()})")


if __name__ == "__main__":
    main()
