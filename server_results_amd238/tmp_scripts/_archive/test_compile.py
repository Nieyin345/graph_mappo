"""Does torch.compile speed up a MAPPO update?

The update runs at ~3% of CPU peak, and the per-operator counts are dominated
by tensor-view churn (`as_strided` 778k, `select` 378k, `unsqueeze` 382k for
480 graphs) rather than matmuls. That is the classic shape of a fusion problem,
which is exactly what torch.compile is for -- but this model has Python control
flow around the matching decoder, so Dynamo will graph-break there. Measure,
do not assume.

Targets, measured separately so we can see which half benefits:
  A. policy.model.batched_forward          (GNN forward, pure tensor ops)
  B. policy._matching_log_prob_entropy_fast (per-graph feasibility matmul)

Usage:
    python .tmp/test_compile.py --device cpu --steps 240 --episodes 8
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import warnings
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


def timed_update(trainer, buffer, repeats=2):
    t0 = time.perf_counter()
    trainer.update(buffer)
    warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(repeats):
        trainer.update(buffer)
    return (time.perf_counter() - t0) / repeats, warm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--checkpoint", default="outputs/supervised_pg_phased/supervised_pg_phased_latest.pt")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch={torch.__version__} threads={torch.get_num_threads()} "
          f"device={args.device}", flush=True)

    tgm = load_train_module()
    ns = argparse.Namespace(
        mode="random_episode",
        configs=["rl_algorithm.yaml", "train_diag_fast.yaml"],
        seed=None,
        num_updates=1,
        run_name="compile_tmp",
        checkpoint=None,
        device=args.device,
    )
    config = tgm.build_config(ns)
    config["train"]["rollout_steps"] = args.steps
    config["train"]["episodes_per_update"] = args.episodes

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)
    out = ROOT / "outputs" / "compile_tmp"
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out, device=args.device)
    ckpt = ROOT / args.checkpoint
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)

    buffer = trainer.collect_rollout()
    print(f"buffer: {len(buffer.steps)} steps\n", flush=True)

    base, base_warm = timed_update(trainer, buffer)
    print(f"A) baseline                       : {base:7.2f} s  (first {base_warm:.2f} s)", flush=True)

    # --- compile the GNN forward ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            policy.model.batched_forward = torch.compile(
                policy.model.batched_forward, dynamic=False
            )
        compiled_fwd, warm_fwd = timed_update(trainer, buffer)
        print(f"B) + compiled batched_forward     : {compiled_fwd:7.2f} s  "
              f"(first {warm_fwd:.2f} s incl. compile)  "
              f"-> {base / compiled_fwd:.2f}x on steady state", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"B) compiled batched_forward FAILED: {type(exc).__name__}: {exc}", flush=True)
        compiled_fwd = base

    # --- also compile the matching decoder ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            policy._matching_log_prob_entropy_fast = torch.compile(
                policy._matching_log_prob_entropy_fast, dynamic=False
            )
        both, warm_both = timed_update(trainer, buffer)
        print(f"C) + compiled matching decoder    : {both:7.2f} s  "
              f"(first {warm_both:.2f} s incl. compile)  "
              f"-> {base / both:.2f}x vs baseline", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"C) compiled matching decoder FAILED: {type(exc).__name__}: {exc}", flush=True)
        both = compiled_fwd

    best = min(base, compiled_fwd, both)
    print(f"\nbest {best:.2f} s vs baseline {base:.2f} s = {base / best:.2f}x", flush=True)


if __name__ == "__main__":
    main()
