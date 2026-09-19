"""Does torch.compile(dynamic=True) speed up the PPO update?

The rollout profile says a step costs 44 separate `linear` calls on tiny inputs
-- launch overhead, not arithmetic. The update has the same character: its
block-diagonal forwards are a long chain of small elementwise ops and matmuls
over shapes that vary from chunk to chunk.

`torch.compile` is the tool for exactly that, and the earlier attempt failed for
a reason that has a specific fix: with `dynamic=False`, Dynamo re-specialises on
every new shape, hit `config.recompile_limit (8)`, and fell back. `dynamic=True`
makes it emit shape-generic guards instead -- which is the documented answer to
that failure mode, and has not been tested here.

    python .tmp/compile_dyn.py
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode", default="dynamic", choices=["dynamic", "default", "both"])
    args = ap.parse_args()

    tgm = load_train_module()
    ns = argparse.Namespace(
        mode="random_episode", configs=["train_mappo.yaml"], seed=0,
        num_updates=1, run_name="compile_dyn", checkpoint=None, device=args.device,
    )
    config = tgm.build_config(ns)
    ppo = config["train"]["ppo"]
    print(f"threads={torch.get_num_threads()} chunk={ppo.get('batch_chunk')} "
          f"minibatch={ppo['minibatch_size']}", flush=True)

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)
    out = ROOT / "outputs" / "compile_dyn"
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out, device=args.device)
    ckpt = ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)

    t0 = time.perf_counter()
    buffer = trainer.collect_rollout()
    print(f"rollout: {time.perf_counter() - t0:.2f} s\n", flush=True)

    def timed(tag: str) -> float:
        t0 = time.perf_counter()
        stats = trainer.update(buffer)
        dt = time.perf_counter() - t0
        print(f"  {tag:34s} {dt:7.2f} s  nb={stats.n_minibatches} kl={stats.kl:.5f}",
              flush=True)
        return dt

    base = timed("A) eager")

    modes = ["dynamic", "default"] if args.mode == "both" else [args.mode]
    for m in modes:
        # dynamic=None means "let Dynamo decide"; the earlier failure was with a
        # hard dynamic=False, which re-specialises per shape and hit the
        # recompile limit. Pin it explicitly so the two arms are comparable.
        kwargs = {"dynamic": True} if m == "dynamic" else {"dynamic": False}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                policy.model.batched_forward = torch.compile(
                    policy.model.batched_forward, **kwargs
                )
            dt = timed(f"B) compiled batched_forward ({m})")
            print(f"     -> {base / dt:.2f}x vs eager", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  B) compile ({m}) FAILED: {type(exc).__name__}: {exc}", flush=True)

    print("COMPILE_DYN_DONE")


if __name__ == "__main__":
    main()
