"""Where does one collect_rollout() actually spend its wall time?

The server baseline (2026-09-15) is 27.2 s per rollout (240 steps x 8 episodes,
1920 steps) vs 20.3 s for the update: rollout is now ~57% of a round and it is
Python-bound, so the fix has to name Python-level hotspots, not torch ops.

Usage:
    python .tmp/profile_rollout.py --steps 240 --episodes 8 --top 25
"""
from __future__ import annotations

import argparse
import cProfile
import importlib.util
import io
import pstats
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
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--checkpoint", default="outputs/supervised_pg_phased/supervised_pg_phased_latest.pt")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    tgm = load_train_module()
    ns = argparse.Namespace(
        mode="random_episode",
        configs=["rl_algorithm.yaml", "train_diag_fast.yaml"],
        seed=None,
        num_updates=1,
        run_name="prof_tmp",
        checkpoint=None,
        device="cpu",
    )
    config = tgm.build_config(ns)
    config["train"]["rollout_steps"] = args.steps
    config["train"]["episodes_per_update"] = args.episodes

    print(f"torch={torch.__version__} threads={torch.get_num_threads()} "
          f"steps={args.steps} episodes={args.episodes}", flush=True)
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, "cpu")
    out = ROOT / "outputs" / "prof_tmp"
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out, device="cpu")
    ckpt = ROOT / args.checkpoint
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)

    # Warm-up pass (feature caches, rate provider precompute) then time the real one.
    print("warm-up rollout...", flush=True)
    trainer.collect_rollout()

    print("collecting rollout under cProfile...", flush=True)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    buffer = trainer.collect_rollout()
    pr.disable()
    wall = time.perf_counter() - t0
    print(f"=== collect_rollout wall time: {wall:.2f} s ({len(buffer.steps)} steps) ===", flush=True)

    stream = io.StringIO()
    st = pstats.Stats(pr, stream=stream)
    st.sort_stats("tottime")
    st.print_stats(args.top + 6)
    lines = stream.getvalue().splitlines()
    start = next((i for i, ln in enumerate(lines) if "ncalls" in ln), 0)
    print("\n".join(lines[start:start + args.top + 8]), flush=True)

    # Cumulative share of the top-level trainer/environment calls.
    print("\n=== cumulative top-level ===", flush=True)
    stream = io.StringIO()
    st = pstats.Stats(pr, stream=stream)
    st.sort_stats("cumulative")
    st.print_stats(12)
    lines = stream.getvalue().splitlines()
    start = next((i for i, ln in enumerate(lines) if "ncalls" in ln), 0)
    print("\n".join(lines[start:start + 15]), flush=True)


if __name__ == "__main__":
    main()
