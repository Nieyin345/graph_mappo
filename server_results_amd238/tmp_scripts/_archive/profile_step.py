"""Where do the ~20 ms of a rollout step actually go?

Established so far: a step costs ~20 ms and barely responds to torch threads,
so it is Python/numpy bound. But "Python bound" does not say WHICH Python --
and the candidates have very different fixes:

  * rebuilding the per-slot observation from the full topology each step
    (node/edge feature assembly, mask construction, candidate enumeration);
  * reading link rates out of the H5 for the slot;
  * the environment transition itself (queue/QKP bookkeeping over 1978 links);
  * the model forward and the matching sampler.

This runs a few hundred steps under cProfile and prints the top functions by
self time, which separates those cases.

    python .tmp/profile_step.py --steps 300
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
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402
from qkd_rl.rl.algos.rollout_workers import _run_episode  # noqa: E402
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
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--top", type=int, default=28)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    tgm = load_train_module()
    ns = argparse.Namespace(
        mode="random_episode", configs=["train_mappo.yaml"], seed=0,
        num_updates=1, run_name="prof_step", checkpoint=None, device=args.device,
    )
    config = tgm.build_config(ns)
    gamma = float(config["train"]["gamma"])
    gae_lambda = float(config["train"]["gae_lambda"])
    value_target = str(config["train"].get("value_target", "gae"))
    episode_steps = int(config["env"].get("episode_steps", args.steps))

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)
    policy.model.eval()

    # Warm-up, so the profile is not dominated by one-off setup (H5 open,
    # topology build, allocator growth).
    _run_episode(env, policy, 0, 30, gamma, gae_lambda, value_target, episode_steps)

    t0 = time.perf_counter()
    pr = cProfile.Profile()
    pr.enable()
    _run_episode(env, policy, 1, args.steps, gamma, gae_lambda, value_target, episode_steps)
    pr.disable()
    dt = time.perf_counter() - t0
    print(f"\n{args.steps} steps under cProfile: {dt:.2f} s "
          f"({1000 * dt / args.steps:.1f} ms/step, inflated by profiling)\n", flush=True)

    stream = io.StringIO()
    pstats.Stats(pr, stream=stream).sort_stats("tottime").print_stats(args.top)
    lines = stream.getvalue().splitlines()
    start = next((i for i, ln in enumerate(lines) if "ncalls" in ln), 0)
    print("\n".join(lines[start:start + args.top + 6]))


if __name__ == "__main__":
    main()
