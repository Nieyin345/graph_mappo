"""How long does ONE episode actually take? (decisive for the worker sweep)

Doubling n_rollout_workers from 4 to 8 changed nothing (227 -> 238 s), which
means either the pool is not running episodes in parallel, or a single episode
already costs the whole rollout. The pool's wall time cannot tell those apart,
so this times one episode in-process, on the same config.

If this comes out near `rollout_s / 8` the pool is fine and something else is
serial; if it comes out near `rollout_s` the "parallel" rollout is not parallel.

    python .tmp/one_episode.py --steps 1440 --repeats 1
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
    ap.add_argument("--steps", type=int, default=1440)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="0 = leave torch default")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    tgm = load_train_module()
    ns = argparse.Namespace(
        mode="random_episode",
        configs=["train_mappo.yaml"],
        seed=args.seed,
        num_updates=1,
        run_name="one_ep",
        checkpoint=None,
        device=args.device,
    )
    config = tgm.build_config(ns)
    gamma = float(config["train"]["gamma"])
    gae_lambda = float(config["train"]["gae_lambda"])
    value_target = str(config["train"].get("value_target", "gae"))
    episode_steps = int(config["env"].get("episode_steps", args.steps))

    print(f"steps={args.steps} episode_steps={episode_steps} device={args.device}", flush=True)
    print(f"torch threads={torch.get_num_threads()}", flush=True)

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)
    # Load through the trainer rather than a raw torch.load: the checkpoint's
    # state_dict key and the partial-load policy live there, and guessing them
    # is how a probe silently measures a randomly initialised model.
    from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer

    out = ROOT / "outputs" / "one_ep"
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out, device=args.device)
    ckpt = ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)
    policy = trainer.policy
    env = trainer.env
    policy.model.eval()

    times = []
    for i in range(args.repeats + 1):
        t0 = time.perf_counter()
        steps, ep_reward, summary = _run_episode(
            env, policy, args.seed, args.steps, gamma, gae_lambda, value_target, episode_steps
        )
        dt = time.perf_counter() - t0
        # The worker pool ships each episode back by pickling the finished
        # RolloutStep list to a file, then the trainer unpickles it. If that
        # payload is large, that round trip -- not the compute -- is what makes
        # the "parallel" rollout take 8x a single episode.
        import io
        import pickle

        buf = io.BytesIO()
        t1 = time.perf_counter()
        pickle.dump((steps, ep_reward, summary), buf, protocol=4)
        t2 = time.perf_counter()
        payload = buf.getvalue()
        t3 = time.perf_counter()
        pickle.loads(payload)
        t4 = time.perf_counter()
        print(
            f"  episode {i}: {dt:.2f} s for {len(steps)} steps "
            f"({1000 * dt / max(1, len(steps)):.1f} ms/step) reward={ep_reward:.1f}",
            flush=True,
        )
        print(
            f"    ship payload: {len(payload) / 1e6:.1f} MB, "
            f"pickle {t2 - t1:.2f} s, unpickle {t4 - t3:.2f} s",
            flush=True,
        )
        if i:
            times.append(dt)

    times.sort()
    print(f"RESULT one_episode_median={times[len(times) // 2]:.2f} s")


if __name__ == "__main__":
    main()
