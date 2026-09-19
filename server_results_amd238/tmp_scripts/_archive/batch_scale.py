"""Time act_batched at different graph-group sizes on CPU.

The rollout_batch_envs=4 cap was tuned on the laptop GPU (4 graphs 57 ms vs
8 graphs 409 ms per step there). On the server CPU the scaling may be different
-- 8 graphs per block-diagonal forward could be nearly as cheap per graph and
halve the number of sampler/forward calls. This measures it before touching any
config.

Usage:
    python .tmp/batch_scale.py --steps 240 --episodes 8
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

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, "cpu")
    ckpt = ROOT / args.checkpoint
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model_state = state.get("model_state", state)
        policy.model.load_state_dict(model_state)

    envs = [build_env_from_config(config) for _ in range(args.episodes)]
    obs_list = [envs[i].reset(seed=7 + i) for i in range(args.episodes)]

    # Warm up.
    for _ in range(5):
        policy.act_batched(obs_list, build_scores=False, use_edge_arrays=True)

    for g in (1, 2, 4, 8):
        group = obs_list[:g]
        n_call = 20
        t0 = time.perf_counter()
        for _ in range(n_call):
            policy.act_batched(group, build_scores=False, use_edge_arrays=True)
        dt = (time.perf_counter() - t0) / n_call
        print(f"G={g:2d}  {dt*1000:8.2f} ms/step  ({dt*1000/g:6.2f} ms per graph)", flush=True)


if __name__ == "__main__":
    main()
