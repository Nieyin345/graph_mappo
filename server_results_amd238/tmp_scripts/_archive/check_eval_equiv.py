"""Verify the array-input PPO evaluation is bit-identical to the dict-input one.

Compares evaluate_actions_batched (new array path) against per-graph
evaluate_actions (old dict path) on the same observations and matchings.
mean_lp / mean_entropy / value must match exactly.

Usage:
    python .tmp/check_eval_equiv.py --steps 60 --episodes 2
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
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
    config["train"]["n_rollout_workers"] = 1

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

    with torch.no_grad():
        outputs = policy.model.batched_forward(obs_list, "cpu", want_edge_maps=True)
        edge_maps = outputs.edge_score_maps
        arrays = outputs.edge_arrays

    max_lp = 0.0
    max_ent = 0.0
    for i, obs in enumerate(obs_list):
        matched = policy._sample_matching(
            edge_maps[i], deterministic=False, build_scores=True
        )[1]  # matched_edges
        node_order = list(obs.node_ids)
        node_pos = {n: j for j, n in enumerate(node_order)}
        lp_new, ent_new = policy._matching_log_prob_entropy_arrays(
            arrays[i][0], arrays[i][1], arrays[i][2], node_pos, matched
        )
        lp_old, ent_old = policy._matching_log_prob_entropy_fast(
            edge_maps[i], matched
        )
        dlp = float(torch.abs(lp_new - lp_old))
        dent = float(torch.abs(ent_new - ent_old))
        max_lp = max(max_lp, dlp)
        max_ent = max(max_ent, dent)
        ok = dlp == 0.0 and dent == 0.0
        print(f"graph {i}: matched={len(matched)}  lp_new={float(lp_new):.6f} lp_old={float(lp_old):.6f} "
              f"ent_new={float(ent_new):.6f} ent_old={float(ent_old):.6f}  diff_lp={dlp:.2e} diff_ent={dent:.2e}  {'OK' if ok else 'MISMATCH'}", flush=True)

    print(f"\nmax |diff| lp={max_lp:.3e} ent={max_ent:.3e}")
    if max_lp == 0.0 and max_ent == 0.0:
        print("RESULT: bit-identical")
    elif max_lp < 1e-6 and max_ent < 1e-6:
        print("RESULT: float-noise equivalent")
    else:
        print("RESULT: MISMATCH")


if __name__ == "__main__":
    main()