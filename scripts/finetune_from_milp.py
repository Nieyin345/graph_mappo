"""Fine-tune RL checkpoint using MILP demos (behavior cloning).

Loads checkpoint → forward MILP demo steps through actor → computes
cross-entropy loss between actor logits and MILP actions → updates actor.

Usage:
    python scripts/finetune_from_milp.py
    python scripts/finetune_from_milp.py --epochs 10 --lr 1e-4 --checkpoint path/to/ckpt.pt
"""
from __future__ import annotations
import os, sys, time, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
from qkd_rl.core.config import (ConfigValidator, deep_merge, load_config)
from qkd_rl.env.factory import load_default_config, build_env_from_config
from qkd_rl.algos.checkpoint import load_checkpoint, save_checkpoint
from qkd_rl.models.graph_mappo import GraphMAPPOActorCritic


def _make_obs_for_bc(step_data, action_space):
    """Build minimal GraphObservation for actor forward from stored demo step."""
    from qkd_rl.env.graph_builder import GraphObservation
    from qkd_rl.env.state import EnvState

    state = EnvState.__new__(EnvState)
    state.t = step_data.get("t", 0)

    masks = step_data.get("action_masks", {})
    return GraphObservation(
        node_features=step_data["node_features"],
        edge_index=step_data["edge_index"],
        edge_features=step_data["edge_features"],
        node_ids=step_data["node_ids"],
        edge_ids=step_data.get("edge_ids", []),
        physical_edge_ids=step_data["physical_edge_ids"],
        demand_edge_ids=[],
        action_candidates=step_data.get("action_candidates", {}),
        action_masks=masks,
        raw_action_masks={n: [bool(m) for m in ms] for n, ms in masks.items()},
        state=state,
    )


def _find_action_index(obs, node_id, action_str, action_space):
    """Return the index of `action_str` in the candidate list for `node_id`."""
    cands = obs.action_candidates.get(node_id, [])
    for i, a in enumerate(cands):
        if a == action_str:
            return i
    return -1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", type=str, default="outputs/milp_demos")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    demos_dir = ROOT / args.demos
    demo_files = sorted(demos_dir.glob("episode_*.pt"))
    if not demo_files:
        print(f"No demo files in {demos_dir}"); return

    # Build env for action space
    config = load_default_config(ROOT)
    config = deep_merge(config, load_config([ROOT / "configs" / "env_full.yaml"]))
    config = deep_merge(config, load_config([ROOT / "configs" / "global.yaml"]))
    config["env"]["episode_steps"] = 240
    config["scenario"]["time_limit"]["days"] = 366
    ConfigValidator().validate(config)
    env = build_env_from_config(config)
    action_space = env.action_resolver.action_space

    # Build model
    model = GraphMAPPOActorCritic(action_space, config)

    # Load checkpoint
    all_ckpts = sorted(ROOT.glob("outputs/exp_*/checkpoint_final.pt"))
    if not all_ckpts:
        all_ckpts = sorted(ROOT.glob("outputs/exp_*/checkpoint_best_val.pt"))
    ckpt_path = args.checkpoint or (str(all_ckpts[-1]) if all_ckpts else None)
    if ckpt_path:
        data = load_checkpoint(ckpt_path, "cpu")
        model.load_state_dict(data.model_state)
        print(f"Loaded: {ckpt_path} (update={data.update})")
    else:
        print("Random init (no checkpoint).")

    model.train()
    opt = torch.optim.Adam(model.actor.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load all steps
    all_steps = []
    for f in demo_files:
        demo = torch.load(f, map_location="cpu", weights_only=False)
        for s in demo["trajectory"]:
            all_steps.append(s)
    print(f"Demo steps: {len(all_steps)}, device: {device}")

    for epoch in range(args.epochs):
        random.shuffle(all_steps)
        total_loss = 0.0
        n_valid = 0
        t0 = time.perf_counter()
        for step_data in all_steps:
            try:
                obs = _make_obs_for_bc(step_data, action_space)
                milp_actions = step_data["milp_actions"]
                # Forward model → get logits
                out = model.forward(obs, device=str(device), build_logits_dict=True)
                # For each node, compute cross-entropy: -log(softmax(logits)[action_idx])
                loss = 0.0
                for node_id in obs.node_ids:
                    action = milp_actions.get(node_id, "idle")
                    logits = out.logits.get(node_id)
                    if logits is None or logits.numel() == 0:
                        continue
                    # Find the action index
                    cands = obs.action_candidates.get(node_id, [])
                    aidx = -1
                    for i, a in enumerate(cands):
                        if a == action:
                            aidx = i
                            break
                    if aidx < 0 or aidx >= logits.size(0):
                        continue
                    # log_softmax then negative log prob
                    lp = torch.log_softmax(logits, dim=0)
                    loss -= lp[aidx]
                if loss == 0.0:
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)
                opt.step()
                total_loss += float(loss.detach().cpu())
                n_valid += 1
            except Exception as e:
                continue
        avg_loss = total_loss / max(1, n_valid) if n_valid else 0.0
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} ({n_valid}/{len(all_steps)} steps, "
              f"{time.perf_counter()-t0:.0f}s)")

    out_path = args.out or str(demos_dir / "finetuned.pt")
    save_checkpoint(out_path, 9999, model, metrics={"bc_loss": total_loss / max(1, n_valid) if n_valid else 0,
               "epochs": args.epochs, "demos": len(demo_files)})
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()