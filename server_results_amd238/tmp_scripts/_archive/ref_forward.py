"""Reference outputs for the encoder/critic rewrite.

The planned change replaces the single concatenated edge tensor with two
per-type tensors (physical / demand) that are never sliced. LayerNorm is
row-wise and the pooling is a plain mean, so the math should be *identical* --
but "should be" is not good enough for a rewrite that touches the forward of a
trained model. This dumps the log-probs, entropies and values for one fixed
chunk of steps so the two versions can be compared elementwise.

    python .tmp/ref_forward.py --tag before
    ...edit...
    python .tmp/ref_forward.py --tag after
    python .tmp/ref_forward.py --compare
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer  # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402

OUT = ROOT / ".tmp" / "ref_forward"


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "tgm", ROOT / "scripts" / "rl" / "train_graph_mappo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(tag: str):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="before")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--episodes", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)

    tgm = load_train_module()
    ns = argparse.Namespace(
        mode="random_episode",
        configs=["rl_algorithm.yaml", "train_diag_fast.yaml"],
        seed=0,
        num_updates=1,
        run_name="ref_tmp",
        checkpoint=None,
        device=args.device,
    )
    config = tgm.build_config(ns)
    config["train"]["rollout_steps"] = args.steps
    config["train"]["episodes_per_update"] = args.episodes

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)
    out = ROOT / "outputs" / "ref_tmp"
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out, device=args.device)
    ckpt = ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)

    buffer = trainer.collect_rollout()
    batch = list(buffer.steps)[: args.steps]
    return args, policy, model, batch


def snapshot(policy, batch, device):
    chunk = batch
    with torch.no_grad():
        results = policy.evaluate_actions_batched(
            [s.obs for s in chunk],
            [s.actions for s in chunk],
            [list(s.matched_edges or []) for s in chunk],
        )
    flat: dict[str, list[torch.Tensor]] = {"log_prob": [], "entropy": [], "value": []}
    for log_probs, entropies, value in results:
        for key, d in (("log_prob", log_probs), ("entropy", entropies)):
            for node_id in sorted(d):
                flat[key].append(d[node_id].detach().reshape(-1).double().cpu())
        flat["value"].append(value.detach().reshape(-1).double().cpu())
    return {k: torch.cat(v) if v else torch.zeros(0) for k, v in flat.items()}


def main() -> None:
    args, policy, model, batch = build("")
    OUT.mkdir(parents=True, exist_ok=True)

    if args.compare:
        a = torch.load(OUT / "before.pt")
        b = torch.load(OUT / "after.pt")
        ok = True
        for key in sorted(a):
            x, y = a[key], b[key]
            if x.shape != y.shape:
                print(f"{key:10s} SHAPE MISMATCH {tuple(x.shape)} vs {tuple(y.shape)}")
                ok = False
                continue
            diff = (x - y).abs()
            denom = x.abs().clamp(min=1e-12)
            rel = (diff / denom).max().item()
            print(
                f"{key:10s} n={x.numel():8d}  max|d|={diff.max().item():.3e}  "
                f"max rel={rel:.3e}"
            )
            if diff.max().item() > 1e-6 and rel > 1e-4:
                ok = False
        print("\nEQUIVALENT" if ok else "\nDIFFERENT")
        return

    snap = snapshot(policy, batch, args.device)
    torch.save(snap, OUT / f"{args.tag}.pt")
    for key, t in sorted(snap.items()):
        print(f"{key:10s} n={t.numel():8d}  sum={t.sum().item():.6f}  "
              f"mean={t.mean().item():.6f}")
    print(f"\nsaved {OUT / (args.tag + '.pt')}")


if __name__ == "__main__":
    main()
