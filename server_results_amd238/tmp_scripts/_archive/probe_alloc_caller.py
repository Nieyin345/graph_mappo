"""Name the caller of the big `torch.zeros` allocations with a real traceback.

torch.profiler's with_stack did not populate stacks in this environment, so this
takes the direct route: wrap the Python-level allocation functions, and when one
of them produces a tensor above a size threshold, print the traceback. That
names the call site exactly.

Candidates under suspicion (from the shape evidence -- every offending alloc is
exactly "minibatch steps x entities per step", i.e. a batched shape):
    history_encoder.py:107   return torch.zeros((n_entities, hidden_dim))   cold start
    graph_mappo.py:202/219   torch.zeros_like(node_emb)
    graph_mappo.py:725/734   new_zeros((n_directed, history_dim))

Usage:
    python .tmp/probe_alloc_caller.py --device cpu --steps 120 --episodes 2
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer  # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402

MIN_ELEMS = 100_000
HITS: dict[str, list] = {}


def _record(kind: str, shape) -> None:
    try:
        n = 1
        for d in shape:
            n *= int(d)
    except (TypeError, ValueError):
        return
    if n < MIN_ELEMS:
        return
    stack = traceback.extract_stack()[:-2]
    # Keep only frames inside this repo, which is where the caller lives.
    frames = [
        f"{Path(fr.filename).name}:{fr.lineno} {fr.name}"
        for fr in stack
        if "qkd_rl" in fr.filename or "scripts" in fr.filename
    ]
    key = f"{kind} {'x'.join(str(d) for d in shape)}"
    HITS.setdefault(key, []).append(" -> ".join(frames[-6:]))


def patch_torch() -> None:
    real_zeros = torch.zeros
    real_full = torch.full

    def zeros(*a, **kw):
        out = real_zeros(*a, **kw)
        _record("torch.zeros", tuple(out.shape))
        return out

    def full(size, fill_value, *a, **kw):
        out = real_full(size, fill_value, *a, **kw)
        _record("torch.full", tuple(out.shape))
        return out

    torch.zeros = zeros
    torch.full = full

    # Tensor.new_zeros is a C method; assign on the class and fall back quietly
    # if torch refuses.
    real_new_zeros = torch.Tensor.new_zeros

    def new_zeros(self, size, *a, **kw):
        out = real_new_zeros(self, size, *a, **kw)
        _record("new_zeros", tuple(out.shape))
        return out

    try:
        torch.Tensor.new_zeros = new_zeros
    except (TypeError, AttributeError) as exc:  # pragma: no cover
        print(f"(could not patch Tensor.new_zeros: {exc})", flush=True)

    real_zeros_like = torch.zeros_like

    def zeros_like(inp, *a, **kw):
        out = real_zeros_like(inp, *a, **kw)
        _record("zeros_like", tuple(out.shape))
        return out

    torch.zeros_like = zeros_like


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
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--episodes", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(1)
    patch_torch()

    tgm = load_train_module()
    ns = argparse.Namespace(
        mode="random_episode",
        configs=["rl_algorithm.yaml", "train_diag_fast.yaml"],
        seed=None,
        num_updates=1,
        run_name="alloc_tmp",
        checkpoint=None,
        device=args.device,
    )
    config = tgm.build_config(ns)
    config["train"]["rollout_steps"] = args.steps
    config["train"]["episodes_per_update"] = args.episodes

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)
    out = ROOT / "outputs" / "alloc_tmp"
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out, device=args.device)
    ckpt = ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)

    print("collecting rollout (allocations during rollout are NOT reported)...", flush=True)
    buffer = trainer.collect_rollout()
    HITS.clear()  # only care about the update
    print(f"buffer {len(buffer.steps)} steps -- running update\n", flush=True)
    trainer.update(buffer)

    print("=== big allocations during one update ===")
    for key, callers in sorted(HITS.items(), key=lambda kv: -len(kv[1])):
        uniq = {}
        for c in callers:
            uniq[c] = uniq.get(c, 0) + 1
        print(f"\n{key}   x{len(callers)} calls")
        for c, n in sorted(uniq.items(), key=lambda kv: -kv[1])[:3]:
            print(f"      x{n}  {c}")


if __name__ == "__main__":
    main()
