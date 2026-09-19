"""Catch the exact call site of the expensive edge-shaped `fill_`.

The clean profile showed ~1000 calls of `aten::fill_` on tensors of shape
[~100k, 128] (45-52 MB) per update, ~22 ms each -- together ~45% of the update,
spent allocating and zeroing memory that is immediately overwritten.

Reading the source produced a candidate that does not fit the observed shape:
the `new_zeros` at graph_mappo.py:725/734 runs once per observation (guarded by
the CPU tensor cache), so it should allocate one GRAPH's edge block (~393 rows),
not the whole minibatch's ~100k. Guessing further is wasteful, so this records
Python stack traces alongside the ops and prints the call site for the top
`fill_` entries.

Usage:
    python .tmp/probe_fill_stack.py --device cpu --steps 120 --episodes 2 --top 6
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import defaultdict
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
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--min-elems", type=int, default=100000, help="only report tensors at least this big")
    ap.add_argument("--checkpoint", default="outputs/supervised_pg_phased/supervised_pg_phased_latest.pt")
    args = ap.parse_args()

    torch.set_num_threads(1)
    tgm = load_train_module()
    ns = argparse.Namespace(
        mode="random_episode",
        configs=["rl_algorithm.yaml", "train_diag_fast.yaml"],
        seed=None,
        num_updates=1,
        run_name="probe_tmp",
        checkpoint=None,
        device=args.device,
    )
    config = tgm.build_config(ns)
    config["train"]["rollout_steps"] = args.steps
    config["train"]["episodes_per_update"] = args.episodes

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)
    out = ROOT / "outputs" / "probe_tmp"
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out, device=args.device)
    ckpt = ROOT / args.checkpoint
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)

    print("collecting rollout...", flush=True)
    buffer = trainer.collect_rollout()
    print(f"buffer {len(buffer.steps)} steps\n", flush=True)

    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        trainer.update(buffer)

    # Group by (op, input shape, stack) so each distinct call site is its own row.
    rows = []
    for ev in prof.key_averages(group_by_input_shape=True, group_by_stack_n=8):
        if "fill_" not in ev.key and "zeros" not in ev.key:
            continue
        shape = ""
        for s in (ev.input_shapes or []):
            try:
                dims = [int(d) for d in s]
            except TypeError:
                continue
            if dims and max(dims) > 1000:
                shape = "x".join(str(d) for d in dims)
                break
        if not shape:
            continue
        try:
            n_elems = 1
            for part in shape.split("x"):
                n_elems *= int(part)
        except ValueError:
            continue
        if n_elems < args.min_elems:
            continue
        rows.append((ev.self_cpu_time_total / 1000.0, ev.count, shape, ev.key, ev.stack))

    rows.sort(key=lambda r: -r[0])
    print(f"=== big alloc/fill sites (self ms, count, shape) ===")
    for ms, cnt, shape, key, stack in rows[: args.top]:
        print(f"\n--- {ms:.1f} ms   {cnt} calls   {shape}   {key}")
        frames = [f for f in (stack or "").split(";") if f.strip()]
        for f in frames[-7:]:
            print(f"      {f.strip()[:150]}")
    if not rows:
        print("(none above the element threshold)")


if __name__ == "__main__":
    main()
