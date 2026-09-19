"""Name the tensors behind the expensive `aten::fill_` / `aten::add_` calls.

The clean single-thread profile of one MAPPO update showed:

    aten::fill_   74601 ms / 26366 calls  (~2.8 ms per call)
    aten::add_    20066 ms / 19059 calls  (~1.05 ms per call)

Together 57% of a 165 s update, for operations that should be microseconds.
cProfile could not see this because the backward pass re-executes the same
elementwise ops -- profiler records them under their forward names -- so the
cost is attributed to `run_backward` as one opaque call.

2.8 ms for a fill means the tensors are large, not that the call is slow. This
runs with record_shapes=True and groups by (op, input shape) so the culprit is
named exactly instead of guessed at.

Usage:
    python .tmp/profile_fill.py --device cpu --steps 240 --episodes 8 --top 20
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


def shape_str(ev) -> str:
    shapes = []
    for s in (ev.input_shapes or []):
        try:
            shapes.append("x".join(str(d) for d in s))
        except TypeError:
            shapes.append(str(s))
    if not shapes:
        return "(no shape)"
    # Only the first input: for fill_/add_ that is the tensor being written.
    return shapes[0] + (f"  [{len(shapes)} inputs]" if len(shapes) > 1 else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--want", default="fill_,add_,copy_,index_put_,cat")
    ap.add_argument("--checkpoint", default="outputs/supervised_pg_phased/supervised_pg_phased_latest.pt")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    wanted = tuple(a.strip() for a in args.want.split(",") if a.strip())

    tgm = load_train_module()
    ns = argparse.Namespace(
        mode="random_episode",
        configs=["rl_algorithm.yaml", "train_diag_fast.yaml"],
        seed=None,
        num_updates=1,
        run_name="fill_tmp",
        checkpoint=None,
        device=args.device,
    )
    config = tgm.build_config(ns)
    config["train"]["rollout_steps"] = args.steps
    config["train"]["episodes_per_update"] = args.episodes

    print(f"threads={torch.get_num_threads()}", flush=True)
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)
    out = ROOT / "outputs" / "fill_tmp"
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
        profile_memory=True,
    ) as prof:
        trainer.update(buffer)

    groups: dict[tuple[str, str], list] = defaultdict(lambda: [0, 0.0, 0.0])
    for ev in prof.key_averages(group_by_input_shape=True):
        base = ev.key.split("::")[-1].split(".")[-1]
        if not any(w in ev.key for w in wanted):
            continue
        g = groups[(ev.key, shape_str(ev))]
        g[0] += ev.count
        g[1] += ev.self_cpu_time_total / 1000.0

    print("=== cost grouped by op + input shape (self time, single thread) ===")
    print(f"{'ms':>10}  {'count':>9}  {'ms/call':>8}  op / shape")
    ranked = sorted(groups.items(), key=lambda kv: -kv[1][1])[: args.top]
    for (op, shape), (cnt, ms, _) in ranked:
        per = ms / max(1, cnt)
        print(f"{ms:10.1f}  {cnt:9d}  {per:8.3f}  {op[:44]}  {shape[:52]}")

    total = sum(v[1] for v in groups.values())
    print(f"\nsum over these ops: {total / 1000:.1f} s")


if __name__ == "__main__":
    main()
