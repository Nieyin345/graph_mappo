"""Where does a MAPPO update actually spend its time? (clean measurement)

Earlier attempts at this were unreliable for two reasons, both fixed here:

  1. They ran while other jobs were competing for the CPU, so the numbers were
     whatever the scheduler happened to hand out.
  2. torch.profiler's `self_cpu_time_total` is summed ACROSS THREADS, so a cheap
     op that ran on 12 threads looked 12x more expensive than its wall cost.
     Running with torch.set_num_threads(1) makes CPU time == wall time per op,
     which makes the table directly readable. The update is slower in absolute
     terms at one thread, but the BREAKDOWN is what we are after.

cProfile already established the top-level shape (run_backward ~75% of the
update, on both a laptop and a server). This script names the operators inside
that call.

Usage:
    python .tmp/profile_clean.py --device cpu --steps 240 --episodes 8 --top 18
"""
from __future__ import annotations

import argparse
import cProfile
import importlib.util
import io
import pstats
import sys
import time
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


def op_table(prof, top):
    rows = defaultdict(lambda: [0, 0.0, 0.0])  # name -> [count, fwd_ms, bwd_ms]
    for ev in prof.key_averages():
        ms = ev.self_cpu_time_total / 1000.0
        if ms <= 0:
            continue
        is_bwd = "Backward" in ev.key or ev.key.startswith("autograd::")
        b = rows[ev.key]
        b[0] += ev.count
        b[2 if is_bwd else 1] += ms
    fwd = sorted(rows.items(), key=lambda kv: -kv[1][1])[:top]
    bwd = sorted(rows.items(), key=lambda kv: -kv[1][2])[:top]
    return rows, fwd, bwd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--threads", type=int, default=1)
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
        device=args.device,
    )
    config = tgm.build_config(ns)
    config["train"]["rollout_steps"] = args.steps
    config["train"]["episodes_per_update"] = args.episodes

    print(f"torch={torch.__version__} threads={torch.get_num_threads()} "
          f"device={args.device}", flush=True)
    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)
    out = ROOT / "outputs" / "prof_tmp"
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out, device=args.device)
    ckpt = ROOT / args.checkpoint
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)

    print("collecting rollout...", flush=True)
    t0 = time.perf_counter()
    buffer = trainer.collect_rollout()
    print(f"rollout: {time.perf_counter() - t0:.2f} s ({len(buffer.steps)} steps)\n", flush=True)

    # --- 1. cProfile for the top-level shape (wall-clock attribution) ---
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    trainer.update(buffer)
    pr.disable()
    upd = time.perf_counter() - t0
    print(f"=== update wall time: {upd:.2f} s ===")
    stream = io.StringIO()
    st = pstats.Stats(pr, stream=stream)
    st.sort_stats("tottime")
    st.print_stats(8)
    lines = stream.getvalue().splitlines()
    start = next((i for i, ln in enumerate(lines) if "ncalls" in ln), 0)
    print("\n".join(lines[start:start + 10]))

    # --- 2. torch.profiler for the operator breakdown ---
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        trainer.update(buffer)

    rows, fwd, bwd = op_table(prof, args.top)
    total_f = sum(v[1] for v in rows.values())
    total_b = sum(v[2] for v in rows.values())

    print(f"\n=== op self-time split (single thread, so ms == wall ms) ===")
    print(f"  forward-side total : {total_f / 1000:8.2f} s")
    print(f"  backward-side total: {total_b / 1000:8.2f} s")
    if total_f + total_b > 0:
        print(f"  backward share     : {100 * total_b / (total_f + total_b):.1f}%")

    print(f"\n=== top {args.top} by SELF time, forward side (ms) ===")
    print(f"{'ms':>9}  {'count':>9}  op")
    for name, (cnt, f, _b) in fwd:
        if f > 0:
            print(f"{f:9.1f}  {cnt:9d}  {name[:84]}")

    print(f"\n=== top {args.top} by SELF time, BACKWARD side (ms) ===")
    print(f"{'ms':>9}  {'count':>9}  op")
    for name, (cnt, _f, b) in bwd:
        if b > 0:
            print(f"{b:9.1f}  {cnt:9d}  {name[:84]}")


if __name__ == "__main__":
    main()
