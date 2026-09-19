"""Identify which operator CONTAINS the expensive `aten::fill_` calls.

Two earlier probes narrowed this down but could not name the culprit:

  * Python-level `torch.zeros` / `zeros_like` / `new_zeros` in the model account
    for only 9 large allocations per update (99 MB total) -- nowhere near the
    10.4 s the profiler attributes to ~496 fills of a ~40 MB edge tensor.
  * `with_stack=True` produced empty stacks in this environment.

So the fills originate inside C++ (autograd gradient-buffer zeroing, or an
internal helper of an op on the edge path). This exports a chrome trace and, for
every large `fill_` event, finds the smallest enclosing event -- i.e. the parent
operator that caused it -- then aggregates by that parent.

Usage:
    python .tmp/trace_fill_parent.py --device cpu --steps 120 --episodes 2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.algos.mappo_trainer import MAPPOTrainer  # noqa: E402
from qkd_rl.rl.algos.policy import MAPPOPolicy  # noqa: E402
from qkd_rl.rl.models.graph_mappo import GraphMAPPOActorCritic  # noqa: E402

MIN_ELEMS = 100_000


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "tgm", ROOT / "scripts" / "rl" / "train_graph_mappo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def event_elems(ev) -> int:
    dims = None
    for key in ("Input Dims", "InputDim", "dims"):
        v = (ev.get("args") or {}).get(key)
        if v:
            dims = v
            break
    if not dims:
        return 0
    best = 0
    for shape in dims:
        try:
            n = 1
            for d in shape:
                n *= int(d)
        except (TypeError, ValueError):
            continue
        best = max(best, n)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    torch.set_num_threads(1)
    tgm = load_train_module()
    ns = argparse.Namespace(
        mode="random_episode",
        configs=["rl_algorithm.yaml", "train_diag_fast.yaml"],
        seed=None,
        num_updates=1,
        run_name="trace_tmp",
        checkpoint=None,
        device=args.device,
    )
    config = tgm.build_config(ns)
    config["train"]["rollout_steps"] = args.steps
    config["train"]["episodes_per_update"] = args.episodes

    env = build_env_from_config(config)
    model = GraphMAPPOActorCritic(env.action_resolver.action_space, config)
    policy = MAPPOPolicy(model, args.device)
    out = ROOT / "outputs" / "trace_tmp"
    out.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(env, policy, config, out, device=args.device)
    ckpt = ROOT / "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt"
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)

    print("collecting rollout...", flush=True)
    buffer = trainer.collect_rollout()
    print(f"buffer {len(buffer.steps)} steps\n", flush=True)

    from torch.profiler import ProfilerActivity, profile

    trace_path = Path(tempfile.gettempdir()) / "qkd_trace.json"
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        trainer.update(buffer)
    prof.export_chrome_trace(str(trace_path))
    print(f"trace written to {trace_path}", flush=True)

    with trace_path.open("r", encoding="utf-8") as f:
        trace = json.load(f)
    events = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"]
    trace_path.unlink(missing_ok=True)

    # Containment is reported in microseconds; an event spans [ts, ts+dur).
    #
    # The naive scan is O(events x fills), which is hopeless on a multi-million
    # event trace, so candidates are pre-filtered: a parent must last at least
    # as long as the fill it contains, which removes almost everything (the vast
    # majority of ops are sub-millisecond). Candidates are kept sorted by
    # duration ascending so the first one that contains the fill is the smallest.
    fills = [
        e for e in events
        if "fill_" in e.get("name", "") and event_elems(e) >= MIN_ELEMS
    ]
    print(f"large fill_ events: {len(fills)}  (of {len(events)} total)", flush=True)
    if fills:
        max_fill_dur = max(f.get("dur", 0) for f in fills)
        candidates = sorted(
            (e for e in events if e.get("dur", 0) >= max(1, max_fill_dur * 0.2)),
            key=lambda e: e.get("dur", 0),
        )
        print(f"containment candidates: {len(candidates)}\n", flush=True)
    else:
        candidates = []

    parents: dict[str, list] = defaultdict(lambda: [0, 0.0, 0])
    for fl in fills:
        f_start, f_end = fl["ts"], fl["ts"] + fl.get("dur", 0)
        chain = []
        for e in candidates:
            if e is fl:
                continue
            if e["ts"] <= f_start and e["ts"] + e.get("dur", 0) >= f_end:
                chain.append(e)
                if len(chain) >= 5:
                    break
        # chain is sorted by duration ascending: chain[0] is the innermost
        # enclosing op (usually `aten::zero_` itself), and the LAST entry is the
        # outermost -- the backward node that allocated the buffer.
        pname = " <- ".join(e["name"] for e in chain[:4]) if chain else "(no enclosing op)"
        rec = parents[pname]
        rec[0] += 1
        rec[1] += fl.get("dur", 0) / 1000.0
        rec[2] = max(rec[2], event_elems(fl))

    print(f"{'ms':>10}  {'count':>7}  {'max elems':>12}  parent operator")
    for name, (cnt, ms, mx) in sorted(parents.items(), key=lambda kv: -kv[1][1])[: args.top]:
        print(f"{ms:10.1f}  {cnt:7d}  {mx:12d}  {name[:70]}")

    total = sum(v[1] for v in parents.values())
    print(f"\ntotal large-fill time: {total / 1000:.1f} s")


if __name__ == "__main__":
    main()
