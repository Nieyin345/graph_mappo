"""Which field of GraphObservation is the 115 KB/step shipping cost?

A step's ship payload measured 23.1 MB for 200 steps, pickled at ~10 MB/s
(2.3 s to dump, 3.0 s to load) -- slow because GraphObservation holds plain
Python lists of floats and strings, so pickling means walking hundreds of
thousands of individual Python objects. The worker pool ships one episode's
worth (1440 steps ~ 166 MB) per episode, and the trainer unpickles all of it
serially, which is what makes the "parallel" rollout take 8x a single episode.

Sizes are reported both raw (what pickle has to walk) and after the obvious
compaction (numpy bytes), so the fix targets the field that actually dominates.

    python .tmp/obs_size.py
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qkd_rl.env.factory import build_env_from_config  # noqa: E402
from qkd_rl.rl.algos.rollout_buffer import state_free_obs  # noqa: E402


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "tgm", ROOT / "scripts" / "rl" / "train_graph_mappo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def as_compact(v):
    """Best-effort numpy form of a list-of-floats / list-of-bools field."""
    try:
        a = np.asarray(v)
    except Exception:
        return None
    if a.dtype == object or a.ndim == 0:
        return None
    return a


def main() -> None:
    tgm = load_train_module()
    ns = type("NS", (), dict(
        mode="random_episode", configs=["train_mappo.yaml"], seed=0,
        num_updates=1, run_name="obsz", checkpoint=None, device="cpu",
    ))()
    config = tgm.build_config(ns)
    env = build_env_from_config(config)
    obs = state_free_obs(env.reset(seed=0))

    rows = []
    for name in vars(obs):
        v = getattr(obs, name)
        if v is None:
            continue
        try:
            raw = len(pickle.dumps(v, protocol=4))
        except Exception as exc:  # noqa: BLE001
            rows.append((name, -1, -1, f"(unpicklable: {exc})"))
            continue
        compact = as_compact(v)
        if compact is not None:
            small = compact.nbytes
            note = f"numpy {compact.shape} {compact.dtype}"
        else:
            small = raw
            note = type(v).__name__
        rows.append((name, raw, small, note))

    rows.sort(key=lambda r: -r[1])
    print(f"{'field':26s} {'raw B':>10} {'compact B':>10}  {'ratio':>6}  note")
    tot_raw = tot_small = 0
    for name, raw, small, note in rows:
        ratio = f"{raw / small:.1f}x" if small > 0 else "-"
        print(f"{name:26s} {raw:10d} {small:10d}  {ratio:>6}  {note}")
        if raw > 0:
            tot_raw += raw
            tot_small += small
    print(f"{'TOTAL':26s} {tot_raw:10d} {tot_small:10d}  "
          f"{tot_raw / max(1, tot_small):.1f}x")
    print(f"\nper step: {tot_raw / 1e3:.0f} KB raw -> {tot_small / 1e3:.0f} KB compact")
    print(f"one 1440-step episode: {tot_raw * 1440 / 1e6:.0f} MB -> "
          f"{tot_small * 1440 / 1e6:.0f} MB")
    print(f"one 8-episode update : {tot_raw * 1440 * 8 / 1e6:.0f} MB -> "
          f"{tot_small * 1440 * 8 / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
