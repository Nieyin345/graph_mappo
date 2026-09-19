"""Summarise a real run: per-update timings, and the config that produced them.

The full-scale runs are an order of magnitude slower than the diag benchmark
(update 381/998/415 s vs 21 s), which is more than the 6x larger buffer
explains. This dumps the numbers next to the resolved config so the comparison
is against what the run actually used, not against what a config file says
today.

    python .tmp/run_report.py outputs/full_rnd15 outputs/full_d8
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


def main() -> None:
    dirs = sys.argv[1:] or ["outputs/full_rnd15"]
    for d in dirs:
        p = Path(d)
        print(f"===== {p} =====")

        cfg_path = p / "resolved_config.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text())
            train = cfg.get("train", {})
            ppo = train.get("ppo", {})
            print("  -- train --")
            for k in (
                "rollout_steps",
                "episodes_per_update",
                "n_rollout_workers",
                "rollout_worker_device",
                "rollout_batch",
                "value_target",
            ):
                print(f"     {k:24s} {train.get(k)!r}")
            print("  -- ppo --")
            for k in (
                "epochs",
                "minibatch_size",
                "batch_chunk",
                "clip_eps",
                "entropy_coef",
                "value_coef",
                "target_kl",
                "normalize_advantages",
            ):
                print(f"     {k:24s} {ppo.get(k)!r}")
            print("  -- runtime --")
            print(f"     {cfg.get('runtime', {})!r}")
            hist = cfg.get("features", {}).get("history_encoder", {})
            print(f"     history_encoder.enabled  {hist.get('enabled')!r}")
            print(f"     graph.num_layers         {cfg.get('graph', {}).get('num_layers')!r}")
            print(f"     graph.hidden_dim         {cfg.get('graph', {}).get('hidden_dim')!r}")

        m = p / "metrics.jsonl"
        if m.exists():
            print("  -- per update --")
            rows = [json.loads(line) for line in m.read_text().splitlines() if line.strip()]
            print(f"     {'upd':>4} {'rollout_s':>10} {'update_s':>9} {'elapsed_s':>10} {'nb':>4} {'success':>8}")
            for r in rows:
                print(
                    f"     {r.get('update', '?'):>4} {r.get('rollout_s', 0):>10.1f} "
                    f"{r.get('update_s', 0):>9.1f} {r.get('elapsed_s', 0):>10.1f} "
                    f"{r.get('n_minibatches', '?'):>4} {r.get('mean_success_rate', 0):>8.4f}"
                )
            tot = sum(r.get("elapsed_s", 0) for r in rows)
            if rows:
                print(f"     mean elapsed {tot / len(rows):.1f} s over {len(rows)} updates")
        print()


if __name__ == "__main__":
    main()
