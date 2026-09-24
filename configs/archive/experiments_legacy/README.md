# Legacy experiment presets

These overlays are retained for reproducibility but are **not current training recommendations**.
They were moved out of the top-level `configs/` directory to prevent accidental use as active presets.

- `train_speed_optimized.yaml`: old CPU/16GB tuning overlay from the pre-2026-09-18 snapshot; current server tuning lives in `rl_algorithm.yaml`.
- `var2_smoke.yaml`: one-off validation smoke overlay used when `per_seed_success` was first added.

Active experiments may remain in top-level `configs/` even when they are not referenced by source code; do not infer obsolescence from reference count alone.
- `_exp_chunk128.yaml` / `_exp_chunk256.yaml`: completed PPO `batch_chunk` cache/throughput probes from the old server tuning cycle.
- `train_v1_on_path.yaml`: historical overlay whose v1 behavior lived in the old `_compute_on_pending_path` implementation; under current code its YAML is identical to v2 and therefore cannot reproduce v1 by itself.
