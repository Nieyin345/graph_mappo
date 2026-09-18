# Legacy training mode files

These files were the UI's former source of training profiles under `configs/modes/`.

The project now uses `configs/train_profiles.yaml` as the single canonical profile
registry, matching `scripts/train/train_graph_mappo.py`. The desktop UI reads and saves
the same registry, so a profile cannot silently diverge between CLI and UI anymore.

The old files are retained here for historical reference only. They are not loaded
by the training entrypoint or the UI.

Canonical profiles currently include:
- `random_episode`
- `continuous`
- `fixed_day`
- `curriculum`
- `demand_edge`
