# 2026-09 diagnostic archive

Historical one-off probes, launchers, and experiment-watch scripts from the September 2026 debugging cycle.
They were previously Git-tracked under `.tmp/`, which made a scratch directory behave like permanent source code.

These files are retained for reproducibility; they are **not active project entrypoints**. Current reusable diagnostics live directly under `scripts/diag/`.

Some launchers encode the CloudLab paths, run names, or experiment schedules that were valid at the time. Read the script header before reusing one on a new node.

New one-off probes belong in `.tmp/` and should stay untracked. If a probe becomes reusable or is cited by project documentation, promote it to `scripts/diag/` rather than tracking it in `.tmp/`.
