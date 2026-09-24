"""Small, read-only status report for the server's paired QKD experiment."""
from __future__ import annotations

import json
import argparse
import time
import re
from pathlib import Path


ROOT = Path("/opt/qkd/ppo-stock-fix-20260922/outputs")
RUNS = (42, 43, 44)


def latest_validation(path: Path) -> tuple[int | None, dict | None]:
    if not path.exists():
        return None, None
    last_update = None
    last_validation = None
    current_update = None
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "update" in row:
            current_update = int(row["update"])
        if row.get("eval_validation") is not None:
            last_update = current_update
            last_validation = row["eval_validation"]
    return last_update, last_validation


def validation_at(path: Path, update: int | None) -> dict | None:
    if update is None or not path.exists():
        return None
    current_update = None
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "update" in row:
            current_update = int(row["update"])
        if current_update == update and row.get("eval_validation") is not None:
            return row["eval_validation"]
    return None


def running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
        return state not in "ZX"
    except (FileNotFoundError, ValueError, IndexError):
        return False


def describe(seed: int, family: str = "pairhist") -> dict:
    prefix = "pairhist_v2" if family == "pairhist" else "dminj_control"
    name = f"{prefix}_s{seed}_from_dminj30"
    directory = ROOT / name
    pid_path = directory / "train.pid"
    pid = int(pid_path.read_text().strip()) if pid_path.exists() else None
    rows = []
    metrics = directory / "metrics.jsonl"
    if metrics.exists():
        for line in metrics.read_text(errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "update" in item:
                rows.append(item)
    checkpoints = sorted(directory.glob("checkpoint_update_*.pt"))
    _, baseline = latest_validation(
        Path(f"/opt/qkd/graph_mappo/outputs/dminj_s{seed}/metrics.jsonl")
    )
    validation_update, validation = latest_validation(metrics)
    comparison = None
    if baseline and validation:
        paired = []
        old_by_seed = dict(zip(baseline.get("seeds", []), baseline.get("per_seed_success", [])))
        for val_seed, success in zip(validation.get("seeds", []), validation.get("per_seed_success", [])):
            if val_seed in old_by_seed:
                paired.append(float(success) - float(old_by_seed[val_seed]))
        comparison = {
            "baseline_mean_success_rate": baseline.get("mean_success_rate"),
            "current_mean_success_rate": validation.get("mean_success_rate"),
            "paired_mean_delta": sum(paired) / len(paired) if paired else None,
            "paired_positive_count": sum(delta > 0 for delta in paired),
            "paired_count": len(paired),
        }
    pairhist_comparison = None
    if family == "control" and validation:
        pairhist = validation_at(
            ROOT / f"pairhist_v2_s{seed}_from_dminj30" / "metrics.jsonl",
            validation_update,
        )
        if pairhist:
            pairhist_by_seed = dict(zip(pairhist.get("seeds", []), pairhist.get("per_seed_success", [])))
            paired = [
                float(success) - float(pairhist_by_seed[val_seed])
                for val_seed, success in zip(validation.get("seeds", []), validation.get("per_seed_success", []))
                if val_seed in pairhist_by_seed
            ]
            pairhist_comparison = {
                "pairhist_mean_success_rate": pairhist.get("mean_success_rate"),
                "control_mean_success_rate": validation.get("mean_success_rate"),
                "paired_mean_delta_control_minus_pairhist": sum(paired) / len(paired) if paired else None,
                "paired_positive_count": sum(delta > 0 for delta in paired),
                "paired_count": len(paired),
                "pairhist_key_efficiency": pairhist.get("mean_key_efficiency"),
                "control_key_efficiency": validation.get("mean_key_efficiency"),
            }
    final = directory / "checkpoint_final.pt"
    final_update = None
    if final.exists():
        note = directory / "finalization_note.txt"
        match = re.search(r"update (\d+)", note.read_text()) if note.exists() else None
        final_update = int(match.group(1)) if match else (rows[-1]["update"] if rows else None)
    metric_row = next(
        (row for row in reversed(rows) if final_update is None or row["update"] <= final_update),
        None,
    )
    log = directory / "train.log"
    log_tail = log.read_text(errors="replace")[-3000:] if log.exists() else ""
    alive = running(pid) if pid is not None else False
    last_activity = max(
        (path.stat().st_mtime for path in (metrics, log) if path.exists()),
        default=time.time(),
    )
    stalled = alive and time.time() - last_activity > 2 * 3600
    return {
        "seed": seed,
        "run": name,
        "pid": pid,
        "running": alive,
        "complete": final.exists(),
        "failed": not alive and not final.exists(),
        "stalled": stalled,
        "latest_update": rows[-1]["update"] if rows else 30,
        "final_update": final_update,
        "latest_metrics": {
            key: metric_row.get(key) for key in (
                "mean_success_rate", "mean_reward", "kl", "clip_frac",
                "value_return_corr", "rollout_s", "update_s"
            )
        } if metric_row else None,
        "latest_checkpoint": checkpoints[-1].name if checkpoints else None,
        "validation_update": validation_update,
        "validation_comparison": comparison,
        "pairhist_comparison": pairhist_comparison,
        "log_tail": log_tail[-700:] if (not alive and not final.exists()) or stalled else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("pairhist", "control"), default="pairhist")
    args = parser.parse_args()
    print(json.dumps({"runs": [describe(seed, args.family) for seed in RUNS]}, ensure_ascii=False))
