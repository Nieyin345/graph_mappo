"""Checkpoint save/load helpers for MAPPO training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CheckpointData:
    update: int
    model_state: dict
    optimizer_state: dict | None
    config: dict | None
    metrics: dict | None


def save_checkpoint(
    path: str | Path,
    update: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    config: dict | None = None,
    metrics: dict | None = None,
) -> Path:
    """Save model + optimizer state to ``path`` and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "update": int(update),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "metrics": metrics,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> CheckpointData:
    """Load a checkpoint saved by :func:`save_checkpoint`."""
    payload = torch.load(Path(path), map_location=str(device), weights_only=False)
    return CheckpointData(
        update=int(payload["update"]),
        model_state=payload["model_state"],
        optimizer_state=payload.get("optimizer_state"),
        config=payload.get("config"),
        metrics=payload.get("metrics"),
    )
