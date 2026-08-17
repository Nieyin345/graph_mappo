"""Configuration manager for the QKD-RL UI.

Unified loading/merging/serializing of all config layers:
  default.yaml → env_*.yaml → features.yaml → graph_mappo.yaml
  → rate_provider.yaml → train_profiles.yaml → user overrides
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
OUTPUT_DIR = ROOT / "outputs"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def load_default_config() -> dict:
    """Load the full default config chain (no profile)."""
    from qkd_rl.env.factory import load_default_config as _factory_default
    config = _factory_default(ROOT)
    # env_full
    config = _deep_merge(config, _load_yaml(CONFIG_DIR / "env_full.yaml"))
    # global
    config = _deep_merge(config, _load_yaml(CONFIG_DIR / "global.yaml"))
    return config


def load_train_profiles() -> dict[str, dict]:
    """Return {profile_name: profile_dict} from configs/modes/ directory."""
    modes_dir = CONFIG_DIR / "modes"
    if not modes_dir.is_dir():
        return {}
    profiles: dict[str, dict] = {}
    for f in sorted(modes_dir.glob("*.yaml")):
        name = f.stem  # e.g. "random_episode" from "random_episode.yaml"
        profiles[name] = _load_yaml(f)
    return profiles


def get_profile_keys() -> list[str]:
    """Return sorted list of available profile names from modes/ directory."""
    return sorted(load_train_profiles().keys())


def get_profile(name: str) -> dict:
    """Return the profile dict for a given name from configs/modes/{name}.yaml."""
    path = CONFIG_DIR / "modes" / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"Unknown profile: {name}. Available modes: {get_profile_keys()}")
    return copy.deepcopy(_load_yaml(path))


def build_resolved_config(profile_name: str, overrides: dict | None = None) -> dict:
    """Build the full resolved config from profile + overrides.

    This mimics the logic in train_graph_mappo.py:build_config.
    """
    config = load_default_config()
    # Merge profile
    profile = get_profile(profile_name)
    config = _deep_merge(config, profile)
    # Merge user overrides
    if overrides:
        config = _deep_merge(config, overrides)
    # Resolve feature dims
    from qkd_rl.core.config import ConfigValidator
    config = copy.deepcopy(config)
    ConfigValidator().validate(config)
    return config


def flatten_config_for_ui(config: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested config into dot-separated keys for UI."""

    def _flatten(d: dict, parent: str) -> dict[str, Any]:
        items: dict[str, Any] = {}
        for k, v in d.items():
            key = f"{parent}.{k}" if parent else k
            if isinstance(v, dict):
                items.update(_flatten(v, key))
            else:
                items[key] = v
        return items

    return _flatten(config, "")


def unflatten_from_ui(flat: dict[str, Any]) -> dict:
    """Rebuild nested dict from dot-separated keys."""

    def _set(d: dict, keys: list[str], value: Any):
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    result: dict = {}
    for key, value in flat.items():
        _set(result, key.split("."), value)
    return result


# ---------------------------------------------------------------------------
# checkpoint browsing
# ---------------------------------------------------------------------------

def list_checkpoints() -> list[dict]:
    """List all training checkpoints in outputs/."""
    entries: list[dict] = []
    if not OUTPUT_DIR.is_dir():
        return entries
    for run_dir in sorted(OUTPUT_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        # Check for checkpoint file
        ckpt = run_dir / "checkpoint_final.pt"
        best_ckpt = run_dir / "checkpoint_best_val.pt"
        metrics = run_dir / "metrics.jsonl"
        config = run_dir / "resolved_config.yaml"
        info = {
            "name": run_dir.name,
            "path": str(run_dir),
            "has_checkpoint": ckpt.is_file() or best_ckpt.is_file(),
            "checkpoint_path": str(best_ckpt if best_ckpt.is_file() else ckpt) if ckpt.is_file() or best_ckpt.is_file() else None,
            "has_metrics": metrics.is_file(),
            "metrics_path": str(metrics) if metrics.is_file() else None,
            "has_config": config.is_file(),
            "config_path": str(config) if config.is_file() else None,
            "last_modified": run_dir.stat().st_mtime,
        }
        if info["has_metrics"]:
            try:
                lines = metrics.read_text(encoding="utf-8").strip().splitlines()
                if lines:
                    last = json.loads(lines[-1])
                    info["last_success_rate"] = last.get("mean_success_rate", None)
                    info["update_count"] = last.get("update", 0)
            except Exception:
                pass
        entries.append(info)
    entries.sort(key=lambda e: e["last_modified"], reverse=True)
    return entries


def get_baselines() -> list[str]:
    """Return available baseline policy names from run_baselines.py."""
    return [
        "random", "greedy_demand", "greedy_matching", "greedy_qkp",
        "greedy_rate", "greedy_relay", "greedy_relay_diffusion_v3", "milp",
    ]


def load_baselines_config() -> dict:
    """Load the full baseline config from baselines.yaml."""
    path = CONFIG_DIR / "baselines.yaml"
    if not path.is_file():
        return {}
    return _load_yaml(path).get("baselines", {})


def save_baselines_config(cfg: dict) -> None:
    """Save baseline config to baselines.yaml."""
    path = CONFIG_DIR / "baselines.yaml"
    data = {"baselines": cfg}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, indent=2)


def generate_command(profile_name: str, overrides: dict, run_name: str, checkpoint: str | None = None) -> str:
    """Generate the CLI command for training."""
    parts = [
        "D:\\anaconda1\\envs\\pytorch\\python.exe",
        "scripts/train_graph_mappo.py",
        f"--mode {profile_name}",
        f"--run-name {run_name}",
    ]
    # Write overrides to a temp file
    override_path = ROOT / "outputs" / ".ui_override.yaml"
    with override_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(overrides, f, sort_keys=False, allow_unicode=True)
    parts.append(f"--configs ui_override.yaml" if not Path(override_path.name).exists() else f"--configs {override_path.name}")

    # Actually, need to handle the path correctly. Let's generate the override relative to project root.
    # The override file will be written before training starts.
    parts_str = " `\n  ".join(parts)
    return parts_str, override_path