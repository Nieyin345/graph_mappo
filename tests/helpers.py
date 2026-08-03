"""Shared test scaffolding: build the small QKD env from a tiny H5 dataset.

The production rate provider is H5-only. Tests exercise the exact same
``H5RateProvider`` code path against a small generated H5 file (the
small-scenario topology), so no mock/fake data generator lives in the
package.
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qkd_rl.core.types import LinkType
from qkd_rl.data.scenario_builder import ScenarioBuilder
from qkd_rl.env.factory import build_env_from_config, load_default_config

# One shared tiny H5 dataset for the whole test session.
SMALL_H5_DIR = Path(tempfile.gettempdir()) / "qkd_rl_test_h5"

_LINK_TYPE_NAME = {
    LinkType.GS_HAP: "HAP-GS",
    LinkType.GS_SAT: "SAT-GS",
    LinkType.HAP_SAT: "SAT-HAP",
    LinkType.SAT_SAT: "SAT-SAT",
}


def ensure_small_h5(data_dir: Path = SMALL_H5_DIR, t_steps: int = 5000) -> Path:
    """Idempotently write the small-scenario H5 dataset + registries."""
    data_dir = Path(data_dir)
    link_data = data_dir / "link_data.h5"
    if link_data.exists():
        return data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    config = load_default_config(ROOT)
    scenario = ScenarioBuilder(config).build_small()
    node_by_id = {node.node_id: idx for idx, node in enumerate(scenario.nodes)}
    edges = scenario.edges
    link_type_name = [""] * len(edges)
    for idx, edge in enumerate(edges):
        link_type_name[idx] = _LINK_TYPE_NAME[edge.link_type]

    with (data_dir / "node_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_id", "type", "name", "lat", "lon", "alt_km"])
        writer.writeheader()
        for node in scenario.nodes:
            writer.writerow(
                {
                    "node_id": node_by_id[node.node_id],
                    "type": node.node_type.value.upper(),
                    "name": node.node_id,
                    "lat": 0.0,
                    "lon": 0.0,
                    "alt_km": (node.alt_m or 0.0) / 1000.0,
                }
            )

    with (data_dir / "link_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["link_id", "node_u", "node_v", "link_type"])
        writer.writeheader()
        for idx, edge in enumerate(edges):
            writer.writerow(
                {
                    "link_id": idx,
                    "node_u": node_by_id[edge.src],
                    "node_v": node_by_id[edge.dst],
                    "link_type": link_type_name[idx],
                }
            )

    n_links = len(edges)
    rng = np.random.RandomState(0)
    registry = np.zeros(n_links, dtype=[("link_id", "<i4"), ("node_u", "<i4"), ("node_v", "<i4"), ("link_type", "S10")])
    for idx, edge in enumerate(edges):
        registry[idx] = (
            idx,
            node_by_id[edge.src],
            node_by_id[edge.dst],
            link_type_name[idx].encode("utf-8"),
        )
    with h5py.File(link_data, "w") as f:
        f.create_dataset("k_max", data=rng.uniform(0.0, 50.0, size=(t_steps, n_links)).astype(np.float32))
        f.create_dataset("los", data=rng.randint(0, 2, size=(t_steps, n_links)).astype(np.int8))
        f.create_dataset("distance", data=np.zeros((t_steps, n_links), dtype=np.float32))
        f.create_dataset("zenith", data=np.zeros((t_steps, n_links), dtype=np.float32))
        f.create_dataset("link_registry", data=registry)
        f.attrs["theta_sec"] = 60
    return data_dir


def point_config_to_h5(config: dict, data_dir: Path = SMALL_H5_DIR) -> dict:
    """Point a config at the small H5 dataset and force the H5 provider."""
    ensure_small_h5(data_dir)
    config["rate_provider"]["provider"] = "h5"
    config["rate_provider"]["h5"]["dataset_dir"] = str(data_dir)
    return config


def build_test_env(project_root=".", data_dir: Path = SMALL_H5_DIR):
    """Build the small-scenario env backed by the tiny H5 dataset."""
    config = point_config_to_h5(load_default_config(project_root), data_dir)
    return build_env_from_config(config)
