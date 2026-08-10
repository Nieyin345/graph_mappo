# -*- coding: utf-8 -*-
"""Tests for scripts/compact_h5.py (lean H5 compaction + backup)."""
from __future__ import annotations

import h5py
import numpy as np

from scripts.compact_h5 import compact_h5


def _write_full(tmp_path, t_steps: int = 1000, n_links: int = 20) -> np.ndarray:
    rng = np.random.RandomState(1)
    kmax = rng.uniform(0.0, 100.0, size=(t_steps, n_links)).astype(np.float32)
    kmax[kmax < 80.0] = 0.0  # ~80% zeros, like the real dataset
    los = (kmax > 0.0).astype(np.int8)
    distance = rng.uniform(50.0, 2000.0, size=(t_steps, n_links)).astype(np.float32)
    zenith = np.full((t_steps, n_links), np.nan, dtype=np.float32)
    dt = np.dtype([("link_id", "i4"), ("node_u", "i4"), ("node_v", "i4"), ("link_type", "S10")])
    reg = np.array(
        [(i, i, i + n_links, b"SAT-GS") for i in range(n_links)], dtype=dt
    )
    with h5py.File(tmp_path / "link_data.h5", "w") as f:
        f.create_dataset("k_max", data=kmax)
        f.create_dataset("los", data=los)
        f.create_dataset("distance", data=distance)
        f.create_dataset("zenith", data=zenith)
        f.create_dataset("link_registry", data=reg)
        f.attrs["theta_sec"] = 60
        f.attrs["min_elevation_deg"] = 5.0
    return kmax


def test_compact_h5_drops_unused_and_backs_up(tmp_path):
    kmax = _write_full(tmp_path)
    status = compact_h5(tmp_path)
    assert status["status"] == "compacted"
    assert (tmp_path / "link_data_full.h5").exists()  # original preserved
    assert (tmp_path / "link_data_compact_backup.h5").exists()  # obtained-file backup

    with h5py.File(tmp_path / "link_data.h5", "r") as f:
        assert set(f.keys()) == {"k_max", "link_registry"}
        assert f["k_max"].compression == "gzip"
        assert f.attrs["theta_sec"] == 60
        assert f.attrs["min_elevation_deg"] == 5.0
        np.testing.assert_array_equal(f["k_max"][:], kmax)
    # backup identical to lean target
    with h5py.File(tmp_path / "link_data_compact_backup.h5", "r") as f:
        np.testing.assert_array_equal(f["k_max"][:], kmax)
    # original full file still has all datasets
    with h5py.File(tmp_path / "link_data_full.h5", "r") as f:
        assert set(f.keys()) == {"k_max", "los", "distance", "zenith", "link_registry"}


def test_compact_h5_noop_when_already_lean(tmp_path):
    _write_full(tmp_path)
    compact_h5(tmp_path)
    # second run: source is already lean -> no-op, backup still ensured
    status = compact_h5(tmp_path)
    assert status["status"] == "noop"
    with h5py.File(tmp_path / "link_data.h5", "r") as f:
        assert set(f.keys()) == {"k_max", "link_registry"}
