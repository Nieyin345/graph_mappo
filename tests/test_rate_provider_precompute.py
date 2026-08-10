"""Tests for the precomputed training-time rate interface.

Covers:
- ``EdgeWindow.rates_norm`` matches per-element ``RateNormalizer`` exactly.
- Chunk-cached reads stay correct across chunk boundaries and after LRU
  eviction (no stale / missing rows when walking the whole dataset).
- Same-``t`` window-dict caching (``step()`` builds the state twice per tick).
- Dataset-end padding still produces zero rates / unavailable slots.
"""

from __future__ import annotations

import csv

import h5py
import numpy as np
import pytest

from qkd_rl.core.types import Edge, LinkType
from qkd_rl.link.rate_provider import H5RateProvider, RateNormalizer
from tests.test_h5_rate_provider import NODES, LINKS, _edges


def _write_chunked_dataset(tmp_path, t_steps: int = 3000, chunk_rows: int = 512) -> dict:
    with (tmp_path / "node_registry.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "type", "name", "lat", "lon", "alt_km"])
        writer.writerows(NODES)
    with (tmp_path / "link_registry.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["link_id", "node_u", "node_v", "link_type"])
        writer.writerows(LINKS)
    L = len(LINKS)
    rng = np.random.RandomState(7)
    kmax = rng.uniform(0.0, 30.0, size=(t_steps, L)).astype(np.float32)
    los = np.ones((t_steps, L), dtype=np.int8)
    los[500:900, 0] = 0
    kmax[500:900, 0] = 0.0
    kmax[1200, 1] = np.nan
    with h5py.File(tmp_path / "link_data.h5", "w") as f:
        f.create_dataset("k_max", data=kmax, chunks=(chunk_rows, L), compression="gzip", compression_opts=4)
        f.create_dataset("los", data=los, chunks=(chunk_rows, L), compression="gzip", compression_opts=4)
        dt = np.dtype([("link_id", "i4"), ("node_u", "i4"), ("node_v", "i4"), ("link_type", "S10")])
        reg = np.array([(link_id, u, v, link_type.encode()) for link_id, u, v, link_type in LINKS], dtype=dt)
        f.create_dataset("link_registry", data=reg)
        f.attrs["theta_sec"] = 60
    return {"kmax": kmax, "los": los, "T": t_steps, "L": L}


def _provider(tmp_path, horizon: int = 6, min_link_rate: float = 0.5, normalizer: RateNormalizer | None = None):
    cfg = {
        "h5": {"dataset_dir": str(tmp_path)},
        "rate": {"min_link_rate": min_link_rate, "unavailable_if_nan": True},
    }
    provider = H5RateProvider(cfg, seed=0)
    provider.setup(_edges(), horizon=horizon, min_link_rate=min_link_rate, normalizer=normalizer)
    return provider


def test_rates_norm_matches_scalar_normalizer(tmp_path):
    _write_chunked_dataset(tmp_path)
    norm_cfg = {"mode": "log_p99", "p99": None, "eps": 1.0e-8, "stats_path": None, "h5": {"dataset_dir": str(tmp_path)}}
    normalizer = RateNormalizer(norm_cfg)
    provider = _provider(tmp_path, normalizer=normalizer)
    edges = _edges()
    for t in (0, 100, 511, 512, 513, 1200, 1500):
        windows = provider.get_all_edge_windows(t)
        for edge in edges:
            window = windows[edge.edge_id]
            expected = [normalizer.transform_scalar(rate, edge.link_type.value) for rate in window.rates]
            assert window.rates_norm == pytest.approx(expected, rel=1e-5, abs=1e-6)
    provider.close()


def test_chunk_cache_correct_across_boundaries_and_eviction(tmp_path):
    data = _write_chunked_dataset(tmp_path)
    provider = _provider(tmp_path)
    # Force a tiny cache so LRU eviction happens on every chunk crossing.
    provider._chunk_cache_size = 2
    horizon = provider.horizon
    for t in range(0, data["T"] - horizon, 173):
        windows = provider.get_all_edge_windows(t)
        for idx, edge in enumerate(_edges()):
            window = windows[edge.edge_id]
            expected = data["kmax"][t : t + horizon + 1, idx]
            cleaned = np.where(np.isnan(expected), 0.0, np.maximum(expected, 0.0))
            assert window.rates == pytest.approx(cleaned.tolist(), abs=1e-5)
            exp_avail = (cleaned >= provider.min_link_rate) & (data["los"][t : t + horizon + 1, idx] == 1)
            assert window.available == exp_avail.tolist()
    provider.close()


def test_same_t_window_dict_is_cached(tmp_path):
    _write_chunked_dataset(tmp_path)
    provider = _provider(tmp_path)
    w1 = provider.get_all_edge_windows(1000)
    w2 = provider.get_all_edge_windows(1000)
    assert w1 is w2
    # A different t rebuilds the dict.
    w3 = provider.get_all_edge_windows(1001)
    assert w3 is not w1
    assert set(w3) == set(w1)
    provider.close()


def test_chunked_window_year_end_padding(tmp_path):
    data = _write_chunked_dataset(tmp_path)
    provider = _provider(tmp_path)
    window = provider.get_all_edge_windows(data["T"] - 2)
    edge = _edges()[0]
    w = window[edge.edge_id]
    assert len(w.rates) == provider.horizon + 1
    assert w.rates[-1] == 0.0
    assert w.available[-1] is False
    assert w.rates_norm is None  # no normalizer passed -> fallback path
    provider.close()


def test_get_edge_window_with_normalizer(tmp_path):
    _write_chunked_dataset(tmp_path)
    norm_cfg = {"mode": "log_p99", "p99": None, "eps": 1.0e-8, "stats_path": None, "h5": {"dataset_dir": str(tmp_path)}}
    normalizer = RateNormalizer(norm_cfg)
    provider = _provider(tmp_path, normalizer=normalizer)
    edge = _edges()[0]
    window = provider.get_edge_window(edge.edge_id, 600)
    expected = [normalizer.transform_scalar(rate, edge.link_type.value) for rate in window.rates]
    assert window.rates_norm == pytest.approx(expected, rel=1e-5, abs=1e-6)
    provider.close()
