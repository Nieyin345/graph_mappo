from __future__ import annotations

import csv
import json
import math
import warnings
import zlib

import numpy as np
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from qkd_rl.core.types import Edge, H5_LINK_TYPE_MAP, LinkType


@dataclass
class EdgeWindow:
    edge_id: str
    rates: list[float]
    available: list[bool]
    link_type: LinkType
    # Precomputed normalized rates (RateNormalizer applied once per window
    # block, vectorized). None when no normalizer is configured (legacy /
    # direct-construction callers), in which case GraphBuilder falls back to
    # per-element RateNormalizer.transform_scalar.
    rates_norm: list[float] | None = None


class LazyEdgeWindows:
    """Dict-like window store backed by the cached numpy window blocks.

    ``EnvState.edge_windows`` is a ``LazyEdgeWindows`` during training: the
    per-edge ``EdgeWindow`` objects (each O(H) Python lists) are materialized
    only for edges that are actually read (active edges, activated links,
    resolver candidates) instead of all 4241 links every step. Column gather
    helpers keep the mask builder and node availability fully vectorized.
    """

    def __init__(self, provider, t: int, blocks):
        self._provider = provider
        self._t = t
        self._blocks = blocks
        self._cache: dict[str, EdgeWindow] = {}

    @property
    def blocks(self):
        return self._blocks

    def __getitem__(self, edge_id: str) -> EdgeWindow:
        window = self._cache.get(edge_id)
        if window is None:
            if self._provider is None:
                raise KeyError(edge_id)
            window = self._provider.edge_window_from_blocks(edge_id, self._blocks)
            self._cache[edge_id] = window
        return window

    def __getstate__(self) -> dict:
        # RolloutStep obs are pickled when workers hand results back to the
        # trainer. The provider holds an open h5py handle that cannot cross
        # process boundaries, so only the materialized EdgeWindows survive the
        # pickle; everything else is rebuilt in the destination process.
        return {"cache": dict(self._cache)}

    def __setstate__(self, state: dict) -> None:
        self._cache = state["cache"]
        self._provider = None
        self._t = 0
        self._blocks = None

    def get(self, edge_id: str, default=None):
        try:
            return self[edge_id]
        except KeyError:
            return default

    def __contains__(self, edge_id: str) -> bool:
        return edge_id in self._provider._edge_to_link

    def __len__(self) -> int:
        return len(self._provider._edge_to_link)

    def keys(self):
        return self._provider._edge_to_link.keys()

    def items(self):
        for edge_id in self._provider._edge_to_link:
            yield edge_id, self[edge_id]

    def edge_columns(self, edge_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized ``(available[0], rates[0])`` at t for the given edges."""
        idx = self._provider.edge_link_ids(edge_ids)
        return self._blocks[1][0][idx], self._blocks[0][0][idx]

    def link_ids(self, edge_ids: list[str]) -> np.ndarray:
        """Link ids for the given edge ids (vectorized gather index)."""
        return self._provider.edge_link_ids(edge_ids)

    def available0(self, edge_ids: list[str]) -> np.ndarray:
        """Vectorized ``available[0]`` at t for the given edges."""
        idx = self._provider.edge_link_ids(edge_ids)
        return self._blocks[1][0][idx]

    def rates0(self, edge_ids: list[str]) -> np.ndarray:
        """Vectorized ``rates[0]`` at t for the given edges."""
        idx = self._provider.edge_link_ids(edge_ids)
        return self._blocks[0][0][idx]


class RateProvider(Protocol):
    def setup(self, edges: list[Edge], horizon: int, min_link_rate: float) -> None:
        ...

    def get_rate(self, edge_id: str, t: int) -> float:
        ...

    def is_available(self, edge_id: str, t: int) -> bool:
        ...

    def get_edge_window(self, edge_id: str, t: int) -> EdgeWindow:
        ...

    def get_all_edge_windows(self, t: int) -> dict[str, EdgeWindow]:
        ...


_H5_TYPE_TO_VALUE = {name: link_type.value for name, link_type in H5_LINK_TYPE_MAP.items()}


class RateNormalizer:
    """Rate feature normalization calibrated from ``dataset/global/rate_stats.json``.

    ``mode=log_p99`` maps each rate via ``log1p(rate) / log1p(p99)``. The p99
    denominator is read from the stats file written by
    ``scripts/estimate_rate_stats.py`` (global and per-link-type); an explicit
    ``p99`` in the config overrides it. ``stats_path: null`` falls back to
    ``<dataset_dir>/rate_stats.json``.
    """

    def __init__(self, config: dict):
        self.config = config
        self.mode = config.get("mode", "identity")
        self.p99 = float(config.get("p99") or 10.0)
        self.eps = float(config.get("eps", 1.0e-8))
        self.per_type_p99: dict[str, float] = {}

        stats_path = config.get("stats_path")
        if not stats_path:
            dataset_dir = Path(config.get("h5", {}).get("dataset_dir", "dataset/global"))
            stats_path = dataset_dir / "rate_stats.json"
        stats_path = Path(stats_path)
        if stats_path.exists():
            try:
                with stats_path.open("r", encoding="utf-8") as fh:
                    stats = json.load(fh)
                self.p99 = float(stats["global"]["p99"])
                for name, per in stats.get("per_link_type", {}).items():
                    value = _H5_TYPE_TO_VALUE.get(name)
                    if value is not None:
                        self.per_type_p99[value] = float(per["p99"])
            except (OSError, KeyError, ValueError) as exc:
                warnings.warn(f"Failed to load rate stats {stats_path}: {exc}")
        if config.get("p99"):
            self.p99 = float(config["p99"])

    def transform_scalar(self, value: float, link_type: str | None = None) -> float:
        if self.mode != "log_p99":
            return value
        p99 = self.per_type_p99.get(link_type, self.p99)
        return math.log1p(max(value, 0.0)) / max(math.log1p(p99), self.eps)

    def inv_log1p_p99(self, link_type: str | None = None) -> float:
        """Precomputed multiplier ``1 / log1p(p99)`` for a link type."""
        if self.mode != "log_p99":
            return 1.0
        p99 = self.per_type_p99.get(link_type, self.p99)
        return 1.0 / max(math.log1p(p99), self.eps)


class H5RateProvider:
    """Read link rates / LOS availability from the H5 dataset produced by
    ``build_global_tensor.py`` (``dataset/global/link_data.h5``).

    Datasets (all shaped ``(T, L)`` unless noted):

    - ``link_registry``: structured array (link_id, node_u, node_v, link_type)
    - ``k_max``: secure key rate in bits-per-second (float32)
    - ``los``: line-of-sight flag (int8, 1 = visible)
    - ``distance``: km (float32), ``zenith``: degrees (float32, NaN off-LOS)

    Node/link registries come from ``node_registry.csv`` + ``link_registry.csv``
    (falling back to the H5 ``link_registry`` dataset). Env ``Edge`` endpoints
    are matched to H5 node names; override with ``node_name_map`` when the
    scenario uses different node ids.
    """

    def __init__(self, config: dict, seed: int = 0):
        self.config = config
        self.seed = seed
        h5_cfg = config.get("h5", {})
        dataset_dir = Path(h5_cfg.get("dataset_dir", "dataset/global"))
        self.dataset_dir = dataset_dir
        self.link_data_path = Path(h5_cfg.get("link_data_path") or dataset_dir / "link_data.h5")
        self.node_registry_path = Path(h5_cfg.get("node_registry_path") or dataset_dir / "node_registry.csv")
        self.link_registry_path = Path(h5_cfg.get("link_registry_path") or dataset_dir / "link_registry.csv")
        self.rate_dataset_key = h5_cfg.get("rate_dataset_key", "k_max")
        self.los_dataset_key = h5_cfg.get("los_dataset_key", "los")
        self.distance_dataset_key = h5_cfg.get("distance_dataset_key", "distance")
        self.zenith_dataset_key = h5_cfg.get("zenith_dataset_key", "zenith")
        self.availability_source = h5_cfg.get("availability_source", "los_and_rate")
        self.out_of_range_policy = h5_cfg.get("out_of_range_policy", "zero")
        self.node_name_map = dict(h5_cfg.get("node_name_map", {}) or {})

        self.edges: dict[str, Edge] = {}
        self.horizon = 0
        self.min_link_rate = 0.0
        self._file = None
        self._edge_to_link: dict[str, int] = {}
        self._edge_link_type: dict[str, LinkType] = {}
        self._pair_to_link: dict[tuple[str, str], int] = {}
        self._T = 0
        self._L = 0
        self.slot_seconds = 60
        self._window_t: int | None = None
        self._window_windows: dict[str, EdgeWindow] | None = None
        self._blocks_t: int | None = None
        self._blocks: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None = None
        self._has_los = False
        self._los_warned = False
        self._chunk_rows: int | None = None
        self._chunk_caches: dict[str, OrderedDict[int, np.ndarray]] = {}
        self._chunk_cache_size = 8
        self._inv_log1p_p99: np.ndarray | None = None
        self._normalizer: RateNormalizer | None = None
        self._link_type_by_link: dict[int, LinkType] = {}
        self.negative_rate_policy = self.config.get("rate", {}).get("negative_rate_policy", "clip_to_zero")

    # ---------- lifecycle ----------
    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        self._chunk_caches = {}
        self._window_windows = None
        self._blocks = None
        self._blocks_t = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def time_bounds(self) -> tuple[int, int]:
        return (0, self._T)

    def setup(
        self, edges: list[Edge], horizon: int, min_link_rate: float, normalizer: RateNormalizer | None = None
    ) -> None:
        self.edges = {edge.edge_id: edge for edge in edges}
        self.horizon = horizon
        self.min_link_rate = min_link_rate
        self._normalizer = normalizer
        self._load_registries()
        for edge in edges:
            src_name = self._resolve_name(edge.src)
            dst_name = self._resolve_name(edge.dst)
            pair = tuple(sorted((src_name, dst_name)))
            link_id = self._pair_to_link.get(pair)
            if link_id is None:
                raise ValueError(
                    f"No H5 link for edge {edge.edge_id!r} ({src_name!r} -- {dst_name!r}). "
                    "Check `node_name_map` / node_registry / link_registry."
                )
            self._edge_to_link[edge.edge_id] = link_id
            self._edge_link_type[edge.edge_id] = edge.link_type
        self._open()
        self._prepare_normalization()

    # ---------- RateProvider interface ----------
    def get_rate(self, edge_id: str, t: int) -> float:
        link_id = self._edge_to_link[edge_id]
        return self._clean_rate(self._read_scalar(self.rate_dataset_key, t, link_id))

    def is_available(self, edge_id: str, t: int) -> bool:
        link_id = self._edge_to_link[edge_id]
        rate = self._read_scalar(self.rate_dataset_key, t, link_id)
        los = self._read_scalar(self.los_dataset_key, t, link_id) if self._has_los else None
        return self._available_from(rate, los)

    def get_edge_window(self, edge_id: str, t: int) -> EdgeWindow:
        link_id = self._edge_to_link[edge_id]
        rates: list[float] = []
        available: list[bool] = []
        for offset in range(self.horizon + 1):
            tt = t + offset
            rate = self._read_scalar(self.rate_dataset_key, tt, link_id)
            los = self._read_scalar(self.los_dataset_key, tt, link_id) if self._has_los else None
            rates.append(self._clean_rate(rate))
            available.append(self._available_from(rate, los))
        rates_norm = None
        if self._normalizer is not None:
            link_type = self._edge_link_type[edge_id].value
            rates_norm = [self._normalizer.transform_scalar(rate, link_type) for rate in rates]
        return EdgeWindow(edge_id, rates, available, self._edge_link_type[edge_id], rates_norm)

    def get_all_edge_windows(self, t: int) -> dict[str, EdgeWindow]:
        """Return the (H+1, L) rate/LOS window for every edge.

        Rates are served from the in-memory chunk cache (whole compressed
        chunks are read and cached once), so a step never touches the H5 file.
        Cleaning, availability and ``log_p99`` normalization are applied with
        vectorized numpy ops per window block instead of per-element Python
        calls. The window dict is cached per ``t`` because ``step()`` builds
        the state twice per tick (before stepping and for the observation).
        """
        if self._window_t == t and self._window_windows is not None:
            return self._window_windows
        blocks = self.get_window_blocks(t)
        rates_clean, avail, rates_norm = blocks
        # One vectorized transpose-tolist pass instead of a per-column numpy
        # slice + tolist for every edge (4241 columns in the full scenario).
        col_rates = rates_clean.T.tolist()
        col_avail = avail.T.tolist()
        col_norm = rates_norm.T.tolist() if rates_norm is not None else None
        windows: dict[str, EdgeWindow] = {}
        for edge_id, link_id in self._edge_to_link.items():
            windows[edge_id] = EdgeWindow(
                edge_id,
                col_rates[link_id],
                col_avail[link_id],
                self._edge_link_type[edge_id],
                col_norm[link_id] if col_norm is not None else None,
            )
        self._window_t = t
        self._window_windows = windows
        return windows

    def get_window_blocks(self, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return ``(rates_clean, avail, rates_norm)`` blocks of shape ``(H+1, L)``.

        The heavy per-step work (H5 chunk reads, nan cleaning, availability,
        vectorized normalization) happens once per ``t`` here; ``EdgeWindow``
        objects are materialized lazily per edge on demand.
        """
        if self._blocks_t == t and self._blocks is not None:
            return self._blocks
        expected = self.horizon + 1
        if self.out_of_range_policy == "raise" and (t < 0 or t + expected > self._T):
            raise IndexError(f"t={t} + horizon={self.horizon} outside dataset range [0, {self._T})")
        t0 = min(max(t, 0), self._T)
        t1 = min(max(t + expected, 0), self._T)
        rates_block = self._read_dataset_rows(self.rate_dataset_key, t0, t1)
        if self.negative_rate_policy == "raise" and np.any(np.nan_to_num(rates_block, nan=0.0) < 0.0):
            raise ValueError("Negative rate encountered with rate.negative_rate_policy=raise.")
        rates_clean = np.where(np.isnan(rates_block), 0.0, np.maximum(rates_block, 0.0))
        rate_ok = rates_clean >= self.min_link_rate
        source = self.availability_source
        if self._has_los:
            los_block = self._read_dataset_rows(self.los_dataset_key, t0, t1)
            los_ok = los_block == 1
            if source == "los":
                avail = los_ok
            elif source == "rate":
                avail = rate_ok
            elif source == "los_and_rate":
                avail = los_ok & rate_ok
            else:
                raise ValueError(f"Unknown availability_source: {source!r}")
        else:
            # Lean H5 (k_max only): los is redundant because k_max=0 whenever
            # los=0, so with min_link_rate>0 availability reduces to rate>=min.
            avail = rate_ok
        rates_norm = self._normalize_block(rates_clean)
        # Pad short blocks (dataset end) according to out_of_range_policy.
        n = rates_clean.shape[0]
        if n < expected:
            pad = expected - n
            if self.out_of_range_policy == "pad_last" and n > 0:
                rates_clean = np.concatenate(
                    [rates_clean, np.repeat(rates_clean[-1:], pad, axis=0)], axis=0
                )
                avail = np.concatenate([avail, np.repeat(avail[-1:], pad, axis=0)], axis=0)
                if rates_norm is not None:
                    rates_norm = np.concatenate(
                        [rates_norm, np.repeat(rates_norm[-1:], pad, axis=0)], axis=0
                    )
            elif self.out_of_range_policy == "pad_last" and n == 0:
                # t is entirely past the dataset end; repeat the last row.
                last_rate_block = self._read_dataset_rows(self.rate_dataset_key, self._T - 1, self._T)
                if self.negative_rate_policy == "raise" and np.any(
                    np.nan_to_num(last_rate_block, nan=0.0) < 0.0
                ):
                    raise ValueError("Negative rate encountered with rate.negative_rate_policy=raise.")
                last_rate = np.where(np.isnan(last_rate_block), 0.0, np.maximum(last_rate_block, 0.0))
                last_rate_ok = last_rate >= self.min_link_rate
                if self._has_los:
                    last_los = self._read_dataset_rows(self.los_dataset_key, self._T - 1, self._T)
                    last_los_ok = last_los == 1
                    if source == "los":
                        last_avail = last_los_ok
                    elif source == "rate":
                        last_avail = last_rate_ok
                    else:
                        last_avail = last_los_ok & last_rate_ok
                else:
                    last_avail = last_rate_ok
                last_norm = self._normalize_block(last_rate)
                rates_clean = np.repeat(last_rate, expected, axis=0)
                avail = np.repeat(last_avail, expected, axis=0)
                rates_norm = np.repeat(last_norm, expected, axis=0) if last_norm is not None else None
            else:
                rates_clean = np.concatenate(
                    [rates_clean, np.zeros((pad, self._L), dtype=rates_clean.dtype)], axis=0
                )
                avail = np.concatenate([avail, np.zeros((pad, self._L), dtype=bool)], axis=0)
                if rates_norm is not None:
                    rates_norm = np.concatenate(
                        [rates_norm, np.zeros((pad, self._L), dtype=rates_norm.dtype)], axis=0
                    )
        self._blocks_t = t
        self._blocks = (rates_clean, avail, rates_norm)
        return self._blocks

    def edge_window_from_blocks(self, edge_id: str, blocks) -> EdgeWindow:
        """Materialize one ``EdgeWindow`` from a cached window block."""
        rates_clean, avail, rates_norm = blocks
        link_id = self._edge_to_link[edge_id]
        col_rate = rates_clean[:, link_id]
        col_avail = avail[:, link_id]
        col_norm = rates_norm[:, link_id] if rates_norm is not None else None
        return EdgeWindow(
            edge_id,
            col_rate.tolist(),
            col_avail.tolist(),
            self._edge_link_type[edge_id],
            col_norm.tolist() if col_norm is not None else None,
        )

    def edge_link_ids(self, edge_ids: list[str]) -> np.ndarray:
        """Link ids for the given edge ids (vectorized gather index)."""
        return np.fromiter(
            (self._edge_to_link[edge_id] for edge_id in edge_ids), dtype=np.int64, count=len(edge_ids)
        )

    # ---------- internals ----------
    def _resolve_name(self, node_id: str) -> str:
        return self.node_name_map.get(node_id, node_id)

    def _load_registries(self) -> None:
        if not self.node_registry_path.exists():
            raise FileNotFoundError(f"node_registry.csv not found: {self.node_registry_path}")
        node_rows: list[tuple[int, str, str]] = []
        with self.node_registry_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                node_rows.append((int(row["node_id"]), row["type"].strip().upper(), row["name"].strip()))
        id_to_name = {idx: name for idx, _t, name in node_rows}

        link_rows: list[tuple[int, int, int, str]] = []
        if self.link_registry_path.exists():
            with self.link_registry_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    link_rows.append(
                        (int(row["link_id"]), int(row["node_u"]), int(row["node_v"]), row["link_type"].strip().upper())
                    )
        else:
            import h5py

            with h5py.File(self.link_data_path, "r") as f:
                for item in f["link_registry"][:]:
                    link_rows.append(
                        (
                            int(item["link_id"]),
                            int(item["node_u"]),
                            int(item["node_v"]),
                            bytes(item["link_type"]).decode("utf-8").strip().upper(),
                        )
                    )

        self._pair_to_link = {}
        for link_id, u, v, link_type in link_rows:
            name_u = id_to_name.get(u)
            name_v = id_to_name.get(v)
            if name_u is None or name_v is None:
                raise ValueError(f"link {link_id} references unknown node ids {u}, {v}")
            if link_type not in H5_LINK_TYPE_MAP:
                raise ValueError(f"Unknown H5 link type {link_type!r} in link {link_id}")
            self._link_type_by_link[link_id] = H5_LINK_TYPE_MAP[link_type]
            self._pair_to_link[tuple(sorted((name_u, name_v)))] = link_id

    def _open(self):
        if self._file is None:
            import h5py

            self._file = h5py.File(self.link_data_path, "r")
            ds = self._file[self.rate_dataset_key]
            self._T, self._L = ds.shape
            self.slot_seconds = int(self._file.attrs.get("theta_sec", 60))
            self._has_los = self.los_dataset_key in self._file
            self._chunk_rows = None if ds.chunks is None else int(ds.chunks[0])
            self._chunk_caches = {}
            if not self._has_los and not self._los_warned:
                self._los_warned = True
                if self.availability_source == "los":
                    warnings.warn(
                        f"H5 {self.link_data_path.name} has no '{self.los_dataset_key}' dataset; "
                        "availability falls back to rate>=min_link_rate."
                    )
        return self._file

    def _read_dataset_rows(self, dataset_key: str, r0: int, r1: int) -> np.ndarray:
        """Return rows ``[r0, r1)`` of a ``(T, L)`` dataset.

        Compressed chunked datasets are read whole-chunk and cached so that a
        single-row read does not re-decompress its containing chunk on every
        step (the old per-row reads cost ~70ms each on the full H5). Reads for
        contiguous (uncompressed) datasets are served directly.
        """
        r0 = min(max(r0, 0), self._T)
        r1 = min(max(r1, 0), self._T)
        if r1 <= r0:
            return np.asarray(self._file[dataset_key][0:0])
        if self._chunk_rows is None:
            return np.asarray(self._file[dataset_key][r0:r1])
        cache = self._chunk_caches.setdefault(dataset_key, OrderedDict())
        parts: list[np.ndarray] = []
        for chunk_idx in range(r0 // self._chunk_rows, (r1 - 1) // self._chunk_rows + 1):
            chunk = cache.pop(chunk_idx, None)
            if chunk is None:
                c0 = chunk_idx * self._chunk_rows
                c1 = min(c0 + self._chunk_rows, self._T)
                chunk = np.asarray(self._file[dataset_key][c0:c1])
                cache[chunk_idx] = chunk
                while len(cache) > self._chunk_cache_size:
                    cache.popitem(last=False)
            else:
                cache[chunk_idx] = chunk  # refresh LRU order
            lo = max(r0, chunk_idx * self._chunk_rows) - chunk_idx * self._chunk_rows
            hi = min(r1, (chunk_idx + 1) * self._chunk_rows) - chunk_idx * self._chunk_rows
            parts.append(chunk[lo:hi])
        return np.concatenate(parts, axis=0)

    def _prepare_normalization(self) -> None:
        """Precompute the per-column ``1 / log1p(p99)`` multiplier once.

        Rate normalization depends only on the H5 rates and the calibrated p99
        stats, so the per-step Python-level ``transform_scalar`` loop in
        ``GraphBuilder`` is replaced by a single vectorized ``log1p`` multiply.
        """
        self._inv_log1p_p99 = None
        if self._normalizer is None or self._normalizer.mode != "log_p99":
            return
        inv = np.ones(self._L, dtype=np.float32)
        for link_id, link_type in self._link_type_by_link.items():
            if 0 <= link_id < self._L:
                inv[link_id] = self._normalizer.inv_log1p_p99(link_type.value)
        self._inv_log1p_p99 = inv

    def _normalize_block(self, rates_clean: np.ndarray) -> np.ndarray | None:
        if self._inv_log1p_p99 is None:
            return None
        return np.log1p(rates_clean) * self._inv_log1p_p99[None, :]

    def _read_scalar(self, dataset_key: str, t: int, link_id: int) -> float:
        if t < 0 or t >= self._T:
            if self.out_of_range_policy == "raise":
                raise IndexError(f"t={t} outside dataset range [0, {self._T})")
            if self.out_of_range_policy == "pad_last":
                return float(self._read_dataset_rows(dataset_key, self._T - 1, self._T)[0, link_id])
            return 0.0
        return float(self._read_dataset_rows(dataset_key, t, t + 1)[0, link_id])

    def _clean_rate(self, value: float) -> float:
        value = float(value)
        if math.isnan(value):
            if self.config.get("rate", {}).get("unavailable_if_nan", True):
                return 0.0
            return value
        rate_policy = self.config.get("rate", {}).get("negative_rate_policy", "clip_to_zero")
        if rate_policy == "raise" and value < 0:
            raise ValueError(f"Negative rate {value} with rate.negative_rate_policy=raise.")
        return max(0.0, value)

    def _available_from(self, rate: float, los: float | None) -> bool:
        src = self.availability_source
        rate_ok = self._clean_rate(rate) >= self.min_link_rate
        if not self._has_los or los is None:
            return rate_ok
        los_ok = int(los) == 1
        if src == "los":
            return los_ok
        if src == "rate":
            return rate_ok
        if src == "los_and_rate":
            return los_ok and rate_ok
        raise ValueError(f"Unknown availability_source: {src!r}")


def build_rate_provider(
    config: dict, edges: list[Edge], seed: int, normalizer: RateNormalizer | None = None
) -> RateProvider:
    provider_name = config["rate_provider"]["provider"]
    if provider_name != "h5":
        raise ValueError(f"Rate provider {provider_name!r} is not supported; training reads the H5 dataset (use 'h5').")
    provider_cfg = config["rate_provider"]
    horizon = int(config["features"]["edge"]["prediction_horizon"])
    min_link_rate = float(provider_cfg["rate"]["min_link_rate"])
    if normalizer is None:
        normalizer = RateNormalizer(provider_cfg.get("normalization", {}))
    provider = H5RateProvider(provider_cfg, seed=seed)
    provider.setup(edges, horizon=horizon, min_link_rate=min_link_rate, normalizer=normalizer)
    return provider
