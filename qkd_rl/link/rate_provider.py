from __future__ import annotations

import csv
import math
import zlib
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


class RateNormalizer:
    def __init__(self, config: dict):
        self.config = config
        self.mode = config.get("mode", "identity")
        self.p99 = float(config.get("p99", 10.0))
        self.eps = float(config.get("eps", 1.0e-8))

    def transform_scalar(self, value: float) -> float:
        if self.mode == "log_p99":
            return math.log1p(max(value, 0.0)) / max(math.log1p(self.p99), self.eps)
        return value


def _stable_hash(text: str) -> int:
    """Deterministic string hash (Python's hash() is salted per process)."""
    return zlib.crc32(text.encode("utf-8"))


class MockRateProvider:
    def __init__(self, config: dict, seed: int = 0):
        self.config = config
        self.seed = seed
        self.edges: dict[str, Edge] = {}
        self.horizon = 0
        self.min_link_rate = 0.0

    def setup(self, edges: list[Edge], horizon: int, min_link_rate: float) -> None:
        self.edges = {edge.edge_id: edge for edge in edges}
        self.horizon = horizon
        self.min_link_rate = min_link_rate

    def get_rate(self, edge_id: str, t: int) -> float:
        edge = self.edges[edge_id]
        base_by_type = {
            LinkType.GS_HAP: 1.8,
            LinkType.GS_SAT: 1.2,
            LinkType.HAP_SAT: 2.4,
            LinkType.SAT_SAT: 3.0,
        }
        base = base_by_type[edge.link_type]
        phase = (_stable_hash(edge_id) % 97) / 97.0
        daily = 0.5 + 0.5 * math.sin(2 * math.pi * ((t % 1440) / 1440.0 + phase))
        outage = 0.0 if ((t + _stable_hash(edge_id)) % 113) < 8 else 1.0
        return max(0.0, base * (0.2 + daily) * outage)

    def is_available(self, edge_id: str, t: int) -> bool:
        return self.get_rate(edge_id, t) >= self.min_link_rate

    def get_edge_window(self, edge_id: str, t: int) -> EdgeWindow:
        rates = [self.get_rate(edge_id, t + offset) for offset in range(self.horizon + 1)]
        return EdgeWindow(
            edge_id=edge_id,
            rates=rates,
            available=[rate >= self.min_link_rate for rate in rates],
            link_type=self.edges[edge_id].link_type,
        )

    def get_all_edge_windows(self, t: int) -> dict[str, EdgeWindow]:
        return {edge_id: self.get_edge_window(edge_id, t) for edge_id in self.edges}


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

    # ---------- lifecycle ----------
    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def time_bounds(self) -> tuple[int, int]:
        return (0, self._T)

    def setup(self, edges: list[Edge], horizon: int, min_link_rate: float) -> None:
        self.edges = {edge.edge_id: edge for edge in edges}
        self.horizon = horizon
        self.min_link_rate = min_link_rate
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

    # ---------- RateProvider interface ----------
    def get_rate(self, edge_id: str, t: int) -> float:
        link_id = self._edge_to_link[edge_id]
        return self._clean_rate(self._read_scalar(self.rate_dataset_key, t, link_id))

    def is_available(self, edge_id: str, t: int) -> bool:
        link_id = self._edge_to_link[edge_id]
        rate = self._read_scalar(self.rate_dataset_key, t, link_id)
        los = self._read_scalar(self.los_dataset_key, t, link_id)
        return self._available_from(rate, los)

    def get_edge_window(self, edge_id: str, t: int) -> EdgeWindow:
        link_id = self._edge_to_link[edge_id]
        rates: list[float] = []
        available: list[bool] = []
        for offset in range(self.horizon + 1):
            tt = t + offset
            rate = self._read_scalar(self.rate_dataset_key, tt, link_id)
            los = self._read_scalar(self.los_dataset_key, tt, link_id)
            rates.append(self._clean_rate(rate))
            available.append(self._available_from(rate, los))
        return EdgeWindow(edge_id, rates, available, self._edge_link_type[edge_id])

    def get_all_edge_windows(self, t: int) -> dict[str, EdgeWindow]:
        """Vectorized batched read: one (H+1, L) slice for rate and LOS."""
        h5 = self._open()
        expected = self.horizon + 1
        t0 = min(max(t, 0), self._T)
        t1 = min(max(t + expected, 0), self._T)
        rates_block = h5[self.rate_dataset_key][t0:t1, :]
        los_block = h5[self.los_dataset_key][t0:t1, :]
        n = rates_block.shape[0]
        windows: dict[str, EdgeWindow] = {}
        for edge_id, link_id in self._edge_to_link.items():
            rates: list[float] = []
            available: list[bool] = []
            for i in range(expected):
                if i < n:
                    rate = rates_block[i, link_id]
                    los = los_block[i, link_id]
                else:
                    rate = 0.0
                    los = 0
                rates.append(self._clean_rate(rate))
                available.append(self._available_from(rate, los))
            windows[edge_id] = EdgeWindow(edge_id, rates, available, self._edge_link_type[edge_id])
        return windows

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
            self._pair_to_link[tuple(sorted((name_u, name_v)))] = link_id

    def _open(self):
        if self._file is None:
            import h5py

            self._file = h5py.File(self.link_data_path, "r")
            ds = self._file[self.rate_dataset_key]
            self._T, self._L = ds.shape
            self.slot_seconds = int(self._file.attrs.get("theta_sec", 60))
        return self._file

    def _read_scalar(self, dataset_key: str, t: int, link_id: int) -> float:
        if t < 0 or t >= self._T:
            if self.out_of_range_policy == "raise":
                raise IndexError(f"t={t} outside dataset range [0, {self._T})")
            if self.out_of_range_policy == "pad_last":
                return float(self._open()[dataset_key][self._T - 1, link_id])
            return 0.0
        return float(self._open()[dataset_key][t, link_id])

    def _clean_rate(self, value: float) -> float:
        value = float(value)
        if math.isnan(value):
            if self.config.get("rate", {}).get("unavailable_if_nan", True):
                return 0.0
            return value
        return max(0.0, value)

    def _available_from(self, rate: float, los: float) -> bool:
        src = self.availability_source
        los_ok = int(los) == 1
        rate_ok = self._clean_rate(rate) >= self.min_link_rate
        if src == "los":
            return los_ok
        if src == "rate":
            return rate_ok
        if src == "los_and_rate":
            return los_ok and rate_ok
        raise ValueError(f"Unknown availability_source: {src!r}")


def build_rate_provider(config: dict, edges: list[Edge], seed: int) -> RateProvider:
    provider_name = config["rate_provider"]["provider"]
    provider_cfg = config["rate_provider"]
    horizon = int(config["features"]["edge"]["prediction_horizon"])
    min_link_rate = float(provider_cfg["rate"]["min_link_rate"])
    if provider_name == "mock":
        provider = MockRateProvider(provider_cfg, seed=seed)
    elif provider_name == "h5":
        provider = H5RateProvider(provider_cfg, seed=seed)
    else:
        raise NotImplementedError(f"Rate provider {provider_name!r} is not implemented yet.")
    provider.setup(edges, horizon=horizon, min_link_rate=min_link_rate)
    return provider