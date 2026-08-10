from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NodeType(str, Enum):
    GS = "gs"
    HAP = "hap"
    SAT = "sat"


class LinkType(str, Enum):
    GS_HAP = "gs_hap"
    GS_SAT = "gs_sat"
    HAP_SAT = "hap_sat"
    SAT_SAT = "sat_sat"


H5_LINK_TYPE_MAP = {
    "HAP-GS": LinkType.GS_HAP,
    "SAT-GS": LinkType.GS_SAT,
    "SAT-HAP": LinkType.HAP_SAT,
    "SAT-SAT": LinkType.SAT_SAT,
}


@dataclass(frozen=True)
class Node:
    node_id: str
    node_type: NodeType
    lat: float | None = None
    lon: float | None = None
    alt_m: float | None = None


@dataclass(frozen=True)
class Edge:
    edge_id: str
    src: str
    dst: str
    link_type: LinkType

    def other(self, node_id: str) -> str:
        if node_id == self.src:
            return self.dst
        if node_id == self.dst:
            return self.src
        raise ValueError(f"Node {node_id!r} is not incident to edge {self.edge_id!r}.")


@dataclass(frozen=True)
class KeyRequest:
    request_id: str
    src_gs: str
    dst_gs: str
    amount: float
    arrival_t: int
    deadline_t: int
    priority: float = 1.0
    # Key amount already served in previous partial-service steps; the
    # remaining demand is ``amount - served_amount``.
    served_amount: float = 0.0

